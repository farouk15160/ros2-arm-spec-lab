"""Build, inspect, execute and benchmark a versioned robot project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config_contract import fields, number, read_yaml, text, vector, version
from .physical_robot import plain, physical_report, resolve_robot, robot_fingerprint
from .project_config import load_project


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _context(path):
    from .scene_config import resolve_environment, resolve_sensors
    project = load_project(path)
    robot = resolve_robot(project)
    environment, sensors = resolve_environment(project), resolve_sensors(project)
    return project, robot, environment, sensors


def build_project(path, directory):
    """Write inspectable models/configs from one physical tree; never launches ROS."""
    from .robot_export import build_robot_urdf, build_robot_mjcf
    project, robot, environment, sensors = _context(path)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    simulation = plain(project.sections.get('simulation', {}))
    urdf = build_robot_urdf(robot, sensors=sensors)
    (directory / 'robot.urdf').write_text(urdf)
    (directory / 'robot.xml').write_text(build_robot_mjcf(robot, simulation=simulation,
                                                        environment=environment, sensors=sensors))
    _write_json(directory / 'robot.resolved.json', robot)
    _write_json(directory / 'physics.json', physical_report(robot, gravity=simulation.get('gravity', [0,0,-9.81])))
    if project.sections.get('moveit', {}).get('enabled'):
        from arm_lab_kinematics.pipeline_moveit import write_moveit_config
        write_moveit_config(robot, plain(project.sections['moveit']), directory / 'moveit')
    result = {'robot': robot['name'], 'model_sha256': robot_fingerprint(robot),
              'directory': str(directory), 'artifacts': sorted(p.name for p in directory.iterdir())}
    _write_json(directory / 'manifest.json', result)
    return result


def prepare_benchmark(project, *, reference=None, robot=None, expected=None):
    """Validate static benchmark inputs before motion; usable at ROS startup.

    Returns a validated reference observation or None when no trace is declared.
    expected optionally supplies scenario_id, frame, units, joint_types and
    end_effector known by the caller. Robot and joint order always match project.
    """
    from .benchmark_observations import load_observation
    config = plain(project.sections.get('benchmark', {}))
    selected = robot or config.get('robot') or project.robot_name
    if selected != project.robot_name:
        raise ValueError(f'benchmark robot {selected} does not match simulated robot {project.robot_name}')
    source = reference or config.get('reference')
    if not source:
        return None
    base = project.files.get('benchmark', project.source_path).parent
    ref = load_observation((base / source).resolve() if not reference else Path(reference).resolve())
    required = {**(expected or {}), 'robot': project.robot_name, 'joint_names': list(project.joint_names)}
    for key, value in required.items():
        actual = ref.get(key, ['revolute'] * len(project.joint_names) if key == 'joint_types' else None)
        if actual != value:
            raise ValueError(f'benchmark reference incompatible {key}: expected {value!r}')
    return ref


def benchmark_execution(project, observation, directory, *, reference=None, robot=None, reference_data=None):
    """Always emit an explicit unavailable report when no matching real trace exists."""
    from .benchmark_engine import compare_runs
    from .benchmark_reports import write_benchmark_report
    config = plain(project.sections.get('benchmark', {}))
    ref = reference_data if reference_data is not None else prepare_benchmark(project, reference=reference, robot=robot)
    if ref is not None:
        result = compare_runs(observation, ref, config.get('tolerances', {}))
        write_benchmark_report(result, directory, observation, ref)
        return result
    result = {'schema_version': 1, 'robot': project.robot_name, 'scenario_id': observation['scenario_id'],
              'status': 'incomplete', 'reason': 'No measured/reference trajectory configured.',
              'evidence': {'simulation': observation['evidence'], 'reference': None},
              'metrics': {}, 'missing_channels': sorted(set(observation['samples']) - {'time'}),
              'limitations': ['Manufacturer specifications are not recorded motion traces.']}
    _write_json(Path(directory) / 'benchmark.json', result)
    Path(directory, 'benchmark.md').write_text(
        f'# {project.robot_name} benchmark\n\nStatus: **incomplete**\n\n'
        'No reference trajectory is configured. No real-robot agreement has been established.\n')
    return result


def _scenario(path, names):
    data = read_yaml(path)
    fields(data, ('schema_version', 'scenario_id', 'joint_names', 'points'),
           ('acceleration_limits', 'end_effector'), path='scenario')
    version(data['schema_version'], 'scenario')
    text(data['scenario_id'], 'scenario.scenario_id')
    if data['joint_names'] != list(names):
        raise ValueError('scenario joint_names must match the model order')
    if 'end_effector' in data:
        text(data['end_effector'], 'scenario.end_effector')
    if not isinstance(data.get('acceleration_limits', {}), dict):
        raise ValueError('scenario.acceleration_limits must be a mapping')
    points = data['points']
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError('scenario.points must contain at least two trajectory points')
    for point in points:
        fields(point, ('time_from_start', 'positions'), ('velocities', 'accelerations'), path='scenario.point')
        number(point['time_from_start'], 'scenario.point.time_from_start')
        for key in ('positions', 'velocities', 'accelerations'):
            if key in point:
                vector(point[key], 'scenario.point.' + key, len(names))
    return data


def _restore_base(sim, record):
    if sim.robot['base']['type'] != 'floating':
        if 'base_start_state' in record:
            raise ValueError('fixed-base replay cannot use base_start_state')
        return
    if 'base_start_state' not in record:
        raise ValueError('floating replay requires recorded base_start_state')
    base = sim.m.joint('floating_base')
    qindex, vindex = int(base.qposadr[0]), int(base.dofadr[0])
    sim.d.qpos[qindex:qindex + 7] = record['base_start_state']['qpos']
    sim.d.qvel[vindex:vindex + 6] = record['base_start_state']['qvel']
    sim.mj.mj_forward(sim.m, sim.d)


def _storage_directory(project, directory):
    configured = project.sections.get('trajectories', {}).get('directory')
    if configured:
        return (project.files['trajectories'].parent / configured).resolve()
    return directory / 'saved_trajectories'


def execute_project(path, scenario, directory, *, reference=None, benchmark=False,
                    benchmark_robot=None, replay=None, save=False):
    from .pipeline_runtime import RobotSimulation, run_trajectory
    from .trajectory_store import load_trajectory, save_trajectory
    project, robot, environment, sensors = _context(path)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    simulation = plain(project.sections.get('simulation', {}))
    sim = RobotSimulation(robot, simulation, environment, sensors)
    acceleration = plain(project.sections.get('moveit', {}).get('acceleration_limits', {}))
    end_effector = None
    if replay:
        record = load_trajectory(replay)
        if record['robot'] != robot['name'] or record['model_sha256'] != sim.model_hash or record['joint_names'] != list(sim.names):
            raise ValueError('replay model identity or joint order mismatch')
        points = record['trajectory']['points']
        scenario_id = record['scenario_id']
        parameters = record['planner']['parameters']
        saved_limits = parameters.get('acceleration_limits', {})
        if not isinstance(saved_limits, dict):
            raise ValueError('replay planner acceleration_limits must be a mapping')
        acceleration = {**acceleration, **saved_limits}
        end_effector = parameters.get('end_effector')
        if end_effector is not None:
            text(end_effector, 'replay planner end_effector')
    else:
        data = _scenario(scenario, sim.names)
        acceleration = {**acceleration, **data.get('acceleration_limits', {})}
        points, scenario_id = data['points'], data['scenario_id']
        end_effector = data.get('end_effector')
    benchmark_enabled = benchmark or reference or project.sections.get('benchmark', {}).get('enabled')
    prepared_reference = None
    if benchmark_enabled:
        prepared_reference = prepare_benchmark(project, reference=reference, robot=benchmark_robot,
            expected={'scenario_id': scenario_id, 'frame': 'world', 'units': 'SI',
                      'joint_types': [joint['type'] for joint in sim.joints],
                      'end_effector': sim.endpoint(end_effector)})
    # An explicit scenario start initializes a fresh headless experiment; ROS replay never teleports.
    sim.set_state(points[0]['positions'])
    if replay:
        _restore_base(sim, record)
    result = run_trajectory(sim, points, scenario_id=scenario_id,
                            acceleration_limits=acceleration, end_effector=end_effector,
                            max_error=simulation.get('max_error', 0.05))
    _write_json(directory / 'observation.json', result['observation'])
    if benchmark_enabled:
        report = benchmark_execution(project, result['observation'], directory,
                                     reference=reference, robot=benchmark_robot,
                                     reference_data=prepared_reference)
        result = {**result, 'benchmark': report,
                  'trajectory': {**result['trajectory'], 'benchmark': report}}
    if save and result['passed']:
        record = result['trajectory']
        result = {**result, 'saved_trajectory': str(save_trajectory(record, _storage_directory(project, directory)))}
    _write_json(directory / 'execution.json', result)
    return {key: value for key, value in result.items() if key not in ('observation', 'trajectory')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('build', 'analyze', 'simulate', 'replay', 'reference-check'):
        sub = commands.add_parser(name)
        sub.add_argument('project')
        sub.add_argument('--output', '-o', default='pipeline_output')
        if name == 'simulate':
            sub.add_argument('--scenario', required=True)
        if name == 'replay':
            sub.add_argument('--trajectory', required=True)
        if name in ('simulate', 'replay'):
            sub.add_argument('--benchmark', action='store_true')
            sub.add_argument('--benchmark-robot')
            sub.add_argument('--reference')
            sub.add_argument('--save', action='store_true')
    compare = commands.add_parser('compare')
    compare.add_argument('simulation')
    compare.add_argument('reference')
    compare.add_argument('--tolerances', required=True, help='YAML mapping of metric names to absolute tolerances')
    compare.add_argument('--output', '-o', default='benchmark_output')
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        accepted = (result.get('passed', True) and result.get('status') not in ('failed', 'incomplete')
                    and result.get('benchmark', {}).get('status', 'passed') == 'passed')
        return 0 if accepted else 1
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(json.dumps({'error': str(exc), 'passed': False}))
        return 2


def _dispatch(args):
    if args.command == 'build':
        return build_project(args.project, args.output)
    if args.command == 'reference-check':
        from .benchmark_reference import check_reference_model
        project, robot, _, _ = _context(args.project)
        source = project.sections.get('benchmark', {}).get('config')
        if source is None:
            raise ValueError('reference-check requires benchmark.config catalogue path')
        catalogue = (project.files['benchmark'].parent / source).resolve()
        result = check_reference_model(robot, catalogue)
        _write_json(Path(args.output) / 'reference_check.json', result)
        return result
    if args.command == 'analyze':
        project, robot, _, _ = _context(args.project)
        result = physical_report(robot, gravity=project.sections.get('simulation', {}).get('gravity', [0,0,-9.81]))
        _write_json(Path(args.output) / 'physics.json', result)
        return result
    if args.command in ('simulate', 'replay'):
        return execute_project(args.project, getattr(args, 'scenario', None), args.output,
                               reference=args.reference, benchmark=args.benchmark,
                               benchmark_robot=args.benchmark_robot,
                               replay=getattr(args, 'trajectory', None), save=args.save)
    from .benchmark_engine import compare_runs
    from .benchmark_observations import load_observation
    from .benchmark_reports import write_benchmark_report
    sim, ref = load_observation(args.simulation), load_observation(args.reference)
    result = compare_runs(sim, ref, read_yaml(args.tolerances))
    write_benchmark_report(result, args.output, sim, ref)
    return {key: value for key, value in result.items() if key != 'series'}


if __name__ == '__main__':
    raise SystemExit(main())
