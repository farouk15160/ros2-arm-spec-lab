"""End-to-end public CLI checks with real MuJoCo and manufacturer FK evidence."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from arm_lab_model.pipeline_cli import main
from arm_lab_model.trajectory_store import load_trajectory, save_trajectory


CONFIG = Path(__file__).parents[1] / 'config/pipeline'


def test_ur5e_execution_profile_succeeds_without_claiming_a_hardware_benchmark(tmp_path, capsys):
    code, result = invoke(capsys, 'simulate', CONFIG / 'project_ur5e_sim.yaml',
                          '--scenario', CONFIG / 'scenario_ur5e.yaml', '--output', tmp_path)
    assert code == 0 and result['passed']
    assert result['max_contacts'] == result['saturated_steps'] == 0
    assert 'benchmark' not in result
    observation = json.loads((tmp_path / 'observation.json').read_text())
    assert observation['evidence']['kind'] == 'synthetic'


def fixture_project(tmp_path, floating=False):
    inertia = {'mass': 1., 'com': [0., 0., 0.], 'inertia': [.1, .1, .1, 0., 0., 0.], 'source': 'synthetic fixture'}
    robot = {'schema_version': 1, 'format': 'tree', 'name': 'cli_test', 'family': 'test', 'source': 'synthetic',
             'base': {'link': 'base', 'type': 'floating' if floating else 'fixed'},
             'end_effectors': {'tcp': 'arm', 'base_target': 'base'},
             'links': [{'name': 'base', 'inertial': inertia}, {'name': 'arm', 'inertial': deepcopy(inertia)}],
             'joints': [{'name': 'hinge', 'type': 'revolute', 'parent': 'base', 'child': 'arm',
                         'origin': {'xyz': [0., 0., 1.], 'rpy': [0., 0., 0.]}, 'axis': [0., 0., 1.],
                         'limits': {'lower': -3., 'upper': 3., 'velocity': 2., 'acceleration': 4., 'effort': 100.}}]}
    components = {'robot': robot,
                  'simulation': {'schema_version': 1, 'backend': 'mujoco', 'timestep': .001,
                                 'gravity': [0., 0., 0.], 'seed': 0},
                  'trajectories': {'schema_version': 1, 'enabled': True, 'directory': 'custom_saved'}}
    for name, value in components.items():
        (tmp_path / (name + '.yaml')).write_text(yaml.safe_dump(value))
    project = tmp_path / 'project.yaml'
    project.write_text(yaml.safe_dump({'schema_version': 1, **{name: name + '.yaml' for name in components}}))
    scenario = tmp_path / 'scenario.yaml'
    scenario.write_text(yaml.safe_dump({'schema_version': 1, 'scenario_id': 'hold', 'joint_names': ['hinge'],
        'end_effector': 'base_target', 'acceleration_limits': {'hinge': .5},
        'points': [{'time_from_start': 0., 'positions': [0.]}, {'time_from_start': .01, 'positions': [0.]}]}))
    return project, scenario


def invoke(capsys, *args):
    code = main(list(map(str, args)))
    return code, json.loads(capsys.readouterr().out)


def test_simulate_save_replay_preserves_policy_and_endpoint(tmp_path, capsys):
    project, scenario = fixture_project(tmp_path)
    output = tmp_path / 'run'
    code, result = invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', output, '--save')
    assert code == 0 and result['passed']
    record = load_trajectory(result['saved_trajectory'])
    assert Path(result['saved_trajectory']).parent.parent == tmp_path / 'custom_saved'
    assert record['planner']['parameters']['end_effector'] == 'base'
    code, replay = invoke(capsys, 'replay', project, '--trajectory', result['saved_trajectory'], '--output', tmp_path / 'replay')
    assert code == 0 and replay['passed']
    replay_record = json.loads((tmp_path / 'replay/execution.json').read_text())['trajectory']
    assert replay_record['planner']['parameters']['acceleration_limits'] == {'hinge': .5}
    assert replay_record['planner']['parameters']['end_effector'] == 'base'


def test_missing_reference_remains_incomplete_and_time_is_not_missing_channel(tmp_path, capsys):
    project, scenario = fixture_project(tmp_path)
    code, result = invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', tmp_path / 'run', '--benchmark')
    assert code == 1 and result['benchmark']['status'] == 'incomplete'
    assert 'time' not in result['benchmark']['missing_channels']


@pytest.mark.parametrize('points', [[], [{}], [{'positions': [0.]}, {'positions': [0.]}], None])
def test_malformed_scenario_returns_friendly_error(tmp_path, capsys, points):
    project, scenario = fixture_project(tmp_path)
    value = yaml.safe_load(scenario.read_text())
    scenario.write_text(yaml.safe_dump({**value, 'points': points}))
    code, result = invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', tmp_path / 'run')
    assert code == 2 and 'error' in result


def test_build_analyze_and_actual_ur5e_reference(tmp_path, capsys):
    project = CONFIG / 'project_ur5e.yaml'
    code, result = invoke(capsys, 'build', project, '--output', tmp_path / 'build')
    assert code == 0 and {'robot.urdf', 'robot.xml', 'moveit'} <= set(result['artifacts'])
    code, result = invoke(capsys, 'analyze', project, '--output', tmp_path / 'analysis')
    assert code == 0 and result['total_mass'] > 10
    code, result = invoke(capsys, 'reference-check', project, '--output', tmp_path / 'reference')
    assert code == 0 and result['passed'] and result['case_count'] == 5
    assert result['hardware_validation'] is False
    assert (tmp_path / 'reference/reference_check.json').is_file()


def test_compare_simulation_fixture_and_reject_mismatched_reference(tmp_path, capsys):
    project, scenario = fixture_project(tmp_path)
    invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', tmp_path / 'run')
    trace = tmp_path / 'run/observation.json'
    tolerance = tmp_path / 'tolerances.yaml'
    tolerance.write_text('joint_position: 0.001\n')
    code, result = invoke(capsys, 'compare', trace, trace, '--tolerances', tolerance, '--output', tmp_path / 'compare')
    assert code == 0 and result['status'] == 'passed'
    wrong = tmp_path / 'wrong.json'
    wrong.write_text(json.dumps({**json.loads(trace.read_text()), 'scenario_id': 'wrong'}))
    code, result = invoke(capsys, 'compare', trace, wrong, '--tolerances', tolerance, '--output', tmp_path / 'bad')
    assert code == 2 and 'scenario_id' in result['error']


def test_replay_requires_and_restores_floating_start(tmp_path, capsys):
    project, scenario = fixture_project(tmp_path, floating=True)
    code, result = invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', tmp_path / 'run', '--save')
    assert code == 0
    record = load_trajectory(result['saved_trajectory'])
    assert 'base_start_state' in record
    changed = {**record, 'base_start_state': {'qpos': [1., 2., 3., 1., 0., 0., 0.], 'qvel': [.1, 0., 0., 0., 0., 0.]}}
    stored = save_trajectory(changed, tmp_path / 'altered')
    code, _ = invoke(capsys, 'replay', project, '--trajectory', stored, '--output', tmp_path / 'replay')
    assert code == 0
    replay = json.loads((tmp_path / 'replay/execution.json').read_text())
    assert replay['trajectory']['base_start_state'] == changed['base_start_state']
    assert replay['observation']['samples']['tcp_position'][0] == [1., 2., 3.]
    missing = save_trajectory({key: value for key, value in record.items() if key != 'base_start_state'}, tmp_path / 'missing')
    code, result = invoke(capsys, 'replay', project, '--trajectory', missing, '--output', tmp_path / 'bad')
    assert code == 2 and 'base_start_state' in result['error']


def test_replay_identity_and_benchmark_selection_are_rejected(tmp_path, capsys):
    project, scenario = fixture_project(tmp_path)
    _, result = invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', tmp_path / 'run', '--save')
    record = load_trajectory(result['saved_trajectory'])
    altered = save_trajectory({**record, 'model_sha256': 'a' * 64}, tmp_path / 'wrong')
    code, result = invoke(capsys, 'replay', project, '--trajectory', altered, '--output', tmp_path / 'replay')
    assert code == 2 and 'identity' in result['error']
    code, result = invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', tmp_path / 'benchmark',
                          '--benchmark', '--benchmark-robot', 'different')
    assert code == 2 and 'does not match' in result['error']


def test_explicit_reference_produces_report_but_without_tolerances_cannot_pass(tmp_path, capsys):
    project, scenario = fixture_project(tmp_path)
    invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', tmp_path / 'first')
    code, result = invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', tmp_path / 'second',
                          '--reference', tmp_path / 'first/observation.json')
    assert code == 1 and result['benchmark']['status'] == 'incomplete'
    assert result['benchmark']['metrics']['joint_position']['max'] == 0.
    assert (tmp_path / 'second/benchmark.md').is_file()


def test_reference_check_without_catalogue_is_explicit_error(tmp_path, capsys):
    project, _ = fixture_project(tmp_path)
    code, result = invoke(capsys, 'reference-check', project, '--output', tmp_path / 'reference')
    assert code == 2 and 'benchmark.config' in result['error']


@pytest.mark.parametrize('option', ['robot', 'file'])
def test_invalid_benchmark_configuration_rejected_before_motion_validation(tmp_path, capsys, option):
    project, scenario = fixture_project(tmp_path)
    data = yaml.safe_load(scenario.read_text())
    # This command would fail joint-limit validation if execution were attempted.
    data['points'][1]['positions'] = [100.]
    scenario.write_text(yaml.safe_dump(data))
    extra = ['--benchmark', '--benchmark-robot', 'wrong'] if option == 'robot' else ['--reference', tmp_path / 'missing.yaml']
    code, result = invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', tmp_path / 'run', *extra)
    assert code == 2
    assert ('does not match' if option == 'robot' else 'missing.yaml') in result['error']


def test_benchmark_preflight_checks_tcp_identity_and_scenario(tmp_path, capsys):
    from arm_lab_model.pipeline_cli import prepare_benchmark
    from arm_lab_model.project_config import load_project
    project, scenario = fixture_project(tmp_path)
    invoke(capsys, 'simulate', project, '--scenario', scenario, '--output', tmp_path / 'run')
    reference = tmp_path / 'run/observation.json'
    config = load_project(project)
    assert prepare_benchmark(config, reference=reference, expected={'scenario_id': 'hold'})['robot'] == 'cli_test'
    with pytest.raises(ValueError, match='end_effector'):
        prepare_benchmark(config, reference=reference, expected={'end_effector': 'arm'})
    with pytest.raises(ValueError, match='scenario_id'):
        prepare_benchmark(config, reference=reference, expected={'scenario_id': 'different'})
