"""Versioned project composition, independent of the legacy arm runtime."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

from .config import load_config
from .config_contract import choice, fields, identifier, number, read_yaml, text, vector, version
from .robot_topology import validate_robot


COMPONENTS = ('materials', 'simulation', 'moveit', 'environment', 'sensors',
              'perception', 'benchmark', 'trajectories')


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable declaration; loading never starts a runtime or writes assets."""

    source_path: Path
    files: Mapping
    sections: Mapping
    robot_name: str
    joint_names: tuple[str, ...]
    base_dof: int
    legacy_source: Path | None
    missing_parameters: tuple[str, ...] = ()


def _component(name, data):
    if name == 'simulation':
        fields(data, ('schema_version', 'backend', 'timestep', 'gravity', 'seed'),
               ('initial_positions', 'bandwidth_hz', 'max_error'), path=name)
        choice(data['backend'], ('mujoco', 'gazebo'), name + '.backend')
        number(data['timestep'], name + '.timestep', positive=True)
        vector(data['gravity'], name + '.gravity')
        if type(data['seed']) is not int or data['seed'] < 0:
            raise ValueError('simulation.seed: expected nonnegative integer')
        for key in ('bandwidth_hz', 'max_error'):
            if key in data:
                number(data[key], 'simulation.' + key, positive=True)
        if 'initial_positions' in data:
            if not isinstance(data['initial_positions'], list):
                raise ValueError('simulation.initial_positions: expected list')
            vector(data['initial_positions'], 'simulation.initial_positions', len(data['initial_positions']))
    elif name == 'materials':
        fields(data, ('schema_version', 'materials'), path=name)
        if not isinstance(data['materials'], dict):
            raise ValueError('materials.materials: expected mapping')
        for key, entry in data['materials'].items():
            identifier(key, 'materials.name')
            fields(entry, ('density', 'source'), path='materials.' + key)
            number(entry['density'], 'materials.' + key + '.density', positive=True)
            text(entry['source'], 'materials.' + key + '.source')
    else:
        from .pipeline_options import validate_benchmark, validate_moveit, validate_trajectories
        if name in ('moveit', 'benchmark', 'trajectories'):
            {'moveit': validate_moveit, 'benchmark': validate_benchmark,
             'trajectories': validate_trajectories}[name](data)
        elif name in ('environment', 'sensors'):
            from .scene_config import validate_environment, validate_sensors
            {'environment': validate_environment, 'sensors': validate_sensors}[name](data)
        elif name == 'perception':
            from .perception import validate_perception
            validate_perception(data)
    version(data['schema_version'], name)


def _robot_identity(robot, robot_path):
    if not isinstance(robot, dict):
        raise ValueError('robot: expected mapping')
    choice(robot.get('format'), ('tree', 'legacy_arm'), 'robot.format')
    if robot['format'] == 'tree':
        names, base_dof, missing = validate_robot(robot)
        return robot['name'], names, base_dof, None, missing
    fields(robot, ('schema_version', 'format', 'source'), path='robot')
    version(robot['schema_version'], 'robot')
    source = (robot_path.parent / text(robot['source'], 'robot.source')).resolve()
    read_yaml(source)  # Apply strict YAML syntax rules before the compatibility loader.
    try:
        legacy = load_config(str(source))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError(f'{source}: invalid legacy arm configuration: {exc}') from exc
    return legacy.name, tuple(legacy.joint_names), 0, source, ()


def load_project(path: str | Path) -> ProjectConfig:
    """Load component paths relative to the file that declares them."""
    path = Path(path).resolve()
    project = read_yaml(path)
    fields(project, ('schema_version', 'robot'), COMPONENTS, path='project')
    version(project['schema_version'], 'project')
    files = {name: (path.parent / text(project[name], 'project.' + name)).resolve()
             for name in ('robot',) + COMPONENTS if name in project}
    sections = {name: read_yaml(file) for name, file in files.items()}
    for name, data in sections.items():
        if name != 'robot':
            _component(name, data)
    identity = _robot_identity(sections['robot'], files['robot'])
    return ProjectConfig(path, _freeze(files), _freeze(sections), *identity)


def project_summary(project: ProjectConfig) -> dict:
    """Describe validation evidence without implying physics or runtime readiness."""
    return {
        'schema_version': 1, 'valid': True, 'runtime_ready': False,
        'scope': 'Configuration validation only; use robot_pipeline to resolve, export and execute.',
        'robot': project.robot_name, 'format': project.sections['robot']['format'],
        'joint_names': list(project.joint_names), 'joint_dof': len(project.joint_names),
        'base_velocity_dof': project.base_dof,
        'files': {key: str(path) for key, path in project.files.items()},
        'legacy_source': str(project.legacy_source) if project.legacy_source else None,
        'missing_parameters': list(project.missing_parameters),
        'limitations': [
            'The legacy arm commands retain their existing input format; use pipeline commands for projects.',
            'Configuration validation alone does not establish runtime or hardware readiness.',
            'Missing-parameter checks cover declared tree inertials and joint limits only.',
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Validate a project declaration; does not run a robot.')
    parser.add_argument('project', help='versioned project YAML manifest')
    args = parser.parse_args(argv)
    try:
        report = project_summary(load_project(args.project))
    except ValueError as exc:
        print(json.dumps({'schema_version': 1, 'valid': False, 'runtime_ready': False,
                          'error': str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
