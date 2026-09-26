"""Versioned execution options shared by the CLI and ROS adapters."""
from .config_contract import fields, identifier, number, text, version


def enabled_component(data, name, required=(), optional=()):
    fields(data, ('schema_version', 'enabled'), tuple(required) + tuple(optional), path=name)
    version(data['schema_version'], name)
    if type(data['enabled']) is not bool:
        raise ValueError(f'{name}.enabled: expected boolean')
    if data['enabled']:
        missing = set(required) - data.keys()
        if missing:
            raise ValueError(f'{name}: missing required fields {sorted(missing)}')
    return data['enabled']


def validate_moveit(data):
    required = ('groups',)
    optional = ('default_group', 'acceleration_limits', 'velocity_scaling', 'acceleration_scaling', 'planning_time',
                'planner_id', 'position_tolerance', 'orientation_tolerance', 'saved_trajectories_dir')
    if not enabled_component(data, 'moveit', required, optional):
        return
    groups = data['groups']
    if not isinstance(groups, dict) or not groups:
        raise ValueError('moveit.groups: expected nonempty mapping')
    for name, group in groups.items():
        identifier(name, 'moveit.groups')
        fields(group, ('joints', 'base_link', 'tip_link', 'controller'), ('kinematics_solver',), path=name)
        if not isinstance(group['joints'], list) or not group['joints']:
            raise ValueError(f'moveit.groups.{name}.joints: expected nonempty joint list')
        for joint in group['joints']:
            identifier(joint, name + '.joints')
        if len(set(group['joints'])) != len(group['joints']):
            raise ValueError(f'moveit.groups.{name}.joints: duplicates')
        for key in ('base_link', 'tip_link', 'controller'):
            identifier(group[key], name + '.' + key)
        text(group.get('kinematics_solver', 'kdl_kinematics_plugin/KDLKinematicsPlugin'), name + '.kinematics_solver')
    default_group = data.get('default_group', next(iter(groups)))
    if not isinstance(default_group, str) or default_group not in groups:
        raise ValueError('moveit.default_group: expected a configured group')
    if data.get('planner_id', 'RRTConnectkConfigDefault') != 'RRTConnectkConfigDefault':
        raise ValueError('moveit.planner_id: supported planner is RRTConnectkConfigDefault')
    if 'saved_trajectories_dir' in data:
        text(data['saved_trajectories_dir'], 'moveit.saved_trajectories_dir')
    for key in ('velocity_scaling', 'acceleration_scaling', 'planning_time',
                'position_tolerance', 'orientation_tolerance'):
        if key in data:
            number(data[key], 'moveit.' + key, positive=True)
            if key.endswith('scaling') and data[key] > 1:
                raise ValueError('moveit.' + key + ': must not exceed 1')
    accelerations = data.get('acceleration_limits', {})
    if not isinstance(accelerations, dict):
        raise ValueError('moveit.acceleration_limits: expected mapping')
    for name, value in accelerations.items():
        identifier(name, 'moveit.acceleration_limits')
        number(value, 'moveit.acceleration_limits.' + name, positive=True)


def validate_benchmark(data):
    enabled_component(data, 'benchmark', ('robot',), ('reference', 'tolerances', 'config'))
    if data.get('robot') is not None:
        identifier(data['robot'], 'benchmark.robot')
    elif data['enabled']:
        raise ValueError('benchmark.robot: required when enabled')
    for key in ('reference', 'config'):
        if key in data and data[key] is not None:
            text(data[key], 'benchmark.' + key)
    tolerances = data.get('tolerances', {})
    if not isinstance(tolerances, dict):
        raise ValueError('benchmark.tolerances: expected metric mapping')
    for key, value in tolerances.items():
        from .benchmark_observations import CHANNELS
        if key not in set(CHANNELS) | {'duration', 'energy', 'effort'}:
            raise ValueError(f'unknown tolerance channel: {key}')
        if number(value, 'benchmark.tolerances.' + key) < 0:
            raise ValueError('benchmark tolerances must be nonnegative')


def validate_trajectories(data):
    if enabled_component(data, 'trajectories', ('directory',)):
        text(data['directory'], 'trajectories.directory')
