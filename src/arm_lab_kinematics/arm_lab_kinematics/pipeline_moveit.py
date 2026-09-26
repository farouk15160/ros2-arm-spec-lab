"""MoveIt configuration for any validated robot tree, without ROS imports.

Only adjacent links are excluded from self-collision checking. No sampled
"never collides" assumptions and no implicit acceleration ratings are used.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import yaml

from arm_lab_model.config_contract import identifier


_SOLVER = 'kdl_kinematics_plugin/KDLKinematicsPlugin'
_OPTION_KEYS = frozenset(('schema_version', 'enabled', 'groups', 'default_group',
                         'acceleration_limits', 'velocity_scaling',
                         'acceleration_scaling', 'planning_time', 'planner_id',
                         'position_tolerance', 'orientation_tolerance',
                         'saved_trajectories_dir'))


def _positive(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}: expected finite positive number')
    if not math.isfinite(value) or value <= 0 or (maximum and value > maximum):
        raise ValueError(f'{name}: expected finite positive number' +
                         (f' <= {maximum}' if maximum else ''))
    return float(value)


def validate_moveit_options(options):
    """Validate component structure without resolving the robot or importing ROS."""
    if not isinstance(options, Mapping) or set(options) - _OPTION_KEYS:
        raise ValueError('moveit: unknown keys or not a mapping')
    if type(options.get('schema_version')) is not int or options.get('schema_version') != 1 or type(options.get('enabled')) is not bool:
        raise ValueError('moveit: schema_version 1 and boolean enabled required')
    if not options['enabled']:
        return
    groups = options.get('groups')
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError('moveit.groups: expected nonempty mapping')
    for name, group in groups.items():
        identifier(name, 'moveit.groups')
        required = {'joints', 'base_link', 'tip_link', 'controller'}
        if not isinstance(group, Mapping) or not required <= set(group):
            raise ValueError(f'moveit.groups.{name}: joints, base_link, tip_link, controller required')
        if set(group) - (required | {'kinematics_solver'}):
            raise ValueError(f'moveit.groups.{name}: unknown keys')
        if not isinstance(group['joints'], (list, tuple)) or not group['joints']:
            raise ValueError(f'moveit.groups.{name}.joints: expected nonempty list')
        for value in tuple(group['joints']) + tuple(group[k] for k in ('base_link', 'tip_link', 'controller')):
            identifier(value, f'moveit.groups.{name}')
        if len(set(group['joints'])) != len(group['joints']):
            raise ValueError(f'moveit.groups.{name}.joints: duplicates')
        solver = group.get('kinematics_solver', _SOLVER)
        if not isinstance(solver, str) or not solver.strip():
            raise ValueError(f'moveit.groups.{name}.kinematics_solver: expected plugin name')
    if options.get('default_group', next(iter(groups))) not in groups:
        raise ValueError('moveit.default_group: unknown group')
    for key in ('velocity_scaling', 'acceleration_scaling'):
        _positive(options.get(key, 0.1), 'moveit.' + key, 1.0)
    for key, default in (('planning_time', 5.0), ('position_tolerance', 0.005),
                         ('orientation_tolerance', 0.01)):
        _positive(options.get(key, default), 'moveit.' + key)
    if options.get('planner_id', 'RRTConnectkConfigDefault') != 'RRTConnectkConfigDefault':
        raise ValueError('moveit.planner_id: supported planner is RRTConnectkConfigDefault')
    limits = options.get('acceleration_limits', {})
    if not isinstance(limits, Mapping):
        raise ValueError('moveit.acceleration_limits: expected mapping')
    for name, value in limits.items():
        identifier(name, 'moveit.acceleration_limits')
        _positive(value, 'moveit.acceleration_limits.' + name)
    if 'saved_trajectories_dir' in options and not isinstance(options['saved_trajectories_dir'], str):
        raise ValueError('moveit.saved_trajectories_dir: expected path string')


def _chain(robot, group):
    parents = {joint['child']: joint for joint in robot['joints']}
    links = {link['name'] for link in robot['links']}
    if group['base_link'] not in links or group['tip_link'] not in links:
        raise ValueError('moveit group references unknown link')
    node, result, visited = group['tip_link'], (), frozenset()
    while node != group['base_link']:
        if node in visited or node not in parents:
            raise ValueError('moveit group tip must descend from base in a single chain')
        visited = visited | {node}
        joint = parents[node]
        result = ((joint['name'],) if joint['type'] != 'fixed' else ()) + result
        node = joint['parent']
    if tuple(group['joints']) != result:
        raise ValueError('moveit group joints must match base-to-tip chain in order')


def controller_joints(options):
    """Controller name to union of group joints, preserving declaration order."""
    controllers = {}
    for group in options['groups'].values():
        name = group['controller']
        previous = controllers.get(name, ())
        controllers = {**controllers, name: previous + tuple(
            joint for joint in group['joints'] if joint not in previous)}
    owners = [joint for joints in controllers.values() for joint in joints]
    if len(owners) != len(set(owners)):
        raise ValueError('a joint cannot belong to multiple MoveIt controllers')
    return controllers


def _joint_limits(robot, options):
    overrides = options.get('acceleration_limits', {})
    moving = {joint['name']: joint for joint in robot['joints'] if joint['type'] != 'fixed'}
    if set(overrides) - set(moving):
        raise ValueError('moveit.acceleration_limits: unknown joint')
    result = {}
    for name, joint in moving.items():
        limits = joint['limits']
        velocity = _positive(limits.get('velocity'), name + '.velocity')
        accel = overrides.get(name, limits.get('acceleration'))
        if accel is None:
            raise ValueError(f'{name}: acceleration unknown; supply explicit moveit.acceleration_limits')
        accel = _positive(accel, name + '.acceleration')
        if limits.get('acceleration') is not None and accel > limits['acceleration']:
            raise ValueError(f'{name}: planning acceleration exceeds robot limit')
        result[name] = {'has_velocity_limits': True, 'max_velocity': velocity,
                        'has_acceleration_limits': True, 'max_acceleration': accel}
        if limits.get('effort') is not None:
            result[name] = {**result[name], 'has_effort_limits': True,
                            'max_effort': _positive(limits['effort'], name + '.effort')}
    return {'joint_limits': result}


def _srdf(robot, options):
    root = ET.Element('robot', name=robot['name'])
    for name, group in options['groups'].items():
        element = ET.SubElement(root, 'group', name=name)
        ET.SubElement(element, 'chain', base_link=group['base_link'], tip_link=group['tip_link'])
    # Fixed models already contain their explicit world mount in the unified URDF.
    if robot['base']['type'] == 'floating':
        ET.SubElement(root, 'virtual_joint', name='floating_base_joint', type='floating',
                      parent_frame='world', child_link=robot['base']['link'])
    for joint in robot['joints']:
        ET.SubElement(root, 'disable_collisions', link1=joint['parent'],
                      link2=joint['child'], reason='Adjacent')
    return ET.tostring(root, encoding='unicode')


def build_moveit_config(robot, options):
    """Generate MoveIt ROS parameters from unified geometry and explicit policies."""
    validate_moveit_options(options)
    if not options['enabled']:
        raise ValueError('moveit must be enabled to generate planning configuration')
    for group in options['groups'].values():
        _chain(robot, group)
    controllers = controller_joints(options)
    planning = {
        'planning_plugin': 'ompl_interface/OMPLPlanner',
        'request_adapters': ' '.join((
            'default_planner_request_adapters/AddTimeOptimalParameterization',
            'default_planner_request_adapters/FixWorkspaceBounds',
            'default_planner_request_adapters/FixStartStateBounds',
            'default_planner_request_adapters/FixStartStateCollision',
            'default_planner_request_adapters/FixStartStatePathConstraints')),
        'start_state_max_bounds_error': 0.1,
        'planner_configs': {'RRTConnectkConfigDefault': {'type': 'geometric::RRTConnect', 'range': 0.0}},
        **{name: {'planner_configs': ['RRTConnectkConfigDefault']} for name in options['groups']},
    }
    return {
        'robot_description_semantic': _srdf(robot, options),
        'robot_description_kinematics': {name: {
            'kinematics_solver': group.get('kinematics_solver', _SOLVER),
            'kinematics_solver_search_resolution': 0.005,
            'kinematics_solver_timeout': 0.1,
        } for name, group in options['groups'].items()},
        'robot_description_planning': _joint_limits(robot, options),
        'planning_pipelines': ['ompl'], 'default_planning_pipeline': 'ompl',
        'ompl': planning,
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        'moveit_simple_controller_manager': {
            'controller_names': list(controllers),
            **{name: {'type': 'FollowJointTrajectory', 'action_ns': 'follow_joint_trajectory',
                      'default': True, 'joints': list(joints)} for name, joints in controllers.items()},
        },
        'trajectory_execution': {'allowed_execution_duration_scaling': 1.5,
                                 'allowed_goal_duration_margin': 1.0,
                                 'allowed_start_tolerance': 0.01},
        'publish_robot_description': True, 'publish_robot_description_semantic': True,
        'publish_planning_scene': True, 'publish_geometry_updates': True,
        'publish_state_updates': True, 'publish_transforms_updates': True,
    }


def write_moveit_config(robot, options, directory):
    """Export standalone artifacts for inspection and downstream launch tooling."""
    generated = build_moveit_config(robot, options)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {'robot.srdf': generated['robot_description_semantic'],
             'kinematics.yaml': generated['robot_description_kinematics'],
             'joint_limits.yaml': generated['robot_description_planning'],
             'moveit_controllers.yaml': {key: value for key, value in generated.items()
                                         if key.startswith('moveit_')},
             'ompl_planning.yaml': generated['ompl']}
    for name, data in files.items():
        (directory / name).write_text(data if isinstance(data, str) else yaml.safe_dump(data), encoding='utf-8')
    return tuple(directory / name for name in files)
