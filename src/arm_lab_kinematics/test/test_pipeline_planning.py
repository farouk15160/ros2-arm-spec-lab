"""Public generated MoveIt configuration and target-contract tests."""
from copy import deepcopy
import xml.etree.ElementTree as ET

import pytest

from arm_lab_kinematics.pipeline_moveit import build_moveit_config


def robot_fixture():
    return {
        'name': 'two_joint_arm', 'base': {'type': 'fixed', 'link': 'base'},
        'links': [{'name': name} for name in ('base', 'upper', 'tool')],
        'joints': [
            {'name': 'shoulder', 'type': 'revolute', 'parent': 'base', 'child': 'upper',
             'limits': {'lower': -1.5, 'upper': 1.5, 'velocity': 2.0,
                        'acceleration': 3.0, 'effort': 10.0}},
            {'name': 'elbow', 'type': 'revolute', 'parent': 'upper', 'child': 'tool',
             'limits': {'lower': -2.0, 'upper': 2.0, 'velocity': 1.0,
                        'acceleration': 2.0, 'effort': 5.0}},
        ],
    }


def options_fixture():
    return {'schema_version': 1, 'enabled': True,
            'groups': {'arm': {'joints': ['shoulder', 'elbow'], 'base_link': 'base',
                               'tip_link': 'tool', 'controller': 'arm_controller',
                               'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin'}},
            'default_group': 'arm'}


def test_generated_config_uses_robot_limits_and_real_controller_endpoint():
    robot, options = robot_fixture(), options_fixture()
    before = deepcopy((robot, options))
    generated = build_moveit_config(robot, options)
    semantic = ET.fromstring(generated['robot_description_semantic'])
    assert semantic.find('group/chain').attrib == {'base_link': 'base', 'tip_link': 'tool'}
    assert generated['robot_description_planning']['joint_limits']['elbow']['max_acceleration'] == 2.0
    manager = generated['moveit_simple_controller_manager']
    assert manager['arm_controller']['joints'] == ['shoulder', 'elbow']
    assert manager['arm_controller']['action_ns'] == 'follow_joint_trajectory'
    assert (robot, options) == before


def test_acceleration_requires_explicit_policy_and_cannot_exceed_rating():
    robot, options = robot_fixture(), options_fixture()
    robot['joints'][0]['limits']['acceleration'] = None
    with pytest.raises(ValueError, match='acceleration unknown'):
        build_moveit_config(robot, options)
    options['acceleration_limits'] = {'shoulder': 0.5}
    assert build_moveit_config(robot, options)['robot_description_planning']['joint_limits']['shoulder']['max_acceleration'] == 0.5
    options['acceleration_limits'] = {'elbow': 10.0}
    robot['joints'][0]['limits']['acceleration'] = 1.0
    with pytest.raises(ValueError, match='exceeds robot limit'):
        build_moveit_config(robot, options)


def test_chain_rejects_joints_outside_declared_base_tip():
    options = options_fixture()
    options['groups']['arm']['joints'] = ['elbow', 'shoulder']
    with pytest.raises(ValueError, match='chain in order'):
        build_moveit_config(robot_fixture(), options)


def test_floating_robot_semantic_base_and_only_adjacent_collision_exclusions():
    robot = robot_fixture()
    robot['base']['type'] = 'floating'
    semantic = ET.fromstring(build_moveit_config(robot, options_fixture())['robot_description_semantic'])
    assert semantic.find('virtual_joint').attrib == {
        'name': 'floating_base_joint', 'type': 'floating', 'parent_frame': 'world', 'child_link': 'base'}
    pairs = {(item.attrib['link1'], item.attrib['link2']) for item in semantic.findall('disable_collisions')}
    assert pairs == {('base', 'upper'), ('upper', 'tool')}
    assert ('base', 'tool') not in pairs


@pytest.mark.parametrize('key,value', [('velocity_scaling', 0.0), ('acceleration_scaling', 1.1),
                                      ('planning_time', float('nan')), ('unknown', 1),
                                      ('planner_id', 'Nonexistent')])
def test_invalid_planning_options_fail_before_launch(key, value):
    options = {**options_fixture(), key: value}
    with pytest.raises(ValueError):
        build_moveit_config(robot_fixture(), options)


def test_export_writes_valid_standalone_yaml_and_srdf(tmp_path):
    import yaml
    from arm_lab_kinematics.pipeline_moveit import write_moveit_config
    paths = write_moveit_config(robot_fixture(), options_fixture(), tmp_path)
    assert len(paths) == 5
    assert ET.parse(tmp_path / 'robot.srdf').getroot().attrib['name'] == 'two_joint_arm'
    assert yaml.safe_load((tmp_path / 'joint_limits.yaml').read_text())['joint_limits']['shoulder']['max_velocity'] == 2.0


def test_target_validation_normalizes_quaternion_and_rejects_invalid_inputs():
    from arm_lab_kinematics.pipeline_target import validate_target
    assert validate_target([1, 2, 3], [0, 0, 0, 2]) == ((1., 2., 3.), (0., 0., 0., 1.))
    assert validate_target([1, 2, 3], None) == ((1., 2., 3.), None)
    for xyz, xyzw in [([1, 2], None), ([float('nan'), 0, 0], None), ([0, 0, 0], [0, 0, 0, 0])]:
        with pytest.raises(ValueError):
            validate_target(xyz, xyzw)


def test_failed_or_unexecuted_plan_cannot_be_saved():
    from arm_lab_kinematics.pipeline_target import PlanSession
    pending = PlanSession()
    assert not pending.can_save
    preview = pending.planned('trajectory', 'start', {'xyz': [0, 0, 0]})
    assert preview.can_execute and not preview.can_save
    executing = preview.executing()
    assert not executing.can_execute
    assert not executing.finished(False).can_save
    assert executing.finished(True).can_save
    assert not executing.finished(True).planned('next', 'start', {}).can_save


def test_twelve_joint_robot_generates_four_independent_leg_groups():
    from arm_lab_kinematics.pipeline_moveit import controller_joints
    legs = ('front_left', 'front_right', 'rear_left', 'rear_right')
    robot = {'name': 'dog', 'base': {'type': 'floating', 'link': 'body'},
             'links': [{'name': 'body'}] + [{'name': f'{leg}_{i}'} for leg in legs for i in range(3)],
             'joints': [{'name': f'{leg}_joint_{i}', 'type': 'revolute',
                         'parent': 'body' if i == 0 else f'{leg}_{i-1}', 'child': f'{leg}_{i}',
                         'limits': {'velocity': 1.0, 'acceleration': 2.0}}
                        for leg in legs for i in range(3)]}
    groups = {leg: {'joints': [f'{leg}_joint_{i}' for i in range(3)], 'base_link': 'body',
                    'tip_link': f'{leg}_2', 'controller': 'legs_controller'} for leg in legs}
    options = {'schema_version': 1, 'enabled': True, 'groups': groups}
    config = build_moveit_config(robot, options)
    assert len(config['robot_description_planning']['joint_limits']) == 12
    assert len(ET.fromstring(config['robot_description_semantic']).findall('group')) == 4
    assert len(controller_joints(options)['legs_controller']) == 12


def test_ros_goal_preserves_position_only_and_full_pose_semantics():
    pytest.importorskip('moveit_msgs.action')
    from arm_lab_kinematics.pipeline_target import build_goal
    group = options_fixture()['groups']['arm']
    point = build_goal('arm', group, {}, 'base', [.1, .2, .3])
    assert point.planning_options.plan_only
    constraint = point.request.goal_constraints[0]
    assert not constraint.orientation_constraints
    assert constraint.position_constraints[0].link_name == 'tool'
    assert constraint.position_constraints[0].constraint_region.primitive_poses[0].position.z == .3
    pose = build_goal('arm', group, {}, 'base', [.1, .2, .3], [0., 0., 0., 1.])
    assert pose.request.goal_constraints[0].orientation_constraints[0].orientation.w == 1.


def test_successful_ros_plan_roundtrips_with_actual_execution_and_benchmark(tmp_path):
    pytest.importorskip('moveit_msgs.msg')
    from types import SimpleNamespace
    from builtin_interfaces.msg import Duration
    from moveit_msgs.msg import RobotState, RobotTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from arm_lab_model.trajectory_store import load_trajectory
    from arm_lab_kinematics.pipeline_target import PlanSession, save_session, trajectory_message_digest
    trajectory = RobotTrajectory(joint_trajectory=JointTrajectory(joint_names=['joint'], points=[
        JointTrajectoryPoint(positions=[0.], time_from_start=Duration(sec=0)),
        JointTrajectoryPoint(positions=[.5], time_from_start=Duration(sec=1)),
    ]))
    target = {'frame_id': 'world', 'position': [.1, .2, .3], 'orientation_xyzw': [0., 0., 0., 1.]}
    session = PlanSession().planned(trajectory, RobotState(), target).executing().finished(True)
    project = SimpleNamespace(sections={}, source_path=tmp_path / 'project.yaml')
    path = save_session(session, {'name': 'robot'}, {}, project, 'a' * 64, {'status': 'complete'})
    saved = load_trajectory(path)
    assert saved['trajectory']['points'][0]['velocities'] == [.5]
    assert saved['planner']['parameters']['velocity_source'] == 'numerical_gradient'
    assert saved['execution']['status'] == 'succeeded'
    assert saved['benchmark'] == {'status': 'complete'}
    assert len(trajectory_message_digest(trajectory.joint_trajectory)) == 64
    different = deepcopy(trajectory.joint_trajectory)
    different.points[-1].positions = [.7]
    assert trajectory_message_digest(different) != trajectory_message_digest(trajectory.joint_trajectory)


def test_rpy_cli_quaternion_agrees_with_quarter_turn_about_z():
    from arm_lab_kinematics.pipeline_target import quaternion_from_rpy
    import math
    assert quaternion_from_rpy([0., 0., math.pi / 2]) == pytest.approx([0., 0., math.sqrt(.5), math.sqrt(.5)])


def _spin_until(node, predicate, status, timeout=45):
    import time
    import rclpy
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    assert predicate(), f'Timed out waiting for pipeline; latest statuses: {status[-8:]}'


def _trigger(node, name, status):
    from std_srvs.srv import Trigger
    client = node.create_client(Trigger, name)
    assert client.wait_for_service(timeout_sec=5), name
    future = client.call_async(Trigger.Request())
    _spin_until(node, future.done, status)
    return future.result()


def test_live_moveit_preview_execution_save_and_unreachable_target():
    """Opt-in: actual MoveIt/MuJoCo runtime required; exercises real ROS actions."""
    import os
    if os.environ.get('ARM_LAB_LIVE_PIPELINE') != '1':
        pytest.skip('Launch pipeline UR5e first, then set ARM_LAB_LIVE_PIPELINE=1')
    import json
    from pathlib import Path
    import rclpy
    from geometry_msgs.msg import PointStamped, PoseStamped
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from scipy.spatial.transform import Rotation
    from arm_lab_model.physical_robot import forward_tree, resolve_robot
    from arm_lab_model.project_config import load_project
    from arm_lab_model.trajectory_store import load_trajectory

    rclpy.init()
    node = rclpy.create_node('pipeline_integration_check')
    status, joints, scene = [], {}, {}
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, '/arm_lab/planning_status', lambda msg: status.append(json.loads(msg.data)), qos)
    node.create_subscription(String, '/arm_lab/scene_status', lambda msg: scene.update(json.loads(msg.data)), qos)
    node.create_subscription(JointState, '/joint_states', lambda msg: joints.update(zip(msg.name, msg.position)), 10)
    pose_pub = node.create_publisher(PoseStamped, '/arm_lab/target', 10)
    point_pub = node.create_publisher(PointStamped, '/clicked_point', 10)
    try:
        _spin_until(node, lambda: len(joints) >= 6 and scene.get('ready') is True, status)
        project_file = Path(__file__).resolve().parents[2] / 'arm_lab_model/config/pipeline/project_ur5e.yaml'
        robot = resolve_robot(load_project(project_file))
        transforms, _ = forward_tree(robot, joints)
        rotation, position = transforms['tool0']
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, position + [.015, 0, 0])
        q = pose.pose.orientation
        q.x, q.y, q.z, q.w = map(float, Rotation.from_matrix(rotation).as_quat())
        start = len(status)
        pose_pub.publish(pose)
        _spin_until(node, lambda: any(s['state'] in ('preview', 'planning_failed') for s in status[start:]), status)
        assert status[-1]['state'] == 'preview', status[-1]
        response = _trigger(node, '/arm_lab/execute_plan', status)
        assert response.success, response.message
        _spin_until(node, lambda: any(s['state'] in ('succeeded', 'execution_failed', 'failed') for s in status[start:]), status, 120)
        assert status[-1]['state'] == 'succeeded', status[-1]
        response = _trigger(node, '/arm_lab/save_trajectory', status)
        assert response.success, response.message
        assert load_trajectory(response.message)['execution']['status'] == 'succeeded'
        transforms, _ = forward_tree(robot, joints)
        target = PointStamped()
        target.header.frame_id = 'base_link'
        target.point.x, target.point.y, target.point.z = map(float, transforms['tool0'][1] + [0, .01, 0])
        start = len(status)
        point_pub.publish(target)
        _spin_until(node, lambda: any(s['state'] in ('preview', 'planning_failed') for s in status[start:]), status)
        assert status[-1]['state'] == 'preview', status[-1]
        start = len(status)
        target.point.x = 100.
        point_pub.publish(target)
        _spin_until(node, lambda: any(s['state'] == 'planning_failed' for s in status[start:]), status)
        assert not _trigger(node, '/arm_lab/execute_plan', status).success
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_floating_plan_cannot_be_saved_without_base_state():
    from arm_lab_kinematics.pipeline_target import PlanSession, save_session
    session = PlanSession(state='succeeded', trajectory=object())
    with pytest.raises(ValueError, match='floating'):
        save_session(session, {'name': 'dog', 'base': {'type': 'floating'}}, {}, None, 'a' * 64)
