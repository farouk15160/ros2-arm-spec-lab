"""Opt-in live test of generic simulation without a MoveIt process."""
import os
import time

import pytest


def _until(node, predicate, timeout=30):
    import rclpy
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.05)
    assert predicate(), 'Timed out waiting for the running dog1 pipeline'


def test_live_dog1_joint_action_moves_and_completes_without_moveit():
    if os.environ.get('ARM_LAB_LIVE_DOG1') != '1':
        pytest.skip('Launch project_dog1_demo.yaml and set ARM_LAB_LIVE_DOG1=1')
    import json
    import rclpy
    from action_msgs.msg import GoalStatus
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    rclpy.init()
    node = rclpy.create_node('dog1_pipeline_integration_check')
    positions, scene = [], {}
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, '/arm_lab/scene_status', lambda msg: scene.update(json.loads(msg.data)), qos)
    node.create_subscription(JointState, '/joint_states',
        lambda msg: positions.append(msg.position[msg.name.index('front_left_hip_pitch')])
        if 'front_left_hip_pitch' in msg.name else None, 10)
    client = ActionClient(node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
    try:
        _until(node, lambda: positions and scene.get('ready') is True and client.server_is_ready())
        assert scene['planning_scene_applied'] is False
        assert not {'move_group', 'pipeline_target'} & set(node.get_node_names())
        assert abs(positions[-1]) < .02, 'Test expects initial dog1 position near zero'
        request = FollowJointTrajectory.Goal(trajectory=JointTrajectory(
            joint_names=['front_left_hip_pitch'], points=[
                JointTrajectoryPoint(positions=[q], velocities=[0.], accelerations=[0.],
                                     time_from_start=Duration(sec=seconds))
                for seconds, q in ((0, 0.), (2, .25), (4, 0.))]))
        sent = client.send_goal_async(request)
        _until(node, sent.done)
        handle = sent.result()
        assert handle.accepted
        finished = handle.get_result_async()
        _until(node, finished.done)
        result = finished.result()
        assert result.status == GoalStatus.STATUS_SUCCEEDED
        assert result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL, result.result.error_string
        assert max(positions) > .2, 'Joint telemetry must show the commanded intermediate motion'
        assert abs(positions[-1]) < .02
    finally:
        node.destroy_node()
        rclpy.shutdown()
