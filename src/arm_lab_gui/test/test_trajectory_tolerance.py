"""Public joint action tolerance policy, checked independently of ROS."""
import pytest
from arm_lab_gui.trajectory_tolerance import tolerance_policy, tracking_violation, validate_terminal_state


def test_named_tolerances_defaults_disable_and_actual_derivative_errors():
    policy = tolerance_policy(['a', 'b'],
        path=[{'name': 'b', 'position': -1., 'velocity': .2, 'acceleration': .5}],
        goal=[{'name': 'a', 'position': .01, 'velocity': .03, 'acceleration': .1}],
        goal_time=2., path_position=.05, goal_position=.02, goal_velocity=.01)
    assert policy.path[0] == (.05, None, None)
    assert policy.path[1] == (None, .2, .5)
    assert policy.goal[0] == (.01, .03, .1)
    assert policy.goal_time == 2.
    desired = ([0., 0.], [0., 0.], [0., 0.])
    assert tracking_violation(policy, desired, ([0., 1.], [0., .1], [0., .4]), final=False) is None
    assert 'b velocity' in tracking_violation(policy, desired, ([0., 0.], [0., .3], [0., 0.]), final=False)
    assert 'a acceleration' in tracking_violation(policy, desired, ([0., 0.], [0., 0.], [.2, 0.]), final=True)


@pytest.mark.parametrize('entries', [
    [{'name': 'missing', 'position': .1}],
    [{'name': 'a', 'position': -.5}],
    [{'name': 'a', 'velocity': float('nan')}],
    [{'name': 'a'}, {'name': 'a'}],
])
def test_invalid_action_tolerances_are_rejected(entries):
    with pytest.raises(ValueError):
        tolerance_policy(['a'], path=entries)


def test_zero_request_uses_defaults_and_minus_one_explicitly_disables():
    default = tolerance_policy(['a'], goal=[{'name': 'a', 'position': 0., 'velocity': -1.}], goal_time=0.)
    assert default.goal == ((.02, None, None),)
    assert default.goal_time == 1.
    with pytest.raises(ValueError, match='goal_time'):
        tolerance_policy(['a'], goal_time=-.1)


def test_terminal_velocity_must_be_stopped_but_one_sided_acceleration_is_valid():
    validate_terminal_state({'velocities': [0., 1e-15], 'accelerations': [.4, -.2]})
    with pytest.raises(ValueError, match='terminal velocity'):
        validate_terminal_state({'velocities': [.1]})
    with pytest.raises(ValueError, match='effort'):
        validate_terminal_state({'velocities': [0.], 'effort': [1.]})


@pytest.mark.parametrize('saturated,contacts,error', [(1, 0, 0.), (0, 1, 0.), (0, 0, .2)])
def test_controller_aborts_and_holds_for_every_sampled_safety_failure(saturated, contacts, error):
    pytest.importorskip('rclpy')
    import numpy as np
    from types import SimpleNamespace
    from arm_lab_gui.pipeline_sim_node import PipelineSimulationNode
    from control_msgs.action import FollowJointTrajectory
    held = []
    simulation = SimpleNamespace(time=.5, desired=lambda: (np.array([0.]), np.array([0.]), np.array([0.])),
                                 hold=lambda: held.append(True))
    state = SimpleNamespace(execution_policy=tolerance_policy(['a']), execution_fault=None,
                            saturation=saturated, contacts=contacts, sim=simulation,
                            execution_end=1., execution_indices=(0,), rows=[{
                                'joint_position': [error], 'joint_velocity': [0.], 'joint_acceleration': [0.]}])
    PipelineSimulationNode.check_execution_sample(state)
    assert held == [True]
    assert state.execution_fault[0] == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED


def test_action_rejects_unsupported_terminal_velocity_and_accepts_named_derivative_tolerance():
    pytest.importorskip('rclpy')
    from types import SimpleNamespace
    from arm_lab_gui.pipeline_sim_node import PipelineSimulationNode
    from control_msgs.action import FollowJointTrajectory
    from control_msgs.msg import JointTolerance, JointComponentTolerance
    from trajectory_msgs.msg import JointTrajectoryPoint
    request = FollowJointTrajectory.Goal()
    request.trajectory.joint_names = ['a']
    request.trajectory.points = [JointTrajectoryPoint(positions=[0.], velocities=[.1])]
    node = SimpleNamespace(sim=SimpleNamespace(settings={}),
                           param=lambda name: {'goal_position_tolerance': .02,
                                               'goal_velocity_tolerance': .01,
                                               'goal_time_tolerance': 1.}[name])
    with pytest.raises(ValueError, match='terminal velocity'):
        PipelineSimulationNode.request_policy(node, request)
    request.trajectory.points[0].velocities = [0.]
    request.path_tolerance = [JointTolerance(name='a', position=.03, velocity=.2, acceleration=.5)]
    policy = PipelineSimulationNode.request_policy(node, request)
    assert policy.path == ((.03, .2, .5),)
    request.component_goal_tolerance = [JointComponentTolerance()]
    with pytest.raises(ValueError, match='component tolerances'):
        PipelineSimulationNode.request_policy(node, request)
