"""Public trajectory and generic physics runtime contracts."""
import numpy as np
import pytest

from arm_lab_model.joint_trajectory import JointPath


def test_quintic_path_has_exact_endpoints_and_known_midpoint():
    path = JointPath.from_points(['joint'], [
        {'time_from_start': 0, 'positions': [0]},
        {'time_from_start': 2, 'positions': [1]},
    ])
    np.testing.assert_allclose(path.sample(0), [[0], [0], [0]], atol=1e-12)
    np.testing.assert_allclose(path.sample(1), [[0.5], [0.9375], [0]], atol=1e-12)
    np.testing.assert_allclose(path.sample(2), [[1], [0], [0]], atol=1e-12)
    with pytest.raises(ValueError, match='velocity'):
        path.check_limits([{'type': 'revolute', 'limits': {
            'lower': -2, 'upper': 2, 'velocity': 0.8, 'acceleration': 5}}])


@pytest.mark.parametrize('points', [[],
    [{'time_from_start': 0, 'positions': [0]}, {'time_from_start': 0, 'positions': [1]}],
    [{'time_from_start': 0, 'positions': [float('nan')]}],
    [{'time_from_start': 0, 'positions': [0, 1]}],
])
def test_invalid_path_cannot_execute(points):
    with pytest.raises(ValueError):
        JointPath.from_points(['joint'], points)


def test_acceleration_policy_cannot_raise_a_known_robot_rating():
    path = JointPath.from_points(['joint'], [
        {'time_from_start': 0, 'positions': [0]}, {'time_from_start': 1, 'positions': [1]}])
    joint = {'type': 'revolute', 'limits': {'lower': -2, 'upper': 2,
                                          'velocity': 3, 'acceleration': 1}}
    with pytest.raises(ValueError, match='exceeds robot'):
        path.check_limits([joint], {'joint': 10})


def test_immutable_acceleration_policy_is_accepted():
    from types import MappingProxyType
    path = JointPath.from_points(['joint'], [
        {'time_from_start': 0, 'positions': [0]}, {'time_from_start': 2, 'positions': [.1]}])
    path.check_limits([{'type': 'revolute', 'limits': {
        'lower': -2, 'upper': 2, 'velocity': 1, 'acceleration': None}}],
        MappingProxyType({'joint': 1.0}))


def test_ur5e_records_controlled_initial_effort_and_requested_endpoint():
    from pathlib import Path
    from arm_lab_model.project_config import load_project
    from arm_lab_model.physical_robot import resolve_robot
    from arm_lab_model.pipeline_runtime import RobotSimulation, run_trajectory
    project = load_project(Path(__file__).parents[1] / 'config/pipeline/project_ur5e.yaml')
    robot = resolve_robot(project)
    sim = RobotSimulation(robot)
    q = list(project.sections['simulation']['initial_positions'])
    sim.set_state(q)
    result = run_trajectory(sim, [{'time_from_start': 0, 'positions': q},
                                  {'time_from_start': .01, 'positions': q}],
                            scenario_id='hold', acceleration_limits=dict.fromkeys(sim.names, 2),
                            end_effector='tool')
    assert result['passed']
    assert abs(result['observation']['samples']['joint_torque'][0][1]) > 1
    assert np.max(np.abs(result['observation']['samples']['joint_acceleration'][0])) < 1e-6
    assert result['observation']['end_effector'] == 'tool0'
    with pytest.raises(ValueError, match='backend'):
        RobotSimulation(robot, {'backend': 'gazebo'})


@pytest.fixture
def ur5e_sim():
    from pathlib import Path
    from arm_lab_model.project_config import load_project
    from arm_lab_model.physical_robot import resolve_robot
    from arm_lab_model.pipeline_runtime import RobotSimulation
    pytest.importorskip('mujoco')
    project = load_project(Path(__file__).parents[1]/'config/pipeline/project_ur5e.yaml')
    sim = RobotSimulation(resolve_robot(project))
    sim.set_state(project.sections['simulation']['initial_positions'])
    return sim


@pytest.mark.parametrize('offset', [0.0, 0.15, -0.2])
def test_ur5e_initial_gravity_efforts_match_independent_tree_statics(ur5e_sim, offset):
    from arm_lab_model.physical_robot import physical_report
    sim = ur5e_sim
    q = sim.sample()['joint_position']
    q[1] += offset
    q[2] -= offset/2
    sim.set_state(q)
    sim.apply_control()
    expected = physical_report(sim.robot, q=q)['joint_static_effort']
    np.testing.assert_allclose(sim.sample()['joint_torque'],
        [expected[name] for name in sim.names], atol=1e-10)
    assert not sim.saturated


def test_tcp_velocity_matches_frame_origin_with_offset_link_com(ur5e_sim):
    sim = ur5e_sim
    sim.robot = {**sim.robot, 'end_effectors': {'test': 'upper_arm_link'}}
    sim.d.qvel[sim.vidx] = [.7, -.2, .3, 0, 0, 0]
    sim.mj.mj_forward(sim.m, sim.d)
    sample = sim.sample('test')
    jacobian = np.zeros((3, sim.m.nv))
    sim.mj.mj_jacBody(sim.m, sim.d, jacobian, None, sim.m.body('upper_arm_link').id)
    np.testing.assert_allclose(sample['tcp_velocity'], jacobian @ sim.d.qvel, atol=1e-12)


def test_ur5e_tracks_a_real_integrated_joint_motion_and_holds_terminal_pose(ur5e_sim):
    from arm_lab_model.pipeline_runtime import run_trajectory
    sim = ur5e_sim
    start = sim.sample()['joint_position']
    target = [value + delta for value, delta in zip(start,[.05,.03,-.03,0,0,0])]
    result = run_trajectory(sim,[{'time_from_start':0,'positions':start},
        {'time_from_start':1,'positions':target}],scenario_id='synthetic_tracking_check',
        acceleration_limits=dict.fromkeys(sim.names,2))
    assert result['passed']
    assert result['max_tracking_error'] < .001
    assert result['saturated_steps'] == result['max_contacts'] == 0
    assert result['trajectory']['execution']['status'] == 'succeeded'
    samples = result['observation']['samples']
    np.testing.assert_allclose(samples['joint_position'][-1], target, atol=1e-4)
    assert samples['time'][0] == 0
    assert samples['time'][-1] == pytest.approx(1)
    assert result['observation']['evidence']['kind'] == 'synthetic'
    for _ in range(100):
        sim.step()
    desired_position, desired_velocity, desired_acceleration = sim.desired()
    np.testing.assert_allclose(desired_position,target,atol=1e-12)
    assert not np.any(desired_velocity) and not np.any(desired_acceleration)


def test_subset_command_inserts_current_state_and_holds_other_joints(ur5e_sim):
    sim = ur5e_sim
    initial = np.array(sim.sample()['joint_position'])
    path = sim.command([sim.names[-1]], [{'time_from_start':1,'positions':[.04]}],
                       dict.fromkeys(sim.names,2))
    np.testing.assert_allclose(path.sample(0)[0],initial)
    np.testing.assert_allclose(path.sample(1)[0][:-1],initial[:-1])
    assert path.sample(1)[0][-1] == pytest.approx(.04)
    for _ in range(500):
        sim.step()
    final = sim.sample()['joint_position']
    np.testing.assert_allclose(final[:-1],initial[:-1],atol=1e-5)
    assert final[-1] == pytest.approx(.04,abs=1e-4)
    sim.hold()
    held = sim.sample()['joint_position']
    np.testing.assert_allclose(sim.desired()[0],held)
    assert not np.any(sim.desired()[1])


@pytest.mark.parametrize('names,points,message', [
    (['not_a_joint'],[{'time_from_start':1,'positions':[0]}],'unknown'),
    (['wrist_3_joint','wrist_3_joint'],[{'time_from_start':1,'positions':[0,0]}],'duplicate'),
    (['wrist_3_joint'],[{'time_from_start':1,'positions':[0,0]}],'positions'),
    (['wrist_3_joint'],[{'time_from_start':1,'positions':[0],'velocities':[0,0]}],'velocities'),
    (['wrist_3_joint'],[{'time_from_start':0,'positions':[1]},
                        {'time_from_start':1,'positions':[1]}],'start differs'),
    (['wrist_3_joint'],[{'time_from_start':1,'positions':[7]}],'position limit'),
])
def test_invalid_subset_commands_are_rejected_before_motion(ur5e_sim,names,points,message):
    before = ur5e_sim.sample()
    with pytest.raises(ValueError,match=message):
        ur5e_sim.command(names,points,dict.fromkeys(ur5e_sim.names,2))
    assert ur5e_sim.sample() == before


@pytest.mark.parametrize('positions,message', [([], 'joint count'),
    ([0]*5+[float('nan')], 'finite'), ([0]*5+[7], 'outside')])
def test_invalid_joint_states_are_rejected(ur5e_sim,positions,message):
    initial = ur5e_sim.sample()
    with pytest.raises(ValueError,match=message):
        ur5e_sim.set_state(positions)
    assert ur5e_sim.sample() == initial


@pytest.mark.parametrize('bandwidth', [0,-1,float('inf'),float('nan')])
def test_invalid_controller_bandwidth_is_not_executed(bandwidth):
    from arm_lab_model.pipeline_runtime import RobotSimulation
    from arm_lab_model.physical_robot import resolve_robot
    from test_physical_robot import pendulum,project
    sim = RobotSimulation(resolve_robot(project(pendulum())),{'bandwidth_hz':bandwidth})
    with pytest.raises(ValueError,match='bandwidth'):
        sim.step()
    assert sim.time == 0


def test_unknown_effort_cannot_be_used_for_controlled_simulation():
    from arm_lab_model.pipeline_runtime import RobotSimulation
    from arm_lab_model.physical_robot import resolve_robot
    from test_physical_robot import pendulum,project
    robot = pendulum()
    robot['joints'][0]['limits']['effort'] = None
    with pytest.raises(ValueError,match='positive joint effort'):
        RobotSimulation(resolve_robot(project(robot)))


def test_insufficient_actuator_effort_rejects_execution_even_with_loose_tracking_tolerance():
    from arm_lab_model.pipeline_runtime import RobotSimulation,run_trajectory
    from arm_lab_model.physical_robot import resolve_robot
    from test_physical_robot import pendulum,project
    robot = pendulum()
    robot['joints'][0]['limits']['effort'] = .1
    sim = RobotSimulation(resolve_robot(project(robot)))
    result = run_trajectory(sim,[{'time_from_start':0,'positions':[0]},
        {'time_from_start':.1,'positions':[0]}],scenario_id='underpowered_hold',max_error=1)
    assert result['max_tracking_error'] < 1
    assert result['max_contacts'] == 0
    assert result['saturated_steps'] > 0
    assert result['passed'] is False
    assert result['trajectory']['execution']['status'] == 'failed'
    assert max(abs(row[0]) for row in result['observation']['samples']['joint_torque']) <= .1


def test_named_endpoint_is_selected_and_unknown_endpoints_are_rejected(ur5e_sim):
    sim = ur5e_sim
    np.testing.assert_allclose(sim.sample('tool')['tcp_position'],sim.sample('tool0')['tcp_position'])
    assert sim.endpoint('tool') == 'tool0'
    with pytest.raises(ValueError,match='unknown configured'):
        sim.sample('shoulder_link')


def test_floating_contact_execution_captures_base_start_state_without_success_claim():
    from pathlib import Path
    from arm_lab_model.project_config import load_project
    from arm_lab_model.physical_robot import resolve_robot
    from arm_lab_model.scene_config import resolve_environment
    from arm_lab_model.pipeline_runtime import RobotSimulation,run_trajectory
    config = load_project(Path(__file__).parents[1]/'config/pipeline/project_dog12_demo.yaml')
    sim = RobotSimulation(resolve_robot(config),environment=resolve_environment(config))
    result = run_trajectory(sim,[{'time_from_start':0,'positions':[0]*12},
        {'time_from_start':.3,'positions':[0]*12}],scenario_id='synthetic_floor_contact',
        end_effector='rear_right_foot',max_error=1)
    base = result['trajectory']['base_start_state']
    np.testing.assert_allclose(base['qpos'],[0,0,.5,1,0,0,0])
    np.testing.assert_allclose(base['qvel'],[0]*6)
    assert result['observation']['end_effector'] == 'rear_right_foot'
    assert result['max_contacts'] > 0
    assert result['trajectory']['collision']['status'] == 'collision'
    assert result['passed'] is False


@pytest.mark.parametrize('scenario_id,max_error,message', [('',.05,'scenario_id'),
    ('case',0,'max_error'), ('case',float('nan'),'max_error')])
def test_execution_metadata_and_tolerance_validation(ur5e_sim,scenario_id,max_error,message):
    from arm_lab_model.pipeline_runtime import run_trajectory
    q = ur5e_sim.sample()['joint_position']
    with pytest.raises(ValueError,match=message):
        run_trajectory(ur5e_sim,[{'time_from_start':0,'positions':q},
            {'time_from_start':1,'positions':q}],scenario_id=scenario_id,max_error=max_error)
