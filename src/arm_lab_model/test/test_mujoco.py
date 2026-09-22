"""Independent engine checks; optional dependency, no display or ROS needed."""
import copy
from pathlib import Path

import numpy as np
import pytest

from arm_lab_model.config import load_config, MeasuredInertial

mujoco = pytest.importorskip('mujoco')


def test_native_mjcf_agrees_on_full_dynamics_with_asymmetric_links():
    from arm_lab_model.mujoco_backend import crosscheck
    cfg = copy.deepcopy(load_config())
    cfg.mount_xyz = [0.1, -0.2, 0.3]
    cfg.mount_rpy = [0.3, -0.4, 0.2]
    cfg.joints[1].origin_xyz = [0.04, -0.03, 0.02]
    cfg.joints[1].link.inertial = MeasuredInertial(
        2.3, [0.11, -0.02, 0.03], [0.02, 0.03, 0.04, 0.001, -0.002, 0.003])
    result = crosscheck(cfg, samples=20, payload=1.2)
    assert result['passed'], result
    assert result['max_full_dynamics_error_nm'] < 1e-7
    assert result['max_tcp_error_m'] < 1e-10


def test_torque_controlled_hold_and_saturation():
    from arm_lab_model.mujoco_backend import simulate_arm
    cfg = load_config()
    result = simulate_arm(cfg, duration=0.1, timestep=0.001)
    assert result['passed'], result
    assert result['max_tracking_error_rad'] < 1e-5
    cfg = copy.deepcopy(cfg)
    for actuator in cfg.actuators.values():
        actuator.peak_torque = 1e-8
        actuator.continuous_torque = 1e-8
    result = simulate_arm(cfg, duration=0.05, timestep=0.001)
    assert not result['passed']
    assert result['saturation_fraction'] > 0


def test_generic_floating_base_contact_smoke(tmp_path):
    from arm_lab_model.mujoco_backend import simulate_mjcf
    path = tmp_path / 'drop.xml'
    path.write_text('''<mujoco><worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body pos="0 0 .3"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
      </body></worldbody></mujoco>''')
    result = simulate_mjcf(str(path), duration=0.5, timestep=0.001)
    assert result['passed'], result
    assert result['nv'] == 6
    assert result['max_contacts'] > 0
    assert result['final_qpos'][2] == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize('kwargs', [
    {'duration': 0}, {'duration': float('nan')}, {'timestep': -0.1},
])
def test_invalid_simulation_options(kwargs):
    from arm_lab_model.mujoco_backend import simulate_arm
    with pytest.raises(ValueError):
        simulate_arm(load_config(), **kwargs)


def test_cli_failure_is_nonzero(capsys):
    from arm_lab_model.mujoco_backend import main
    assert main(['check', '--samples', '0']) == 2
    assert 'samples' in capsys.readouterr().err


def test_branched_quadruped_example():
    from arm_lab_model.mujoco_backend import simulate_mjcf
    path = Path(__file__).resolve().parents[3] / 'examples' / 'quadruped_drop.xml'
    result = simulate_mjcf(str(path), duration=0.5)
    assert result['passed'], result
    assert result['nv'] == 14  # 6 floating base + 8 hinges across four branches
    assert result['max_contacts'] >= 4


def test_cli_export_and_saved_check(tmp_path, capsys):
    import json
    from arm_lab_model.mujoco_backend import main
    xml = tmp_path / 'arm.xml'
    report = tmp_path / 'check.json'
    assert main(['export', '-o', str(xml)]) == 0
    assert mujoco.MjModel.from_xml_path(str(xml)).nv == load_config().dof
    assert main(['check', '--samples', '3', '-o', str(report)]) == 0
    assert json.loads(report.read_text())['passed']
    assert main(['simulate', '--duration', '0.02']) == 0
    assert json.loads(capsys.readouterr().out)['passed']


def test_generic_keyframe_and_missing_keyframe(tmp_path):
    from arm_lab_model.mujoco_backend import main, simulate_mjcf
    path = tmp_path / 'key.xml'
    path.write_text('''<mujoco><worldbody><body><joint name="j"/>
      <geom type="sphere" size=".1"/></body></worldbody>
      <keyframe><key name="home" qpos=".2"/></keyframe></mujoco>''')
    assert simulate_mjcf(path, duration=0.01, keyframe='home')['passed']
    assert main(['smoke', str(path), '--duration', '0.01', '--keyframe', 'home']) == 0
    with pytest.raises(ValueError, match='unknown keyframe'):
        simulate_mjcf(path, duration=0.01, keyframe='absent')


def test_motion_survives_timestep_refinement():
    from arm_lab_model.mujoco_backend import simulate_arm
    cfg = load_config()
    coarse = simulate_arm(cfg, target='stowed', duration=5, timestep=0.002)
    fine = simulate_arm(cfg, target='stowed', duration=5, timestep=0.001)
    assert coarse['passed'], coarse
    assert fine['passed'], fine
    np.testing.assert_allclose(coarse['final_q_rad'], fine['final_q_rad'], atol=0.002)


@pytest.mark.parametrize('runner', ['arm', 'mjcf'])
def test_numerical_warning_cannot_pass(monkeypatch, tmp_path, runner):
    from arm_lab_model.mujoco_backend import build_mjcf, simulate_arm, simulate_mjcf
    def fail_step(model, data):
        data.warning[0].number += 1
    monkeypatch.setattr(mujoco, 'mj_step', fail_step)
    cfg = load_config()
    if runner == 'arm':
        result = simulate_arm(cfg, duration=0.01)
    else:
        path = tmp_path / 'arm.xml'
        path.write_text(build_mjcf(cfg))
        result = simulate_mjcf(path, duration=0.01)
    assert not result['passed']
    assert result['warnings'][0] == 1


def test_invalid_pose_cannot_run():
    from arm_lab_model.mujoco_backend import simulate_arm
    cfg = load_config()
    cfg.test_poses['outside'] = [100] * cfg.dof
    with pytest.raises(ValueError, match='inside joint limits'):
        simulate_arm(cfg, pose='outside')


# ------------------------------------------------ interactive simulator core
ROOT = Path(__file__).resolve().parents[3]
WORLD = ROOT / 'src' / 'arm_lab_bringup' / 'worlds' / 'arm_test_world.sdf'
CONFIG_DIR = ROOT / 'src' / 'arm_lab_model' / 'config'


def test_benchmark_model_has_no_jaws_or_contact_filters_by_default():
    from arm_lab_model.mujoco_backend import build_mjcf
    xml = build_mjcf(load_config())
    assert 'slide' not in xml and 'contype' not in xml


def test_world_loader_reads_the_gazebo_boxes_and_ground():
    from arm_lab_model.mujoco_backend import load_world
    scene = load_world(WORLD)
    boxes = {b.name: b for b in scene.boxes}
    assert set(boxes) == {'sample_2kg', 'sample_3kg'}
    assert boxes['sample_2kg'].mass == 2.0
    assert boxes['sample_2kg'].size == (0.08, 0.08, 0.08)
    assert boxes['sample_3kg'].pos == (0.55, -0.35, 0.045)
    assert boxes['sample_3kg'].friction == 1.4
    assert scene.ground_friction == 1.0


def test_jaws_keep_the_tool_mass_and_centre_of_mass():
    from arm_lab_model.mujoco_backend import build_mjcf, load_world
    cfg = load_config()
    mj = mujoco.MjModel.from_xml_string(
        build_mjcf(cfg, fingers=True, scene=load_world(WORLD)))
    data = mujoco.MjData(mj)
    for name in cfg.end_effector.finger_joint_names:     # both jaws equally open
        data.qpos[mj.joint(name).qposadr[0]] = 0.03
    mujoco.mj_forward(mj, data)
    tool = mj.body('gripper_rigid_tool')
    assert mj.body_subtreemass[tool.id] == pytest.approx(cfg.end_effector.mass)
    flange = data.site('tcp').xpos - data.xpos[tool.id]
    flange /= np.linalg.norm(flange)
    com = data.subtree_com[tool.id] - data.xpos[tool.id]
    np.testing.assert_allclose(com, flange * cfg.end_effector.com_offset, atol=1e-9)


def test_arm_geoms_collide_with_the_scene_but_not_with_each_other():
    from arm_lab_model.mujoco_backend import build_mjcf, load_world
    mj = mujoco.MjModel.from_xml_string(
        build_mjcf(load_config(), fingers=True, scene=load_world(WORLD)))
    world_body = 0
    boxes = {mj.body('sample_2kg').id, mj.body('sample_3kg').id}
    arm = [g for g in range(mj.ngeom)
           if mj.geom_bodyid[g] != world_body and mj.geom_bodyid[g] not in boxes]
    scene = [g for g in range(mj.ngeom) if g not in arm]

    def collide(a, b):
        return bool(mj.geom_contype[a] & mj.geom_conaffinity[b]
                    or mj.geom_contype[b] & mj.geom_conaffinity[a])
    assert not any(collide(a, b) for a in arm for b in arm if a < b)
    assert all(collide(a, s) for a in arm for s in scene)


def test_trajectory_sampler_follows_hermite_segments_and_holds_the_end():
    from arm_lab_model.mujoco_sim import TrajectoryPoint, TrajectorySampler
    q0, q1 = np.zeros(2), np.array([1.0, -2.0])
    cubic = TrajectorySampler(10.0, q0, np.zeros(2),
                              [TrajectoryPoint(2.0, q1, np.zeros(2))])
    q, v, a = cubic.sample(11.0)                          # halfway, peak speed
    np.testing.assert_allclose(q, q1 / 2)
    np.testing.assert_allclose(v, 1.5 * q1 / 2.0)
    np.testing.assert_allclose(a, 0.0, atol=1e-12)
    np.testing.assert_allclose(cubic.sample(10.0)[0], q0)
    q, v, a = cubic.sample(30.0)
    np.testing.assert_allclose(q, q1)
    assert not v.any() and not a.any() and cubic.done(12.0)
    linear = TrajectorySampler(0.0, q0, np.zeros(2), [TrajectoryPoint(2.0, q1)])
    np.testing.assert_allclose(linear.sample(0.5)[1], q1 / 2.0)


def test_simulation_rejects_unknown_joints_and_holds_on_an_empty_trajectory():
    from arm_lab_model.mujoco_sim import ArmSimulation, TrajectoryPoint
    sim = ArmSimulation(load_config())
    with pytest.raises(ValueError, match='unknown joints'):
        sim.set_trajectory(['elbow'], [TrajectoryPoint(1.0, np.zeros(1))])
    held = sim.desired()[0]
    sim.hold()
    for _ in range(300):
        sim.step()
    assert np.max(np.abs(sim.tracking_error())) < 1e-4
    np.testing.assert_allclose(sim.desired()[0], held)


def test_jaws_close_at_grip_speed_and_stop_at_their_limit():
    from arm_lab_model.mujoco_sim import ArmSimulation
    cfg = load_config()
    sim = ArmSimulation(cfg)
    ee = cfg.end_effector
    sim.set_gripper([-ee.grip_force_max])
    for _ in range(200):
        sim.step()
    speed = abs(sim.joint_state()[2][cfg.dof])
    assert speed == pytest.approx(ee.grip_speed, rel=0.1)
    for _ in range(4000):
        sim.step()
    jaws = sim.joint_state()[1][cfg.dof:]
    assert min(jaws) > -0.002, jaws        # a soft limit, but a stiff one


@pytest.mark.parametrize('path', sorted(CONFIG_DIR.glob('*.yaml')), ids=lambda p: p.name)
def test_every_shipped_config_passes_the_dynamics_crosscheck(path):
    from arm_lab_model.mujoco_backend import crosscheck
    result = crosscheck(load_config(str(path)), samples=10, payload=2.0)
    assert result['passed'], result
