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
