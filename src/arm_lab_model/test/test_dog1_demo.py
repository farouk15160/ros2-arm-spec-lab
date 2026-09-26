"""Public smoke contracts for a literal one-actuator dog-shaped fixture."""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from arm_lab_model.project_config import load_project
from arm_lab_model.physical_robot import physical_report, resolve_robot
from arm_lab_model.robot_export import build_robot_mjcf, build_robot_urdf


CONFIG = Path(__file__).parents[1]/'config/pipeline'
PROJECT = CONFIG/'project_dog1_demo.yaml'
SCENARIO = CONFIG/'scenario_dog1_demo.yaml'
JOINT = 'front_left_hip_pitch'


def test_one_actuated_dog_has_known_mass_and_analytic_leg_gravity():
    mujoco = pytest.importorskip('mujoco')
    from arm_lab_model.pipeline_runtime import RobotSimulation
    project = load_project(PROJECT)
    robot = resolve_robot(project)
    assert project.joint_names == (JOINT,)
    assert robot['base']['type'] == 'fixed'
    assert 'SYNTHETIC' in robot['source']
    assert len([link for link in robot['links'] if link['name'].endswith('_leg')]) == 4
    # Independent box masses: torso7.2 + fourlegs.96 + head1.008 + tail.03 kg.
    report = physical_report(robot,q=[.25])
    assert report['total_mass'] == pytest.approx(9.198)
    expected = .24*9.81*.15*math.sin(.25)
    assert report['joint_static_effort'][JOINT] == pytest.approx(expected)
    sim = RobotSimulation(robot)
    sim.set_state([.25])
    sim.apply_control()
    assert sim.sample()['joint_torque'][0] == pytest.approx(expected)
    xml = ET.fromstring(build_robot_urdf(robot))
    assert len([joint for joint in xml.findall('joint') if joint.get('type') != 'fixed']) == 1
    model = mujoco.MjModel.from_xml_string(build_robot_mjcf(robot))
    assert model.nq == model.nv == model.nu == 1
    assert sum(model.body_mass) == pytest.approx(9.198)


def test_cli_build_swing_save_and_replay_succeed_without_contacts(tmp_path,capsys):
    pytest.importorskip('mujoco')
    from arm_lab_model.pipeline_cli import main
    build = tmp_path/'build'
    output = tmp_path/'swing'
    replay = tmp_path/'replay'
    assert main(['build',str(PROJECT),'--output',str(build)]) == 0
    assert (build/'robot.urdf').is_file()
    assert (build/'robot.xml').is_file()
    assert main(['simulate',str(PROJECT),'--scenario',str(SCENARIO),
                 '--output',str(output),'--save']) == 0
    result = json.loads((output/'execution.json').read_text())
    assert result['passed'] is True
    assert result['max_contacts'] == result['saturated_steps'] == 0
    assert result['max_tracking_error'] < .001
    saved = Path(result['saved_trajectory'])
    assert saved.is_file()
    observation = json.loads((output/'observation.json').read_text())
    assert observation['joint_names'] == [JOINT]
    assert observation['scenario_id'] == 'synthetic_dog1_leg_swing'
    assert observation['evidence']['kind'] == 'synthetic'
    q = np.array(observation['samples']['joint_position'])[:,0]
    assert max(q) > .249 and min(q) < -.199
    assert abs(q[-1]) < .0001
    assert main(['replay',str(PROJECT),'--trajectory',str(saved),'--output',str(replay)]) == 0
    repeated = json.loads((replay/'observation.json').read_text())
    np.testing.assert_allclose(repeated['samples']['joint_position'],
                               observation['samples']['joint_position'],atol=1e-12)
