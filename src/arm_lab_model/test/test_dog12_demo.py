"""Synthetic quadruped integration smoke; never a vendor/hardware benchmark."""
from pathlib import Path

import numpy as np
import pytest

from arm_lab_model.project_config import load_project
from arm_lab_model.physical_robot import physical_report, resolve_robot
from arm_lab_model.scene_config import resolve_environment
from arm_lab_model.config_contract import read_yaml


CONFIG = Path(__file__).parents[1] / 'config/pipeline'


def test_dog12_export_mass_dofs_and_falling_contact_are_consistent():
    pytest.importorskip('mujoco')
    from arm_lab_model.pipeline_runtime import RobotSimulation
    project = load_project(CONFIG/'project_dog12_demo.yaml')
    robot = resolve_robot(project)
    report = physical_report(robot)
    sim = RobotSimulation(robot, dict(project.sections['simulation']), resolve_environment(project))
    assert len(sim.names) == sim.m.nu == 12
    assert sim.m.nv == 18
    assert sim.m.nq == 19
    assert report['total_mass'] == pytest.approx(9.56)
    assert sum(sim.m.body_mass) == pytest.approx(report['total_mass'])
    assert sim.d.subtree_com[0] == pytest.approx(report['center_of_mass'])
    assert 'SYNTHETIC' in robot['source']
    initial_height = sim.d.qpos[2]
    saw_foot_contact = False
    for _ in range(600):
        sim.step()
        for contact in sim.d.contact:
            names = {sim.m.geom(int(index)).name for index in contact.geom}
            if 'floor_collision' in names and any('lower_leg_collision' in name for name in names):
                saw_foot_contact = True
    assert sim.d.qpos[2] < initial_height
    assert saw_foot_contact
    assert np.isfinite(sim.d.qpos).all()
    assert not any(w.number for w in sim.d.warning)


def test_dog12_hold_scenario_reports_contacts_as_failed_not_gait_success():
    pytest.importorskip('mujoco')
    from arm_lab_model.pipeline_runtime import RobotSimulation, run_trajectory
    project = load_project(CONFIG/'project_dog12_demo.yaml')
    robot = resolve_robot(project)
    sim = RobotSimulation(robot, dict(project.sections['simulation']), resolve_environment(project))
    scenario = read_yaml(CONFIG/'scenario_dog12_demo.yaml')
    assert scenario['joint_names'] == list(sim.names)
    result = run_trajectory(sim, scenario['points'], scenario_id=scenario['scenario_id'])
    assert result['max_contacts'] > 0
    assert result['passed'] is False
    assert result['trajectory']['execution']['status'] == 'failed'
    assert result['observation']['evidence']['kind'] == 'synthetic'


def test_dog12_actuators_move_branched_joints_without_a_balance_claim():
    pytest.importorskip('mujoco')
    from arm_lab_model.pipeline_runtime import RobotSimulation
    project = load_project(CONFIG/'project_dog12_demo.yaml')
    robot = resolve_robot(project)
    sim = RobotSimulation(robot, dict(project.sections['simulation']))
    target = [0.005 if index % 2 == 0 else -0.005 for index in range(12)]
    sim.command(sim.names, [{'time_from_start':0,'positions':[0]*12},
                            {'time_from_start':.1,'positions':target}])
    for _ in range(150):
        sim.step()
        assert np.all(np.abs(sim.torque) <= 20)
    assert np.all(np.sign(sim.d.qpos[sim.qidx]) == np.sign(target))
    assert np.max(np.abs(sim.d.qpos[sim.qidx])) > .001
    assert not any(w.number for w in sim.d.warning)
