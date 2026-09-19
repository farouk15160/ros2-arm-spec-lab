"""Regression tests for measured inertials and joint-frame conventions."""

import copy
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from arm_lab_model.config import load_config
from arm_lab_model.kinematics import ArmModel, rpy_to_matrix
from arm_lab_model.urdf_builder import build_urdf


@pytest.fixture
def raw():
    return copy.deepcopy(load_config().raw)


def load_variant(raw, tmp_path):
    path = tmp_path / 'variant.yaml'
    path.write_text(yaml.safe_dump(raw))
    return load_config(str(path))


def test_origin_translation_precedes_joint_rotation(raw, tmp_path):
    raw['joints'][0]['origin_xyz'] = [0.12, -0.04, 0.03]
    raw['joints'][0]['origin_rpy'] = [0.4, -0.3, 0.7]
    cfg = load_variant(raw, tmp_path)
    fs = ArmModel(cfg).frames(np.zeros(cfg.dof))
    expected = fs.base_distal + rpy_to_matrix(cfg.mount_rpy) @ np.array([0.12, -0.04, 0.03])
    np.testing.assert_allclose(fs.joint_origin[0], expected, atol=1e-12)


def test_measured_inertial_is_complete_body_not_extra_mass(raw, tmp_path):
    raw['joints'][0]['link']['inertial'] = {
        'mass': 2.3, 'com': [0.03, -0.02, 0.05],
        'inertia': [0.02, 0.03, 0.04, 0.001, -0.002, 0.003],
    }
    cfg = load_variant(raw, tmp_path)
    link = cfg.joints[0].link
    assert link.mass == pytest.approx(2.3)
    fs = ArmModel(cfg).frames(np.zeros(cfg.dof))
    R = rpy_to_matrix(cfg.mount_rpy) @ rpy_to_matrix(cfg.joints[0].origin_rpy)
    np.testing.assert_allclose(fs.link_com[0], fs.joint_origin[0] + R @ [0.03, -0.02, 0.05])
    I = np.array([[0.02, 0.001, -0.002], [0.001, 0.03, 0.003], [-0.002, 0.003, 0.04]])
    np.testing.assert_allclose(fs.link_inertia[0], R @ I @ R.T)
    root = ET.fromstring(build_urdf(cfg))
    inertial = root.find(f"link[@name='{link.name}']/inertial")
    assert float(inertial.find('mass').get('value')) == pytest.approx(2.3)
    assert float(inertial.find('inertia').get('ixy')) == pytest.approx(0.001)
    np.testing.assert_allclose(np.fromstring(inertial.find('origin').get('xyz'), sep=' '), [0.03, -0.02, 0.05])


@pytest.mark.parametrize('path,value,match', [
    (('joints', 0, 'link', 'length'), -1, 'length'),
    (('materials', 'aluminium_6061', 'density'), float('nan'), 'density'),
    (('actuators', 'big_shoulder', 'efficiency'), 1.2, 'efficiency'),
    (('actuators', 'big_shoulder', 'gear_ratio'), 0, 'gear_ratio'),
    (('actuators', 'big_shoulder', 'friction'), -0.1, 'friction'),
    (('joints', 0, 'axis'), [float('inf'), 0, 1], 'finite'),
    (('joints', 0, 'type'), 'fixed', 'type'),
    (('joints', 0, 'limits', 'lower'), 100, 'lower'),
    (('joints',), [], 'joint'),
])
def test_invalid_physics_rejected(raw, tmp_path, path, value, match):
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        load_variant(raw, tmp_path)


@pytest.mark.parametrize('inertial', [
    {'mass': 2, 'com': [0, 0, 0]},
    {'mass': -1, 'com': [0, 0, 0], 'inertia': [1, 1, 1, 0, 0, 0]},
    {'mass': 2, 'com': [0, 0, 0], 'inertia': [1, 1, 3, 0, 0, 0]},
    {'mass': 2, 'com': [0, 0, 0], 'inertia': [1, 1, 1, 2, 0, 0]},
])
def test_incomplete_or_nonphysical_tensor_rejected(raw, tmp_path, inertial):
    raw['joints'][0]['link']['inertial'] = inertial
    with pytest.raises(ValueError, match='inertial'):
        load_variant(raw, tmp_path)


def test_holding_power_includes_electrical_copper_loss():
    from arm_lab_model.reference import resistance_at
    cfg = load_config()
    model = ArmModel(cfg)
    q = model.resolve_pose('home')
    tau = model.gravity_torque(q)
    power = model.power(q, np.zeros(cfg.dof))
    for i, joint in enumerate(cfg.joints):
        a = joint.actuator
        current = abs(tau[i]) / (a.torque_constant * a.gear_ratio * a.efficiency)
        expected = current**2 * resistance_at(a.phase_resistance, a.ambient_temp) + a.quiescent_power
        assert power['joint_elec_w'][i] == pytest.approx(expected)


def test_viscous_damping_is_output_side_and_matches_export(raw, tmp_path):
    raw['actuators']['big_shoulder']['viscous_damping'] = 0.25
    cfg = load_variant(raw, tmp_path)
    model = ArmModel(cfg)
    q = model.resolve_pose('home')
    v = np.full(cfg.dof, 0.3)
    difference = model.inverse_dynamics(q, v) - model.inverse_dynamics(q, v, include_friction=False)
    np.testing.assert_allclose(difference, model.friction*np.tanh(v/0.02) + model.viscous_damping*v)
    root = ET.fromstring(build_urdf(cfg))
    for joint in cfg.joints:
        tag = root.find(f"joint[@name='{joint.name}']/dynamics")
        assert float(tag.get('damping')) == pytest.approx(joint.actuator.viscous_damping)
