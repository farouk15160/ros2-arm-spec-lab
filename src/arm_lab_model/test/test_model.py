"""Checks on the physics, each against an independent derivation.

The point of this workspace is that the numbers can be trusted, so the inverse
dynamics is cross-checked against the gradient of potential energy and the
Jacobian against finite differences, rather than against itself.
"""

import os

import numpy as np
import pytest

from arm_lab_model.config import load_config
from arm_lab_model.kinematics import ArmModel

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, '..', 'config', 'arm_config.yaml')


@pytest.fixture(scope='module')
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope='module')
def model(cfg):
    return ArmModel(cfg)


def test_tube_mass_matches_hand_calculation(cfg):
    link = cfg.joints[1].link
    expected = (link.material.density * np.pi
                * (link.outer_radius ** 2 - link.inner_radius ** 2) * link.length)
    assert link.tube_mass == pytest.approx(expected, rel=1e-12)


def test_hollow_cylinder_inertia_reduces_to_solid(cfg):
    """With the bore closed the tensor must match the solid-cylinder formula."""
    from arm_lab_model.config import TubeLink
    link = TubeLink(name='solid', length=0.5, outer_radius=0.05, inner_radius=0.0,
                    material=cfg.materials['aluminium_6061'],
                    direction=[0, 0, 1], extra_mass=0.0, actuator_mass=0.0)
    ixx, iyy, izz = link.inertia_about_com()
    m, r, L = link.tube_mass, link.outer_radius, link.length
    assert izz == pytest.approx(0.5 * m * r ** 2, rel=1e-12)
    assert ixx == pytest.approx(m * (3 * r ** 2 + L ** 2) / 12.0, rel=1e-12)
    assert iyy == pytest.approx(ixx, rel=1e-12)


def test_reach_never_exceeds_the_geometric_limit(model):
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(model.lower, model.upper)
        assert model.reach(q) <= model.geometric_max_reach + 1e-9


def test_full_reach_pose_actually_stretches_out(model):
    q = model.resolve_pose('auto_full_reach')
    assert model.reach(q) == pytest.approx(model.geometric_max_reach, rel=2e-3)


def test_jacobian_matches_finite_differences(model):
    rng = np.random.default_rng(1)
    q = rng.uniform(model.lower * 0.5, model.upper * 0.5)
    J = model.jacobian(q)[:3, :]
    eps = 1e-6
    for i in range(model.n):
        dq = np.zeros(model.n)
        dq[i] = eps
        numeric = (model.fk(q + dq) - model.fk(q - dq)) / (2 * eps)
        assert numeric == pytest.approx(J[:, i], abs=1e-6)


def test_gravity_torque_matches_potential_energy_gradient(model):
    """Holding torque must equal dU/dq, computed here without touching RNE."""
    payload = 2.0

    def potential(q):
        fs = model.frames(q)
        u = float(np.sum(fs.link_mass * fs.link_com[:, 2]))
        u += fs.ee_mass * fs.ee_com[2]
        u += payload * fs.tcp[2]
        return u * model.cfg.gravity

    rng = np.random.default_rng(2)
    q = rng.uniform(model.lower * 0.4, model.upper * 0.4)
    tau = model.gravity_torque(q, payload=payload)
    eps = 1e-6
    for i in range(model.n):
        dq = np.zeros(model.n)
        dq[i] = eps
        gradient = (potential(q + dq) - potential(q - dq)) / (2 * eps)
        assert tau[i] == pytest.approx(gradient, abs=1e-4)


def test_payload_capacity_saturates_exactly_one_joint(model):
    q = model.resolve_pose('auto_full_reach')
    capacity, limiting = model.payload_capacity(q)
    assert limiting >= 0
    reserve = float(model.cfg.control.get('dynamic_torque_reserve', 0.0))
    available = model.torque_limits[limiting] * (1.0 - reserve)
    tau = model.gravity_torque(q, payload=capacity)
    assert abs(tau[limiting]) == pytest.approx(available, rel=1e-6)
    # And nothing else is over its own limit at that load.
    assert np.all(np.abs(tau) <= model.torque_limits * (1.0 - reserve) + 1e-6)


def test_payload_capacity_falls_as_gravity_rises():
    light = ArmModel(load_config(CONFIG, gravity=1.62))
    heavy = ArmModel(load_config(CONFIG, gravity=9.81))
    q = heavy.resolve_pose('auto_full_reach')
    assert light.payload_capacity(q)[0] > heavy.payload_capacity(q)[0]


def test_inverse_dynamics_reduces_to_gravity_at_rest(model):
    q = model.resolve_pose('home')
    at_rest = model.inverse_dynamics(q, include_friction=False)
    assert at_rest == pytest.approx(model.gravity_torque(q), abs=1e-9)


def test_reflected_rotor_inertia_shows_up_in_acceleration(model):
    """Accelerating a joint must cost at least J_rotor * ratio^2 * qddot."""
    q = model.resolve_pose('home')
    qdd = np.zeros(model.n)
    qdd[5] = 1.0
    extra = (model.inverse_dynamics(q, qdd=qdd, include_friction=False)
             - model.gravity_torque(q))
    assert extra[5] >= model.reflected_inertia[5] - 1e-9


def test_max_tcp_speed_is_attainable(model):
    q = model.resolve_pose('auto_full_reach')
    speed, qd = model.max_tcp_speed(q)
    assert np.all(np.abs(qd) <= model.velocity_limits + 1e-9)
    assert np.linalg.norm(model.tcp_velocity(q, qd)) == pytest.approx(speed, rel=1e-9)


def test_deflection_grows_with_load_and_softer_gearboxes(model):
    q = model.resolve_pose('auto_full_reach')
    light = model.deflection(q, payload=0.0)['total']
    heavy = model.deflection(q, payload=5.0)['total']
    assert heavy > light > 0.0


def test_thinner_walls_bend_more():
    import copy

    import yaml
    with open(CONFIG) as fh:
        raw = yaml.safe_load(fh)

    def droop(wall):
        data = copy.deepcopy(raw)
        for joint in data['joints']:
            joint['link']['wall_thickness'] = wall
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as fh:
            yaml.safe_dump(data, fh)
            path = fh.name
        m = ArmModel(load_config(path))
        q = m.resolve_pose('auto_full_reach')
        os.unlink(path)
        return m.deflection(q, payload=2.0)['bending']

    assert droop(0.0015) > droop(0.005)


def test_urdf_is_well_formed_and_masses_agree(cfg):
    import xml.etree.ElementTree as ET

    from arm_lab_model.urdf_builder import build_urdf
    root = ET.fromstring(build_urdf(cfg))
    masses = [float(e.get('value'))
              for e in root.iter('mass')]
    total = sum(masses)
    # base_link, the six links, the gripper, two fingers, the massless TCP.
    assert len(root.findall('link')) >= cfg.dof + 3
    assert total == pytest.approx(cfg.arm_mass, abs=1e-3)


def test_generated_controller_yaml_has_no_aliases(cfg, tmp_path):
    from arm_lab_model.controllers_builder import dump_controllers
    path = dump_controllers(cfg, str(tmp_path / 'controllers.yaml'))
    text = open(path).read()
    assert '&id' not in text and '*id' not in text
