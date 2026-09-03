"""Checks on the physics, each against an independent derivation.

The point of this workspace is that the numbers can be trusted, so the inverse
dynamics is cross-checked against the gradient of potential energy and the
Jacobian against finite differences, rather than against itself.
"""

import math
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
    """Within a whisker of the geometric limit, but not exactly on it.

    `geometric_max_reach` is the sum of the link lengths: it assumes every link
    is perfectly collinear. The generated pose also has to keep the tool axis
    aligned with the reach direction, which is what stops the solver folding the
    wrist, and paying for that alignment costs a few millimetres of extension.
    Demanding the full geometric reach here would be demanding a pose that
    doubles the tool back on itself.
    """
    q = model.resolve_pose('auto_full_reach')
    reach = model.reach(q)
    assert reach <= model.geometric_max_reach + 1e-9
    assert reach >= 0.99 * model.geometric_max_reach


def test_generated_reach_poses_hit_their_radius(model):
    """Radii inside the achievable band are hit exactly.

    The band has a floor: with the tool held pointing outward the arm cannot
    fold below roughly the length of its own wrist and tool, so a request below
    that returns the closest achievable pose rather than the requested radius.
    """
    for radius in (0.5, 0.7, 0.9):
        q = model.resolve_pose(f'auto_reach_{radius:.3f}')
        assert model.reach(q) == pytest.approx(radius, abs=2e-3)


def test_unreachably_tight_radius_returns_the_closest_pose(model):
    """Below the floor it returns the nearest pose it can, not an error.

    Holding the tool pointing outward, this arm cannot place its TCP 100 mm from
    the shoulder. What it can do is fold back on itself until the TCP sits close
    to the shoulder axis, and that is what comes back -- a legal, in-limits pose
    whose reach is simply not the number that was asked for.
    """
    q = model.resolve_pose('auto_reach_0.100')
    assert np.all(q >= model.lower - 1e-9)
    assert np.all(q <= model.upper + 1e-9)
    assert model.reach(q) < 0.5


def test_reach_poses_keep_the_tool_pointing_outward(model):
    """The property that keeps these poses out of self-collision."""
    for radius in (0.5, 0.7, 0.9):
        q = model.resolve_pose(f'auto_reach_{radius:.3f}')
        fs = model.frames(q)
        outward = fs.tcp - fs.reach_origin
        outward = outward / np.linalg.norm(outward)
        alignment = float(fs.link_dir[-1] @ outward)
        assert alignment > 0.9, (
            f'at r={radius} the tool axis is {math.degrees(math.acos(alignment)):.0f}'
            ' deg off the reach direction, so the arm is folded')


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


# ---------------------------------------------------------------------------
# Physics verification against independent implementations. These are the
# checks that decide whether anything else in the workspace can be believed.
# ---------------------------------------------------------------------------

def test_dynamics_matches_orocos_kdl(cfg):
    """Cross-check the inverse dynamics against a different team's RNE.

    KDL is fed the *generated URDF*, so this validates the dynamics and the
    URDF export together.
    """
    from arm_lab_model import verification
    if not verification.HAVE_KDL:
        pytest.skip('PyKDL not available')
    plain = verification._fingerless(cfg)
    for gravity_only in (True, False):
        check = verification.compare_dynamics(plain, samples=60,
                                              gravity_only=gravity_only)
        assert check.passed, check.detail


def test_forward_kinematics_matches_orocos_kdl(cfg):
    from arm_lab_model import verification
    if not verification.HAVE_KDL:
        pytest.skip('PyKDL not available')
    check = verification.compare_forward_kinematics(
        verification._fingerless(cfg), samples=60)
    assert check.passed, check.detail


def test_joint_power_equals_rate_of_change_of_energy(cfg):
    """Newton-Euler against the Lagrangian statement of the same mechanics."""
    from arm_lab_model import verification
    check = verification.check_energy_balance(cfg, samples=30)
    assert check.passed, check.detail


def test_point_mass_pendulum_is_exact():
    from arm_lab_model import verification
    check = verification.check_point_mass_pendulum()
    assert check.passed, check.detail


def test_end_effector_rotational_inertia_is_modelled(model):
    """Accelerating the wrist about the tool must cost more than a point mass.

    The tool was modelled as a point mass at first; KDL caught it.
    """
    q = model.resolve_pose('home')
    fs = model.frames(q)
    assert fs.ee_inertia.any(), 'the tool has no rotational inertia'
    qdd = np.zeros(model.n)
    qdd[-1] = 5.0
    with_inertia = model.inverse_dynamics(q, qdd=qdd, include_friction=False)
    baseline = model.gravity_torque(q)
    extra = abs(with_inertia[-1] - baseline[-1])
    assert extra > model.reflected_inertia[-1] * 5.0


def test_urdf_gripper_centre_of_mass_matches_the_config(cfg):
    """`com_offset` describes the whole tool, jaws included.

    The URDF splits the tool into a body and two jaw links, so the body has to
    be placed such that the combined centre of mass lands back on the configured
    offset. Getting this wrong left the URDF a few centimetres of lever arm
    heavier than the model, which showed up against Gazebo as a constant
    0.094 N.m offset on the wrist joints.
    """
    import xml.etree.ElementTree as ET

    from arm_lab_model.urdf_builder import build_urdf
    from arm_lab_model.kinematics import ArmModel

    ee = cfg.end_effector
    if not ee.simulate_fingers:
        pytest.skip('jaws not simulated')

    root = ET.fromstring(build_urdf(cfg))
    model = ArmModel(cfg)
    flange = np.asarray(cfg.joints[-1].link.direction, dtype=float)

    total_mass = 0.0
    moment = np.zeros(3)
    for link in root.findall('link'):
        name = link.get('name')
        if not (name.startswith(ee.name) or name == 'tcp_link'):
            continue
        inertial = link.find('inertial')
        if inertial is None:
            continue
        mass = float(inertial.find('mass').get('value'))
        origin = inertial.find('origin')
        text = '0 0 0' if origin is None else (origin.get('xyz') or '0 0 0')
        local = np.array([float(v) for v in text.split()])
        # The jaw links hang off the gripper body, so add their mount offset.
        if 'finger' in name:
            local = local + flange * ee.body_length
        total_mass += mass
        moment += mass * local

    assert total_mass == pytest.approx(ee.mass, abs=1e-5)
    centre = moment / total_mass
    assert float(centre @ flange) == pytest.approx(ee.com_offset, abs=1e-6)
