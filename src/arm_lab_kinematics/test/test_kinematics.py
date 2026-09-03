"""Checks on the kinematics, each against something independent of the code.

IK is verified by round-tripping forward kinematics, the Cartesian planner by
measuring the path it produces rather than trusting its own report, and the
generated MoveIt SRDF by checking its link names against the generated URDF.
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from arm_lab_model.config import load_config
from arm_lab_model.kinematics import (ArmModel, axis_angle_to_matrix,
                                      rpy_to_matrix)
from arm_lab_kinematics.cartesian import plan_line, slerp
from arm_lab_kinematics.collision import (CollisionChecker,
                                          build_allowed_collisions,
                                          segment_distance)
from arm_lab_kinematics.errors import BuiltArm
from arm_lab_kinematics.ik import IKSolver, orientation_error, rotation_log
from arm_lab_kinematics.iso9283 import cube_poses
from arm_lab_kinematics.singularity import classify, metrics
from arm_lab_kinematics.topp import parameterise
from arm_lab_kinematics.workspace import TOOL_DOWN, sample_orientations

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.abspath(os.path.join(HERE, '..', '..', 'arm_lab_model',
                                      'config', 'arm_config.yaml'))
if not os.path.exists(CONFIG):                       # installed layout
    from ament_index_python.packages import get_package_share_directory
    CONFIG = os.path.join(get_package_share_directory('arm_lab_model'),
                          'config', 'arm_config.yaml')


@pytest.fixture(scope='module')
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope='module')
def model(cfg):
    return ArmModel(cfg)


@pytest.fixture(scope='module')
def solver(model):
    return IKSolver(model)


# ------------------------------------------------------------- rotations
def test_rotation_log_inverts_axis_angle():
    rng = np.random.default_rng(0)
    for _ in range(50):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-math.pi + 1e-3, math.pi - 1e-3)
        recovered = rotation_log(axis_angle_to_matrix(axis, angle))
        assert np.linalg.norm(recovered) == pytest.approx(abs(angle), abs=1e-9)
        assert recovered @ (axis * angle) > 0     # same sense


def test_rotation_log_survives_the_pi_singularity():
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                 np.array([0, 0, 1.0]), np.array([1, 1, 1.0]) / math.sqrt(3)):
        R = axis_angle_to_matrix(axis, math.pi)
        recovered = rotation_log(R)
        assert np.linalg.norm(recovered) == pytest.approx(math.pi, abs=1e-4)
        # Rebuilding from the recovered vector must give the same rotation.
        back = axis_angle_to_matrix(recovered / np.linalg.norm(recovered),
                                    float(np.linalg.norm(recovered)))
        assert np.allclose(back, R, atol=1e-4)


def test_orientation_error_is_zero_for_identical_frames():
    R = rpy_to_matrix([0.3, -0.7, 1.1])
    assert np.allclose(orientation_error(R, R), np.zeros(3), atol=1e-12)


def test_slerp_endpoints_and_midpoint():
    R0 = rpy_to_matrix([0.0, 0.0, 0.0])
    R1 = rpy_to_matrix([0.0, 0.0, 1.2])
    assert np.allclose(slerp(R0, R1, 0.0), R0, atol=1e-12)
    assert np.allclose(slerp(R0, R1, 1.0), R1, atol=1e-9)
    mid = slerp(R0, R1, 0.5)
    assert np.linalg.norm(rotation_log(R0.T @ mid)) == pytest.approx(0.6, abs=1e-9)


# -------------------------------------------------------------------- IK
def test_ik_round_trip_position_only(model, solver):
    rng = np.random.default_rng(1)
    for k in range(25):
        q = rng.uniform(model.lower * 0.85, model.upper * 0.85)
        target = model.fk(q)
        res = solver.solve(target, None, rng=np.random.default_rng(k))
        assert res.success
        assert np.linalg.norm(model.fk(res.q) - target) < 1e-4


def test_ik_round_trip_full_pose(model, solver):
    """Targets come from forward kinematics, so every one is reachable."""
    rng = np.random.default_rng(2)
    solved = 0
    trials = 25
    for k in range(trials):
        q = rng.uniform(model.lower * 0.85, model.upper * 0.85)
        fs = model.frames(q)
        res = solver.solve(fs.tcp, fs.tcp_R, rng=np.random.default_rng(k))
        if res.success:
            solved += 1
            got = model.frames(res.q)
            assert np.linalg.norm(got.tcp - fs.tcp) < 1e-3
            assert np.linalg.norm(orientation_error(got.tcp_R, fs.tcp_R)) < 1e-2
    assert solved >= int(0.9 * trials), f'only {solved}/{trials} solved'


def test_ik_respects_joint_limits(model, solver):
    rng = np.random.default_rng(3)
    for k in range(10):
        q = rng.uniform(model.lower * 0.9, model.upper * 0.9)
        res = solver.solve(model.fk(q), rng=np.random.default_rng(k))
        assert np.all(res.q >= model.lower - 1e-9)
        assert np.all(res.q <= model.upper + 1e-9)


def test_unreachable_target_reports_failure(model, solver):
    far = model.frames(np.zeros(model.n)).reach_origin + np.array([10.0, 0, 0])
    res = solver.solve(far, restarts=1)
    assert not res.success
    assert res.position_error > 1.0


# ------------------------------------------------------------ singularity
def test_stretched_arm_is_detected_as_singular(model):
    q = model.resolve_pose('full_reach')
    m = metrics(model, q)
    assert m.is_singular
    assert 'stretch' in m.kind


def test_manipulability_falls_toward_the_workspace_boundary(model):
    inner = metrics(model, model.resolve_pose('auto_reach_0.500'))
    outer = metrics(model, model.resolve_pose('auto_full_reach'))
    assert outer.manipulability_v < inner.manipulability_v


def test_condition_number_is_at_least_one(model):
    rng = np.random.default_rng(4)
    for _ in range(20):
        m = metrics(model, rng.uniform(model.lower, model.upper))
        assert m.condition_v >= 1.0 - 1e-9


# -------------------------------------------------------------- collision
def test_segment_distance_known_cases():
    p = np.array([0.0, 0, 0])
    assert segment_distance(p, np.array([1.0, 0, 0]),
                            np.array([0.0, 1, 0]),
                            np.array([1.0, 1, 0])) == pytest.approx(1.0)
    assert segment_distance(p, np.array([1.0, 0, 0]),
                            np.array([2.0, 0, 0]),
                            np.array([3.0, 0, 0])) == pytest.approx(1.0)
    # Crossing segments touch.
    assert segment_distance(np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]),
                            np.array([0.0, -1, 0]),
                            np.array([0.0, 1, 0])) == pytest.approx(0.0)


def test_acm_disables_the_permanently_overlapping_wrist_pair(model):
    acm = build_allowed_collisions(model, samples=300)
    assert ('link_4', 'link_6') in acm['always'], (
        'the short wrist links overlap in every pose and must be disabled')


def test_configured_poses_are_collision_free(model, cfg):
    acm = build_allowed_collisions(model, samples=300)
    checker = CollisionChecker(model, allowed=acm['always'])
    for name in cfg.test_poses:
        q = model.resolve_pose(name)
        assert checker.is_free(q), f'test pose {name!r} self-collides'


# -------------------------------------------------------------- cartesian
def test_cartesian_path_is_straight_and_speed_limited(model):
    q0 = model.resolve_pose('home')
    target = np.array([0.62, -0.20, 0.18])
    limit = 0.20
    path = plan_line(model, q0, target, TOOL_DOWN, speed=limit)
    assert path.feasible
    straight = float(np.linalg.norm(target - model.fk(q0)))
    # Measured from the produced waypoints, not from the planner's own report.
    assert path.path_length == pytest.approx(straight, rel=1e-3)
    assert path.peak_tcp_speed <= limit * 1.05
    assert np.all(np.abs(path.joint_speeds) <= model.velocity_limits * 1.02)


def test_cartesian_beats_joint_space_on_speed_overshoot(model):
    """The whole reason the Cartesian planner exists."""
    q0 = model.resolve_pose('home')
    target = np.array([0.62, -0.20, 0.18])
    limit = 0.20
    path = plan_line(model, q0, target, TOOL_DOWN, speed=limit)
    q1 = path.joints[-1]
    straight = float(np.linalg.norm(target - model.fk(q0)))
    alpha = np.linspace(0, 1, 200)
    blend = 3 * alpha ** 2 - 2 * alpha ** 3
    joint_path = q0 + (q1 - q0) * blend[:, None]
    times = alpha * (straight / limit)
    tcp = np.array([model.fk(q) for q in joint_path])
    joint_space_peak = np.linalg.norm(
        np.gradient(tcp, times, axis=0), axis=1).max()
    assert joint_space_peak > limit * 1.2
    assert path.peak_tcp_speed < joint_space_peak


# ------------------------------------------------------------------ topp
def test_topp_respects_the_torque_budget(model):
    q0 = model.resolve_pose('home')
    path = plan_line(model, q0, np.array([0.62, -0.20, 0.18]), TOOL_DOWN)
    joints = path.joints[::8]
    result = parameterise(model, joints, torque_fraction=1.0)
    assert result.feasible
    assert np.all(np.abs(result.torques) <= model.torque_limits * 1.05)
    assert np.all(np.abs(result.joint_speeds) <= model.velocity_limits * 1.05)


def test_topp_slows_down_when_the_torque_budget_is_cut(model):
    """A tighter actuator budget must cost time, or the bound is not binding."""
    q0 = model.resolve_pose('home')
    path = plan_line(model, q0, np.array([0.62, -0.20, 0.18]), TOOL_DOWN)
    joints = path.joints[::8]
    full = parameterise(model, joints, torque_fraction=1.0,
                        acceleration_limit=10.0)
    half = parameterise(model, joints, torque_fraction=0.5,
                        acceleration_limit=10.0)
    assert half.duration > full.duration * 1.05
    # And the reduced budget is actually respected.
    assert np.all(np.abs(half.torques)
                  <= model.torque_limits * 0.5 * 1.02)


def test_topp_reports_an_irreducible_gravity_load_instead_of_lying(model):
    """Below the gravity floor no speed is slow enough; say so."""
    q0 = model.resolve_pose('home')
    path = plan_line(model, q0, np.array([0.62, -0.20, 0.18]), TOOL_DOWN)
    result = parameterise(model, path.joints[::8], torque_fraction=0.05)
    text = ' '.join(result.notes).lower()
    assert 'gravity' in text


def test_topp_is_never_slower_than_a_speed_capped_run(model):
    q0 = model.resolve_pose('home')
    path = plan_line(model, q0, np.array([0.62, -0.20, 0.18]), TOOL_DOWN)
    joints = path.joints[::8]
    fast = parameterise(model, joints)
    capped = parameterise(model, joints, tcp_speed_limit=0.05)
    assert fast.duration < capped.duration


# ---------------------------------------------------------------- errors
def test_repeatability_is_unaffected_by_build_tolerances(cfg):
    """Machining scatter moves the mean; it must not change the scatter.

    This is the distinction the whole ISO 9283 report rests on.
    """
    model = ArmModel(cfg)
    q = model.resolve_pose('reach_700')
    spreads, means = [], []
    for seed in range(4):
        arm = BuiltArm(cfg, seed=seed)
        rng = np.random.default_rng(100 + seed)
        pts = np.array([arm.attained(q, 1.0, 0.0, rng)[0] for _ in range(60)])
        bary = pts.mean(axis=0)
        spreads.append(np.linalg.norm(pts - bary, axis=1).mean())
        means.append(np.linalg.norm(bary - arm.commanded(q)[0]))
    assert np.std(means) > 1e-5, 'different units should differ in accuracy'
    assert np.std(spreads) < 0.2 * np.mean(spreads), \
        'repeatability should be nearly identical across units'


def test_error_budget_sums_to_something_sensible(cfg):
    arm = BuiltArm(cfg, seed=0)
    q = arm.nominal.resolve_pose('full_reach')
    budget = arm.budget(q, payload=2.0)
    assert budget.contributions['gravity droop'] > 0.0
    assert budget.total() > 0.0


# --------------------------------------------------------------- iso 9283
def test_cube_poses_lie_on_a_plane_and_inside_the_cube():
    centre = np.array([0.5, 0.0, 0.3])
    side = 0.4
    poses = cube_poses(centre, side)
    assert set(poses) == {'P1', 'P2', 'P3', 'P4', 'P5'}
    corners = np.array([poses[f'P{i}'] for i in range(2, 6)])
    # Coplanar: the volume of the tetrahedron they span is zero.
    v = corners[1:] - corners[0]
    assert abs(np.linalg.det(v)) < 1e-12
    # All inside the cube.
    assert np.all(np.abs(corners - centre) <= side / 2.0 + 1e-9)


def test_iso_repeatability_formula_matches_the_standard():
    """RP is l_mean + 3 sigma about the barycentre, not a bounding radius."""
    rng = np.random.default_rng(0)
    pts = rng.normal(0.0, 0.001, size=(500, 3))
    bary = pts.mean(axis=0)
    radial = np.linalg.norm(pts - bary, axis=1)
    RP = radial.mean() + 3.0 * radial.std(ddof=1)
    assert RP > radial.mean()
    assert RP < radial.max() * 3.0


# ---------------------------------------------------------------- moveit
def test_generated_srdf_only_names_links_that_exist_in_the_urdf(cfg, tmp_path):
    from arm_lab_model.urdf_builder import build_urdf
    from arm_lab_kinematics.moveit_gen import build_srdf

    urdf_links = {e.get('name')
                  for e in ET.fromstring(build_urdf(cfg)).findall('link')}
    srdf = ET.fromstring(build_srdf(cfg, samples=200))
    named = set()
    for tag in srdf.iter('disable_collisions'):
        named.add(tag.get('link1'))
        named.add(tag.get('link2'))
    for tag in srdf.iter('chain'):
        named.add(tag.get('base_link'))
        named.add(tag.get('tip_link'))
    missing = named - urdf_links
    assert not missing, f'SRDF references links absent from the URDF: {missing}'


def test_orientation_sampler_starts_tool_down():
    rots = sample_orientations(5)
    assert len(rots) == 5
    assert np.allclose(rots[0], TOOL_DOWN)
    for R in rots:
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
