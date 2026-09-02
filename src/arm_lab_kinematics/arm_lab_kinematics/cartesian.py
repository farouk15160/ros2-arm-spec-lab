"""Straight-line Cartesian motion with a genuine TCP speed limit.

The joint-space trajectory controller cannot hold a Cartesian speed limit: it
interpolates in joint space, so the real path bows away from the straight line
and the speed peaks well above its own average. Planning the line in Cartesian
space and solving IK at every waypoint fixes both, and makes a quoted TCP speed
something the system enforces rather than something it hopes for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from arm_lab_model.kinematics import ArmModel, axis_angle_to_matrix

from .ik import IKSolver, rotation_log

from .singularity import metrics as singularity_metrics


@dataclass
class CartesianPath:
    times: np.ndarray                 # seconds, monotonically increasing
    positions: np.ndarray             # (k, 3) commanded TCP positions
    rotations: List[np.ndarray]       # k commanded TCP rotations
    joints: np.ndarray                # (k, n) IK solutions
    tcp_speeds: np.ndarray            # (k,) achieved along-path speed
    joint_speeds: np.ndarray          # (k, n)
    feasible: bool
    notes: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0

    @property
    def peak_tcp_speed(self) -> float:
        return float(np.max(self.tcp_speeds)) if len(self.tcp_speeds) else 0.0

    @property
    def path_length(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())


def slerp(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    """Shortest-arc interpolation between two rotations."""
    delta = rotation_log(R0.T @ R1)
    angle = float(np.linalg.norm(delta))
    if angle < 1e-12:
        return R0.copy()
    return R0 @ axis_angle_to_matrix(delta / angle, angle * s)


def trapezoidal(distance: float, speed: float, accel: float,
                dt: float) -> np.ndarray:
    """Arc-length samples of a trapezoidal (or triangular) speed profile."""
    if distance <= 1e-12:
        return np.array([0.0])
    accel = max(accel, 1e-6)
    speed = max(speed, 1e-6)
    ramp = speed / accel
    ramp_distance = 0.5 * accel * ramp ** 2
    if 2.0 * ramp_distance > distance:              # never reaches cruise
        ramp = math.sqrt(distance / accel)
        speed = accel * ramp
        cruise_time = 0.0
    else:
        cruise_time = (distance - 2.0 * ramp_distance) / speed
    total = 2.0 * ramp + cruise_time

    steps = max(int(math.ceil(total / dt)), 2)
    t = np.linspace(0.0, total, steps + 1)
    s = np.empty_like(t)
    for k, tk in enumerate(t):
        if tk < ramp:
            s[k] = 0.5 * accel * tk ** 2
        elif tk < ramp + cruise_time:
            s[k] = ramp_distance + speed * (tk - ramp)
        else:
            tail = total - tk
            s[k] = distance - 0.5 * accel * tail ** 2
    return np.clip(s, 0.0, distance), t


def plan_line(model: ArmModel,
              q_start: Sequence[float],
              target_position: Sequence[float],
              target_rotation: Optional[np.ndarray] = None,
              speed: Optional[float] = None,
              accel: Optional[float] = None,
              dt: float = 0.02,
              solver: Optional[IKSolver] = None,
              singular_threshold: float = 0.02,
              collision_checker=None,
              collision_retries: int = 4) -> CartesianPath:
    """Plan a straight line from the current TCP pose to the target.

    The path is time-scaled uniformly if any joint would exceed its speed limit,
    so the result is always executable -- slower than asked, never illegal.

    Pass a `collision_checker` and each waypoint is re-solved from fresh seeds
    until a self-collision-free posture is found. Without it the solver has no
    reason to prefer one branch of the nullspace over another and will happily
    thread the tool through the forearm.
    """
    cfg = model.cfg
    motion = cfg.raw.get('motion', {}) if hasattr(cfg, 'raw') else {}
    speed = speed if speed is not None else float(
        motion.get('cartesian_speed', cfg.control.get('tcp_speed_limit', 0.2)))
    accel = accel if accel is not None else float(motion.get('cartesian_accel', 0.3))
    solver = solver or IKSolver(model)
    notes: List[str] = []

    q_start = np.asarray(q_start, dtype=float)
    fs0 = model.frames(q_start, light=True)
    p0, R0 = fs0.tcp.copy(), fs0.tcp_R.copy()
    p1 = np.asarray(target_position, dtype=float)
    R1 = R0.copy() if target_rotation is None else np.asarray(target_rotation)

    distance = float(np.linalg.norm(p1 - p0))
    arc, times = trapezoidal(distance, speed, accel, dt)
    fractions = arc / distance if distance > 1e-12 else np.zeros_like(arc)

    positions = np.array([p0 + (p1 - p0) * f for f in fractions])
    rotations = [slerp(R0, R1, float(f)) for f in fractions]

    joints = np.empty((len(fractions), model.n))
    seed = q_start.copy()
    worst_sigma = math.inf
    colliding = 0
    for k in range(len(fractions)):
        chosen = None
        fallback = None
        attempts = 1 if collision_checker is None else max(collision_retries, 1)
        for attempt in range(attempts):
            res = solver.solve(positions[k], rotations[k],
                               seed=seed if attempt == 0 else None,
                               restarts=1,
                               rng=np.random.default_rng(7919 + attempt))
            if not res.success:
                continue
            fallback = fallback or res
            if collision_checker is None or collision_checker.is_free(res.q):
                chosen = res
                break
        res = chosen or fallback
        if res is None:
            notes.append(f'IK failed at waypoint {k} of {len(fractions)}, '
                         f'{arc[k]:.3f} m into the path')
            return CartesianPath(times[:k], positions[:k], rotations[:k],
                                 joints[:k], np.zeros(k),
                                 np.zeros((k, model.n)), False, notes)
        if chosen is None:
            colliding += 1
        joints[k] = res.q
        seed = res.q
        sigma = singularity_metrics(model, res.q).sigma_min_full
        worst_sigma = min(worst_sigma, sigma)

    if colliding:
        notes.append(f'{colliding} of {len(fractions)} waypoints have no '
                     'self-collision-free posture; the arm cannot follow this '
                     'line without hitting itself')

    if worst_sigma < singular_threshold:
        notes.append(f'passes close to a singularity (smallest singular value '
                     f'{worst_sigma:.4f}); joint speeds will spike there')

    # Uniform time scaling until every joint speed limit is satisfied.
    scale = 1.0
    for _ in range(40):
        scaled_times = times * scale
        joint_speeds = _derivative(joints, scaled_times)
        excess = np.max(np.abs(joint_speeds) / model.velocity_limits)
        if excess <= 1.0 + 1e-6:
            break
        scale *= excess * 1.02
    else:
        notes.append('could not satisfy the joint speed limits by time scaling')

    times = times * scale
    if scale > 1.001:
        notes.append(f'slowed by {scale:.2f}x to stay inside the joint speed '
                     'limits')
    joint_speeds = _derivative(joints, times)
    tcp_speeds = np.linalg.norm(_derivative(positions, times), axis=1)

    return CartesianPath(times, positions, rotations, joints, tcp_speeds,
                         joint_speeds, True, notes)


def _derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Central-difference derivative of a sampled signal."""
    if len(times) < 2:
        return np.zeros_like(values)
    return np.gradient(values, times, axis=0, edge_order=1)
