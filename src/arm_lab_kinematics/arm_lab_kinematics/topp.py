"""Time-optimal parameterisation of a fixed joint path.

Answers "how fast can this arm actually run this path?" using the joint speed
limits, the joint acceleration ceiling and -- the interesting one -- the
actuator torque limits, which is where the speed specification and the torque
specification finally meet.

Method: with the path fixed as q(s), the dynamics collapse into

    tau(s) = a(s) * s_ddot + b(s) * s_dot^2 + c(s)

which is affine in the path acceleration and in the squared path velocity. So at
each point the admissible s_ddot is an interval, the largest feasible s_dot^2 is
found by bisection (the maximum velocity curve), and a forward pass at maximum
acceleration followed by a backward pass at maximum deceleration gives the
fastest profile the limits allow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from arm_lab_model.kinematics import ArmModel

EPS = 1e-9


@dataclass
class TimeOptimalResult:
    times: np.ndarray
    joints: np.ndarray
    joint_speeds: np.ndarray
    joint_accels: np.ndarray
    torques: np.ndarray
    tcp_speeds: np.ndarray
    feasible: bool
    limiting: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0

    @property
    def peak_tcp_speed(self) -> float:
        return float(np.max(self.tcp_speeds)) if len(self.tcp_speeds) else 0.0


def _dynamics_coefficients(model: ArmModel, q, qp, qpp):
    """Split the inverse dynamics into the affine form a*s'' + b*s'^2 + c."""
    c = model.inverse_dynamics(q, include_friction=False)
    a = model.inverse_dynamics(q, qdd=qp, include_friction=False) - c
    b = model.inverse_dynamics(q, qd=qp, qdd=qpp, include_friction=False) - c
    return a, b, c


def _accel_interval(a, b, c, sdot2, qp, qpp, tau_max, acc_max,
                    use_torque: bool = True, use_acceleration: bool = True
                    ) -> Optional[Tuple[float, float]]:
    """Admissible path acceleration at this point, or None if infeasible.

    The two constraint families can be switched off individually so the caller
    can find out *which* of them is binding rather than having to guess.
    """
    lo, hi = -math.inf, math.inf

    def clip(coef, upper, lower):
        nonlocal lo, hi
        if coef > EPS:
            lo = max(lo, lower / coef)
            hi = min(hi, upper / coef)
        elif coef < -EPS:
            lo = max(lo, upper / coef)
            hi = min(hi, lower / coef)
        elif not (lower <= 0.0 <= upper):
            lo, hi = 1.0, -1.0            # forced infeasible

    for i in range(len(a)):
        if use_torque:
            clip(a[i], tau_max[i] - b[i] * sdot2 - c[i],
                 -tau_max[i] - b[i] * sdot2 - c[i])
        if use_acceleration:
            clip(qp[i], acc_max[i] - qpp[i] * sdot2,
                 -acc_max[i] - qpp[i] * sdot2)
    if lo > hi:
        return None
    return lo, hi


def _largest_feasible(a, b, c, qp, qpp, tau_max, acc_max, ceiling: float,
                      **flags) -> float:
    """Biggest s_dot^2 up to `ceiling` that still admits an acceleration."""
    if _accel_interval(a, b, c, ceiling, qp, qpp, tau_max, acc_max,
                       **flags) is not None:
        return ceiling
    lo, hi = 0.0, ceiling
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _accel_interval(a, b, c, mid, qp, qpp, tau_max, acc_max,
                           **flags) is None:
            hi = mid
        else:
            lo = mid
    return lo


def parameterise(model: ArmModel, joints: Sequence[Sequence[float]],
                 torque_fraction: float = 1.0,
                 acceleration_limit: Optional[float] = None,
                 tcp_speed_limit: Optional[float] = None,
                 max_sdot: float = 50.0) -> TimeOptimalResult:
    """Time the given joint path as fast as the limits allow."""
    Q = np.asarray(joints, dtype=float)
    k, n = Q.shape
    notes: List[str] = []
    if k < 3:
        return TimeOptimalResult(np.zeros(k), Q, np.zeros_like(Q),
                                 np.zeros_like(Q), np.zeros_like(Q),
                                 np.zeros(k), False,
                                 notes=['path needs at least three waypoints'])

    cfg = model.cfg
    motion = cfg.raw.get('motion', {})
    if acceleration_limit is None:
        acceleration_limit = float(motion.get('joint_acceleration', 3.0))
    acc_max = np.full(n, acceleration_limit)
    tau_max = model.torque_limits * float(np.clip(torque_fraction, 0.01, 1.0))
    vel_max = model.velocity_limits

    s = np.linspace(0.0, 1.0, k)
    ds = float(s[1] - s[0])
    qp = np.gradient(Q, s, axis=0, edge_order=2)
    qpp = np.gradient(qp, s, axis=0, edge_order=2)

    # ---- maximum velocity curve -----------------------------------------
    mvc2 = np.empty(k)
    limiting = ['-'] * k
    coeffs = []
    for i in range(k):
        a, b, c = _dynamics_coefficients(model, Q[i], qp[i], qpp[i])
        coeffs.append((a, b, c))

        speed_bound = math.inf
        which = 'velocity'
        for j in range(n):
            if abs(qp[i, j]) > EPS:
                bound = vel_max[j] / abs(qp[i, j])
                if bound < speed_bound:
                    speed_bound = bound
                    which = f'speed of {cfg.joint_names[j]}'
        speed_bound = min(speed_bound, max_sdot)
        ceiling = speed_bound ** 2

        # Ask each constraint family on its own how fast it would allow this
        # point to be traversed. Whichever answers lowest is the one binding
        # here -- assuming it is the torque limit is how a tight acceleration
        # ceiling gets mistaken for an actuator problem.
        by_torque = _largest_feasible(a, b, c, qp[i], qpp[i], tau_max, acc_max,
                                      ceiling, use_torque=True,
                                      use_acceleration=False)
        by_accel = _largest_feasible(a, b, c, qp[i], qpp[i], tau_max, acc_max,
                                     ceiling, use_torque=False,
                                     use_acceleration=True)
        candidates = [(ceiling, which), (by_torque, 'torque'),
                      (by_accel, 'joint acceleration')]
        hi2, which = min(candidates, key=lambda pair: pair[0])
        # The joint families interact, so re-check them together.
        hi2 = min(hi2, _largest_feasible(a, b, c, qp[i], qpp[i], tau_max,
                                         acc_max, ceiling))
        mvc2[i] = max(hi2, 0.0)
        limiting[i] = which

    if tcp_speed_limit is not None and tcp_speed_limit > 0:
        for i in range(k):
            Jv = model.jacobian(Q[i])[:3, :]
            tcp_per_s = float(np.linalg.norm(Jv @ qp[i]))
            if tcp_per_s > EPS:
                bound = (tcp_speed_limit / tcp_per_s) ** 2
                if bound < mvc2[i]:
                    mvc2[i] = bound
                    limiting[i] = 'TCP speed limit'

    # ---- forward and backward integration --------------------------------
    # The forward and backward passes each enforce the acceleration interval at
    # their own grid point, but the deceleration the backward pass imposes on a
    # segment is attributed to the point before it, where the admissible
    # interval is slightly different. On a coarse grid that gap is enough to let
    # the realised torque overshoot its budget. Rather than pretend the
    # discretisation is exact, the profile is integrated, the true torques are
    # evaluated, and the velocity curve is pulled down until they comply.
    def integrate(curve):
        sdot2 = np.zeros(k)
        for i in range(k - 1):
            a, b, c = coeffs[i]
            interval = _accel_interval(a, b, c, sdot2[i], qp[i], qpp[i],
                                       tau_max, acc_max)
            top = interval[1] if interval else 0.0
            sdot2[i + 1] = max(min(curve[i + 1], sdot2[i] + 2.0 * top * ds), 0.0)
        sdot2[-1] = 0.0
        for i in range(k - 2, -1, -1):
            a, b, c = coeffs[i + 1]
            interval = _accel_interval(a, b, c, sdot2[i + 1], qp[i + 1],
                                       qpp[i + 1], tau_max, acc_max)
            bottom = interval[0] if interval else 0.0
            sdot2[i] = max(min(sdot2[i], sdot2[i + 1] - 2.0 * bottom * ds), 0.0)
        return sdot2

    def evaluate(sdot2):
        sdot = np.sqrt(np.maximum(sdot2, 0.0))
        times = np.zeros(k)
        for i in range(k - 1):
            pair = sdot[i] + sdot[i + 1]
            times[i + 1] = times[i] + (2.0 * ds / pair if pair > 1e-6 else 0.0)
        sddot = np.zeros(k)
        sddot[:-1] = (sdot2[1:] - sdot2[:-1]) / (2.0 * ds)
        if k > 1:
            sddot[-1] = sddot[-2]
        qd = qp * sdot[:, None]
        qdd = qp * sddot[:, None] + qpp * sdot2[:, None]
        torques = np.array([
            model.inverse_dynamics(Q[i], qd[i], qdd[i], include_friction=False)
            for i in range(k)])
        return times, sdot, qd, qdd, torques

    curve = mvc2.copy()
    for _ in range(14):
        sdot2 = integrate(curve)
        times, sdot, qd, qdd, torques = evaluate(sdot2)
        over = float(np.max(np.abs(torques) / np.maximum(tau_max, EPS)))
        if over <= 1.0 + 1e-3 or float(np.max(sdot2)) <= 1e-12:
            break
        # Torque scales roughly with sdot^2 along the path, so this converges
        # in a handful of rounds.
        curve = np.minimum(curve, np.maximum(sdot2 / over, 0.0))
    else:
        notes.append('could not bring the torques inside the budget by slowing '
                     'down; gravity alone may exceed it')

    _unused_sdot2 = np.zeros(k)
    _unused_sdot2[0] = 0.0
    if not np.all(np.isfinite(times)):
        notes.append('the path stalls: some segment admits no motion at all')
        return TimeOptimalResult(times, Q, np.zeros_like(Q), np.zeros_like(Q),
                                 np.zeros_like(Q), np.zeros(k), False,
                                 limiting, notes)

    tcp = np.array([
        float(np.linalg.norm(model.jacobian(Q[i])[:3, :] @ qd[i]))
        for i in range(k)])

    # Which constraint is actually binding: look at where the profile is riding
    # its own maximum velocity curve, not at where the torque merely peaks.
    active = np.abs(sdot2 - curve) <= 1e-6 * np.maximum(curve, 1.0)
    if active.any():
        labels, counts = np.unique(np.array(limiting)[active], return_counts=True)
        notes.append('binding constraint: '
                     + ', '.join(f'{lab} ({cnt * 100 // active.sum()} % of the path)'
                                 for lab, cnt in
                                 sorted(zip(labels, counts), key=lambda x: -x[1])))
    else:
        notes.append('binding constraint: joint acceleration (the speed limits '
                     'are never reached)')

    stalled = times[-1] <= 1e-9 or float(np.max(sdot)) <= 1e-9
    if stalled:
        notes.append('the arm cannot move along this path at all within the '
                     'given limits: gravity alone exceeds the torque budget, so '
                     'there is no speed slow enough to make it work')

    utilisation = float(np.max(np.abs(torques) / np.maximum(tau_max, EPS)))
    notes.append(f'peak torque {utilisation * 100:.0f} % of the '
                 f'{torque_fraction * 100:.0f} % budget allowed')
    if utilisation > 1.02:
        notes.append('WARNING: the torque bound is violated; the path likely '
                     'demands more than gravity compensation alone can hold')
    return TimeOptimalResult(times, Q, qd, qdd, torques, tcp, not stalled,
                             limiting, notes)
