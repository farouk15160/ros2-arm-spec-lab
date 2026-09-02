"""Full 6-DOF inverse kinematics: position and orientation.

Damped least squares with adaptive damping near singularities and joint-limit
avoidance in the nullspace, plus random restarts so a hard target is not lost to
one bad seed.

The task vector mixes metres and radians, which is meaningless unless the two
are put on a common scale. Orientation is therefore multiplied by a
characteristic length, so a `char_length` of 0.15 m says "one radian of
orientation error matters as much as 150 mm of position error". Every residual
reported back is in its own natural unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from arm_lab_model.kinematics import ArmModel, rpy_to_matrix


def rotation_log(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix, robust at 0 and pi radians."""
    trace = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(trace)
    if theta < 1e-8:
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    if abs(theta - math.pi) < 1e-6:
        # Near pi the skew part vanishes; recover the axis from R + I instead.
        M = (R + np.eye(3)) / 2.0
        diag = np.sqrt(np.maximum(np.diag(M), 0.0))
        k = int(np.argmax(diag))
        axis = M[:, k] / diag[k] if diag[k] > 1e-9 else np.array([1.0, 0.0, 0.0])
        axis = axis / (np.linalg.norm(axis) or 1.0)
        return axis * theta
    return np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) * (theta / (2.0 * math.sin(theta)))


def orientation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Rotation vector taking `R_current` onto `R_target`, in world axes."""
    return rotation_log(R_target @ R_current.T)


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    position_error: float          # metres
    orientation_error: float       # radians
    iterations: int
    restarts: int = 0
    manipulability: float = 0.0

    @property
    def orientation_error_deg(self) -> float:
        return math.degrees(self.orientation_error)


class IKSolver:
    """Damped-least-squares IK over an :class:`ArmModel`."""

    def __init__(self, model: ArmModel,
                 position_tolerance: float = 1e-4,
                 orientation_tolerance: float = 1e-3,
                 max_iterations: int = 150,
                 base_damping: float = 0.008,
                 singular_threshold: float = 0.02,
                 max_damping: float = 0.12,
                 char_length: float = 0.15,
                 nullspace_gain: float = 0.08,
                 step_clamp: float = 0.25):
        self.model = model
        self.n = model.n
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.max_iterations = max_iterations
        self.base_damping = base_damping
        self.singular_threshold = singular_threshold
        self.max_damping = max_damping
        self.char_length = char_length
        self.nullspace_gain = nullspace_gain
        self.step_clamp = step_clamp
        self.lower = model.lower
        self.upper = model.upper
        self._mid = 0.5 * (self.lower + self.upper)
        self._range = np.maximum(self.upper - self.lower, 1e-6)

    @classmethod
    def survey(cls, model: ArmModel, **kwargs) -> 'IKSolver':
        """A cheaper solver for workspace sweeps.

        Mapping asks the same question thousands of times and most answers are
        "no". A failed solve costs the full iteration budget, so the budget is
        what has to come down; a survey trades a slightly pessimistic dexterity
        figure for a run that finishes in under a minute.
        """
        defaults = dict(position_tolerance=5e-4, orientation_tolerance=5e-3,
                        max_iterations=60)
        defaults.update(kwargs)
        return cls(model, **defaults)

    # ------------------------------------------------------------- internals
    def _limit_gradient(self, q: np.ndarray) -> np.ndarray:
        """Downhill direction of a joint-centring cost, for the nullspace."""
        return -(q - self._mid) / (self._range ** 2) / self.n

    def _damping(self, J: np.ndarray) -> Tuple[float, float]:
        """Damping factor, plus how far this pose is from being singular.

        Returns (lambda, conditioning) where conditioning is 1 in a healthy pose
        and falls to 0 at a singularity.
        """
        sigma_min = float(np.linalg.svd(J, compute_uv=False)[-1])
        conditioning = float(np.clip(sigma_min / self.singular_threshold, 0.0, 1.0))
        if conditioning >= 1.0:
            return self.base_damping, 1.0
        return (self.base_damping + self.max_damping * (1.0 - conditioning ** 2),
                conditioning)

    def _task(self, q: np.ndarray, target_p: np.ndarray,
              target_R: Optional[np.ndarray]):
        fs = self.model.frames(q)
        e_pos = target_p - fs.tcp
        if target_R is None:
            return fs, e_pos, np.zeros(3), e_pos.copy()
        e_rot = orientation_error(fs.tcp_R, target_R)
        stacked = np.concatenate([e_pos, self.char_length * e_rot])
        return fs, e_pos, e_rot, stacked

    def _jacobian(self, q: np.ndarray, fs, oriented: bool) -> np.ndarray:
        J = self.model.jacobian(q, fs)
        if not oriented:
            return J[:3, :]
        return np.vstack([J[:3, :], self.char_length * J[3:, :]])

    def _step(self, q, e, J, lam, conditioning):
        """One damped step, with the joint-centring term in the nullspace."""
        JT = J.T
        m = J.shape[0]
        inv = np.linalg.inv(J @ JT + (lam ** 2) * np.eye(m))
        dq = JT @ (inv @ e)
        # The nullspace projector built from a *damped* inverse is only
        # approximate, so the joint-centring term leaks into the task and stalls
        # the last few micrometres. Fade it out as the error shrinks, and near
        # singularities where the projector is least trustworthy.
        if self.nullspace_gain > 0.0:
            taper = math.tanh(float(np.linalg.norm(e))
                              / (20.0 * self.position_tolerance))
            gain = self.nullspace_gain * taper * conditioning
            if gain > 1e-12:
                null = np.eye(self.n) - (JT @ inv) @ J
                dq = dq + null @ (gain * self._limit_gradient(q))
        norm = float(np.linalg.norm(dq))
        if norm > self.step_clamp:
            dq *= self.step_clamp / norm
        return np.clip(q + dq, self.lower, self.upper)

    def _iterate(self, q: np.ndarray, target_p: np.ndarray,
                 target_R: Optional[np.ndarray]) -> Tuple[np.ndarray, float, float, int]:
        """Levenberg-Marquardt: only accept steps that reduce the residual.

        A plain damped-least-squares iteration walks straight into a local
        minimum and sits there. Rejecting uphill steps and raising the damping
        instead turns most of those stalls into convergence.
        """
        oriented = target_R is not None
        lam = self.base_damping
        fs, e_pos, e_rot, e = self._task(q, target_p, target_R)
        residual = float(np.linalg.norm(e))

        for step in range(self.max_iterations):
            p_err = float(np.linalg.norm(e_pos))
            o_err = float(np.linalg.norm(e_rot))
            if p_err < self.position_tolerance and (
                    not oriented or o_err < self.orientation_tolerance):
                return q, p_err, o_err, step

            J = self._jacobian(q, fs, oriented)
            floor, conditioning = self._damping(J)
            lam = max(lam, floor)

            improved = False
            for _ in range(8):
                q_try = self._step(q, e, J, lam, conditioning)
                fs_try, ep_try, er_try, e_try = self._task(q_try, target_p, target_R)
                res_try = float(np.linalg.norm(e_try))
                if res_try < residual:
                    q, fs, e_pos, e_rot, e = q_try, fs_try, ep_try, er_try, e_try
                    residual = res_try
                    lam = max(lam / 1.6, floor)
                    improved = True
                    break
                lam = min(lam * 2.5, 10.0)
            if not improved:
                break                      # no downhill direction left

        return (q, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot)),
                self.max_iterations)

    # ---------------------------------------------------------------- public
    def _result(self, q, p_err, o_err, iters, attempt, oriented) -> IKResult:
        ok = p_err < self.position_tolerance and (
            not oriented or o_err < self.orientation_tolerance)
        return IKResult(q=q, success=ok, position_error=p_err,
                        orientation_error=o_err, iterations=iters,
                        restarts=attempt)

    @staticmethod
    def _better(candidate: IKResult, best: Optional[IKResult]) -> bool:
        if best is None or (candidate.success and not best.success):
            return True
        if candidate.success != best.success:
            return False
        return (candidate.position_error + candidate.orientation_error
                < best.position_error + best.orientation_error)

    def solve(self, target_position: Sequence[float],
              target_rotation: Optional[np.ndarray] = None,
              seed: Optional[Sequence[float]] = None,
              restarts: int = 8,
              wrist_reseeds: int = 5,
              rng: Optional[np.random.Generator] = None) -> IKResult:
        """Solve for `target_position` and, if given, `target_rotation`.

        With an orientation target this runs in two stages. Solving the full
        6-DOF task from a random seed drops into a local minimum surprisingly
        often, because the solver will happily trade position error away to
        reduce orientation error. Getting the position right first and only then
        turning on the orientation term avoids almost all of those, and when it
        does stall, re-seeding just the last three joints explores the wrist
        configurations where the orientation freedom actually lives.
        """
        target_p = np.asarray(target_position, dtype=float)
        target_R = None if target_rotation is None else np.asarray(
            target_rotation, dtype=float)
        oriented = target_R is not None
        rng = rng or np.random.default_rng(0)

        seeds: List[np.ndarray] = []
        if seed is not None:
            seeds.append(np.clip(np.asarray(seed, dtype=float),
                                 self.lower, self.upper))
        seeds.append(self._mid.copy())
        for _ in range(max(restarts, 0)):
            seeds.append(rng.uniform(self.lower, self.upper))

        best: Optional[IKResult] = None
        total_iterations = 0
        wrist = slice(max(self.n - 3, 0), self.n)

        for attempt, start_q in enumerate(seeds):
            # Stage one: position only. This part is reliable.
            q_pos, p_err, _, iters = self._iterate(start_q.copy(), target_p, None)
            total_iterations += iters
            if not oriented:
                candidate = self._result(q_pos, p_err, 0.0, total_iterations,
                                         attempt, False)
                if self._better(candidate, best):
                    best = candidate
                if candidate.success:
                    break
                continue

            if p_err > self.position_tolerance * 20.0:
                continue                   # this seed cannot even reach the point

            # Stage two: add orientation, re-seeding the wrist when it stalls.
            solved = False
            for trial in range(max(wrist_reseeds, 1)):
                q_start = q_pos.copy()
                if trial > 0:
                    q_start[wrist] = rng.uniform(self.lower[wrist], self.upper[wrist])
                q, p_err2, o_err2, iters = self._iterate(q_start, target_p, target_R)
                total_iterations += iters
                candidate = self._result(q, p_err2, o_err2, total_iterations,
                                         attempt, True)
                if self._better(candidate, best):
                    best = candidate
                if candidate.success:
                    solved = True
                    break
            if solved:
                break

        if best is None:                   # every seed failed to reach the point
            q, p_err, o_err, iters = self._iterate(
                self._mid.copy(), target_p, target_R)
            best = self._result(q, p_err, o_err, iters, len(seeds), oriented)

        J = self.model.jacobian(best.q)[:3, :]
        best.manipulability = float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
        return best

    def solve_pose(self, position: Sequence[float], rpy: Sequence[float],
                   **kwargs) -> IKResult:
        """Convenience wrapper taking roll/pitch/yaw instead of a matrix."""
        return self.solve(position, rpy_to_matrix(rpy), **kwargs)

    def reachable(self, position: Sequence[float],
                  rotation: Optional[np.ndarray] = None,
                  restarts: int = 3) -> bool:
        return self.solve(position, rotation, restarts=restarts).success
