"""Manipulability, conditioning and singularity classification.

A 6xN Jacobian stacks metres-per-radian on top of dimensionless rows, so its
determinant and condition number have no physical meaning on their own. Every
metric here is therefore reported for the translational and rotational blocks
separately, and the combined figure uses an explicit characteristic length to
put the two on one scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from arm_lab_model.kinematics import ArmModel

#: Below this smallest singular value the pose is treated as singular.
DEFAULT_SINGULAR_TOLERANCE = 0.02


@dataclass
class Metrics:
    """Conditioning of one pose."""

    manipulability_v: float        # sqrt(det(Jv Jv^T)), m^3/rad^3
    manipulability_w: float        # sqrt(det(Jw Jw^T))
    manipulability: float          # combined, length-scaled
    sigma_min_v: float             # worst-direction speed per unit joint speed
    sigma_max_v: float
    condition_v: float             # sigma_max / sigma_min, >= 1
    sigma_min_full: float
    condition_full: float
    is_singular: bool
    kind: str                      # 'none', 'wrist', 'stretch', 'shoulder', ...

    @property
    def isotropy(self) -> float:
        """1.0 when the TCP moves equally easily in every direction."""
        return 1.0 / self.condition_v if self.condition_v > 0 else 0.0


def _safe_det_sqrt(A: np.ndarray) -> float:
    return float(math.sqrt(max(float(np.linalg.det(A @ A.T)), 0.0)))


def _condition(singular_values: np.ndarray) -> float:
    smallest = float(singular_values[-1])
    if smallest <= 1e-12:
        return math.inf
    return float(singular_values[0]) / smallest


def classify(model: ArmModel, q: Sequence[float],
             fs=None, tolerance: float = 1e-2) -> str:
    """Name the geometric degeneracy, when there is one.

    The numerical test says *that* a pose is singular; this says *why*, which is
    what tells you whether to change the trajectory or change the arm.
    """
    fs = fs or model.frames(q)
    axes = fs.joint_axis
    n = model.n
    reasons: List[str] = []

    # Wrist: two of the last three axes have become parallel, so the wrist has
    # lost a rotational degree of freedom.
    if n >= 3:
        for i in range(max(n - 3, 0), n):
            for j in range(i + 1, n):
                if abs(float(axes[i] @ axes[j])) > 1.0 - tolerance:
                    reasons.append(f'wrist ({model.cfg.joint_names[i]}'
                                   f'/{model.cfg.joint_names[j]} aligned)')
                    break

    # Stretch (elbow): the arm is straight, so it cannot extend any further.
    reach = model.reach(q, fs)
    if reach > model.geometric_max_reach * (1.0 - tolerance):
        reasons.append('stretch (arm fully extended)')

    # Shoulder: the TCP sits on the first joint's axis, so yaw does nothing.
    base_axis = axes[0]
    base_origin = fs.joint_origin[0]
    radial = fs.tcp - base_origin
    radial = radial - float(radial @ base_axis) * base_axis
    if float(np.linalg.norm(radial)) < tolerance * model.geometric_max_reach:
        reasons.append('shoulder (TCP on the base rotation axis)')

    return '; '.join(dict.fromkeys(reasons)) if reasons else 'none'


def metrics(model: ArmModel, q: Sequence[float], fs=None,
            char_length: float = 0.15,
            singular_tolerance: float = DEFAULT_SINGULAR_TOLERANCE) -> Metrics:
    fs = fs or model.frames(q)
    J = model.jacobian(q, fs)
    Jv, Jw = J[:3, :], J[3:, :]
    sv_v = np.linalg.svd(Jv, compute_uv=False)
    sv_w = np.linalg.svd(Jw, compute_uv=False)
    scaled = np.vstack([Jv, char_length * Jw])
    sv_full = np.linalg.svd(scaled, compute_uv=False)

    sigma_min_full = float(sv_full[-1])
    singular = sigma_min_full < singular_tolerance
    return Metrics(
        manipulability_v=_safe_det_sqrt(Jv),
        manipulability_w=_safe_det_sqrt(Jw),
        manipulability=_safe_det_sqrt(scaled),
        sigma_min_v=float(sv_v[-1]),
        sigma_max_v=float(sv_v[0]),
        condition_v=_condition(sv_v),
        sigma_min_full=sigma_min_full,
        condition_full=_condition(sv_full),
        is_singular=singular,
        kind=classify(model, q, fs) if singular else 'none',
    )


def worst_along(model: ArmModel, trajectory: Sequence[Sequence[float]],
                **kwargs) -> Metrics:
    """The least well-conditioned pose on a path -- the one that will bite."""
    worst: Optional[Metrics] = None
    for q in trajectory:
        m = metrics(model, q, **kwargs)
        if worst is None or m.sigma_min_full < worst.sigma_min_full:
            worst = m
    assert worst is not None, 'empty trajectory'
    return worst


def speed_capability(model: ArmModel, q: Sequence[float], fs=None) -> dict:
    """How fast the TCP can go here, best case and guaranteed worst direction.

    The worst-direction figure is the honest one for a speed specification: it
    is the speed the arm can make in *any* commanded direction.
    """
    fs = fs or model.frames(q)
    best, _ = model.max_tcp_speed(q, fs)
    Jv = model.jacobian(q, fs)[:3, :]
    # Guaranteed speed in the worst direction, given per-joint speed limits.
    sv = np.linalg.svd(Jv, compute_uv=False)
    guaranteed = float(sv[-1]) * float(np.min(model.velocity_limits))
    return {'best_direction': best, 'worst_direction': guaranteed}
