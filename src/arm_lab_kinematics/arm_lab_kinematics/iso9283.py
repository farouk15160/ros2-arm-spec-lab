"""ISO 9283 pose accuracy, repeatability and path accuracy.

ISO 9283 is the standard industrial robots are actually quoted against, so
running it here replaces a hand-waved accuracy claim with a number that means
the same thing as the one on a datasheet.

Geometry (clause 6): a cube inscribed in the part of the working space with the
greatest expected use. Five poses lie on one diagonal plane of that cube -- P1
at the centre and P2..P5 at the four corners of the plane, each pulled 10 % of
the plane diagonal back toward the centre. Every pose is approached from the
same direction, 30 cycles by default.

Metrics (clause 7):
    AP  pose accuracy      distance from the commanded pose to the barycentre
                           of the attained poses -- systematic error
    RP  pose repeatability l_mean + 3*sigma_l about that barycentre -- scatter
    AT  path accuracy      worst deviation of the attained path from the
                           commanded straight line
    RT  path repeatability spread of the attained paths about their own mean
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from arm_lab_model.config import ArmConfig
from arm_lab_model.kinematics import ArmModel, matrix_to_rpy

from .errors import BuiltArm, ErrorSpec
from .ik import IKSolver, orientation_error
from .workspace import TOOL_DOWN

DEFAULT_CYCLES = 30


@dataclass
class PoseResult:
    name: str
    commanded: np.ndarray
    barycentre: np.ndarray
    AP: float                      # metres
    RP: float                      # metres
    AP_orientation: np.ndarray     # degrees, per axis
    RP_orientation: np.ndarray     # degrees, per axis
    samples: int
    reachable: bool = True


@dataclass
class Iso9283Result:
    cube_centre: np.ndarray
    cube_side: float
    cycles: int
    poses: List[PoseResult]
    AT: Optional[float] = None
    RT: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def worst_AP(self) -> float:
        return max((p.AP for p in self.poses if p.reachable), default=float('nan'))

    @property
    def worst_RP(self) -> float:
        return max((p.RP for p in self.poses if p.reachable), default=float('nan'))

    @property
    def worst_AP_orientation(self) -> float:
        return max((float(np.max(np.abs(p.AP_orientation)))
                    for p in self.poses if p.reachable), default=float('nan'))

    @property
    def worst_RP_orientation(self) -> float:
        return max((float(np.max(np.abs(p.RP_orientation)))
                    for p in self.poses if p.reachable), default=float('nan'))


def cube_poses(centre: Sequence[float], side: float) -> Dict[str, np.ndarray]:
    """The five ISO 9283 measurement points on a diagonal plane of the cube."""
    c = np.asarray(centre, dtype=float)
    h = side / 2.0
    # Four corners of one diagonal plane: (-,-,-), (+,-,-), (+,+,+), (-,+,+).
    corners = [np.array([-h, -h, -h]), np.array([h, -h, -h]),
               np.array([h, h, h]), np.array([-h, h, h])]
    diagonal = float(np.linalg.norm(corners[2] - corners[0]))
    poses = {'P1': c.copy()}
    for index, corner in enumerate(corners, start=2):
        direction = -corner / (np.linalg.norm(corner) or 1.0)
        poses[f'P{index}'] = c + corner + direction * 0.10 * diagonal
    return poses


def fit_cube(model: ArmModel, wmap=None, fraction: float = 0.55
             ) -> Tuple[np.ndarray, float]:
    """Place the test cube in the busiest part of the workspace.

    With a workspace map, the cube is centred on the tool-down region, which is
    where a sampling arm does its work. Without one, it falls back to a cube at
    two thirds of maximum reach.
    """
    fs0 = model.frames(np.zeros(model.n), light=True)
    base = fs0.joint_origin[0]
    axis = fs0.joint_axis[0]
    reach = model.geometric_max_reach

    if wmap is not None and wmap.tool_down.any():
        r_c = 0.5 * (wmap.r_edges[:-1] + wmap.r_edges[1:])
        z_c = 0.5 * (wmap.z_edges[:-1] + wmap.z_edges[1:])
        idx = np.argwhere(wmap.tool_down)
        radii = r_c[idx[:, 0]]
        heights = z_c[idx[:, 1]]
        centre = base + np.array([float(radii.mean()), 0.0, 0.0]) \
            + axis * float(heights.mean())
        span = min(float(radii.max() - radii.min()),
                   float(heights.max() - heights.min()))
        return centre, max(span * fraction, 0.05)

    centre = base + np.array([reach * 0.6, 0.0, 0.0]) + axis * (reach * 0.15)
    return centre, reach * 0.35


def run(cfg: ArmConfig,
        centre: Optional[Sequence[float]] = None,
        side: Optional[float] = None,
        cycles: int = DEFAULT_CYCLES,
        payload: float = 0.0,
        orientation: Optional[np.ndarray] = None,
        error_spec: Optional[ErrorSpec] = None,
        unit_seed: int = 0,
        wmap=None,
        path_points: int = 25,
        solver: Optional[IKSolver] = None) -> Iso9283Result:
    """Run the test cycle and reduce it to the ISO metrics."""
    model = ArmModel(cfg)
    solver = solver or IKSolver(model)
    arm = BuiltArm(cfg, error_spec, seed=unit_seed)
    R_cmd = TOOL_DOWN if orientation is None else np.asarray(orientation)

    if centre is None or side is None:
        auto_centre, auto_side = fit_cube(model, wmap)
        centre = auto_centre if centre is None else np.asarray(centre, dtype=float)
        side = auto_side if side is None else float(side)
    centre = np.asarray(centre, dtype=float)

    targets = cube_poses(centre, side)
    notes = [f'cube side {side * 1000:.0f} mm centred at '
             f'({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f}) m',
             f'{cycles} cycles, every pose approached from the same direction',
             'tool orientation held constant at tool-down'
             if orientation is None else 'tool orientation held constant']

    # Solve each measurement pose once; the controller reuses that solution.
    joint_targets: Dict[str, Optional[np.ndarray]] = {}
    seed_q = None
    for name, point in targets.items():
        res = solver.solve(point, R_cmd, seed=seed_q)
        joint_targets[name] = res.q if res.success else None
        if res.success:
            seed_q = res.q
        else:
            notes.append(f'{name} is not reachable at the commanded orientation '
                         f'(missed by {res.position_error * 1000:.1f} mm, '
                         f'{res.orientation_error_deg:.1f} deg)')

    rng = np.random.default_rng(unit_seed + 1)
    results: List[PoseResult] = []
    for name, point in targets.items():
        q_cmd = joint_targets[name]
        if q_cmd is None:
            results.append(PoseResult(name, point, point.copy(), float('nan'),
                                      float('nan'), np.full(3, np.nan),
                                      np.full(3, np.nan), 0, reachable=False))
            continue

        cmd_p, cmd_R = arm.commanded(q_cmd)
        positions = np.empty((cycles, 3))
        rotations = np.empty((cycles, 3))
        for cycle in range(cycles):
            p, R, _ = arm.attained(q_cmd, approach_sign=1.0, payload=payload,
                                   rng=rng)
            positions[cycle] = p
            rotations[cycle] = np.degrees(orientation_error(cmd_R, R))

        bary = positions.mean(axis=0)
        AP = float(np.linalg.norm(bary - cmd_p))
        radial = np.linalg.norm(positions - bary, axis=1)
        RP = float(radial.mean() + 3.0 * radial.std(ddof=1 if cycles > 1 else 0))
        results.append(PoseResult(
            name=name, commanded=cmd_p, barycentre=bary, AP=AP, RP=RP,
            AP_orientation=rotations.mean(axis=0),
            RP_orientation=3.0 * rotations.std(axis=0,
                                               ddof=1 if cycles > 1 else 0),
            samples=cycles))

    AT, RT = _path_metrics(arm, solver, targets, R_cmd, cycles, payload,
                           path_points, rng, notes)
    return Iso9283Result(centre, side, cycles, results, AT, RT, notes)


def _path_metrics(arm: BuiltArm, solver: IKSolver, targets, R_cmd,
                  cycles: int, payload: float, path_points: int,
                  rng, notes: List[str]):
    """AT and RT along the commanded straight line from P1 to P2."""
    start, end = targets['P1'], targets['P2']
    commanded = np.linspace(start, end, path_points)
    q_path: List[Optional[np.ndarray]] = []
    seed_q = None
    for point in commanded:
        res = solver.solve(point, R_cmd, seed=seed_q)
        q_path.append(res.q if res.success else None)
        if res.success:
            seed_q = res.q
    if any(q is None for q in q_path):
        notes.append('path accuracy skipped: the P1-P2 line leaves the '
                     'reachable set at the commanded orientation')
        return None, None

    runs = min(cycles, 10)
    attained = np.empty((runs, path_points, 3))
    for r in range(runs):
        for k, q in enumerate(q_path):
            attained[r, k] = arm.attained(q, 1.0, payload, rng)[0]

    # AT: worst distance from an attained point to the commanded line.
    direction = end - start
    length = float(np.linalg.norm(direction))
    unit = direction / (length or 1.0)
    deltas = attained.reshape(-1, 3) - start
    along = deltas @ unit
    perpendicular = deltas - np.outer(along, unit)
    AT = float(np.linalg.norm(perpendicular, axis=1).max())

    # RT: spread of the runs about their own mean path.
    mean_path = attained.mean(axis=0)
    RT = float(np.linalg.norm(attained - mean_path, axis=2).max())
    return AT, RT


def render(result: Iso9283Result, cfg: ArmConfig) -> str:
    t = cfg.spec_targets
    acc_target = float(t.get('raw_positioning_accuracy', 0.010))
    rep_target = float(t.get('vision_corrected_repeatability', 0.003))
    ori_target = float(t.get('orientation_accuracy_deg', 2.0))

    W = 88
    lines = ['', '=' * W, '  ISO 9283 POSE ACCURACY AND REPEATABILITY', '=' * W]
    for note in result.notes:
        lines.append(f'  - {note}')
    lines.append('')
    lines.append(f'{"pose":<6}{"AP mm":>9}{"RP mm":>9}{"AP orient deg":>16}'
                 f'{"RP orient deg":>16}{"commanded xyz m":>28}')
    lines.append('-' * W)
    for p in result.poses:
        if not p.reachable:
            lines.append(f'{p.name:<6}{"unreachable at this orientation":>40}')
            continue
        lines.append(
            f'{p.name:<6}{p.AP * 1000:>9.2f}{p.RP * 1000:>9.2f}'
            f'{np.max(np.abs(p.AP_orientation)):>16.3f}'
            f'{np.max(np.abs(p.RP_orientation)):>16.3f}'
            f'{f"({p.commanded[0]:+.3f},{p.commanded[1]:+.3f},{p.commanded[2]:+.3f})":>28}')
    lines.append('-' * W)

    def verdict(value: float, limit: float) -> str:
        if math.isnan(value):
            return 'n/a '
        return 'PASS' if value <= limit else 'FAIL'

    lines.append('')
    lines.append(f'{"metric":<34}{"measured":>14}{"spec":>14}   result')
    lines.append('-' * W)
    rows = [
        ('AP  pose accuracy (worst)', result.worst_AP * 1000, acc_target * 1000,
         'mm', verdict(result.worst_AP, acc_target)),
        ('RP  pose repeatability (worst)', result.worst_RP * 1000,
         rep_target * 1000, 'mm', verdict(result.worst_RP, rep_target)),
        ('AP  orientation accuracy', result.worst_AP_orientation, ori_target,
         'deg', verdict(result.worst_AP_orientation, ori_target)),
        ('RP  orientation repeatability', result.worst_RP_orientation, ori_target,
         'deg', verdict(result.worst_RP_orientation, ori_target)),
    ]
    if result.AT is not None:
        rows.append(('AT  path accuracy', result.AT * 1000, acc_target * 1000,
                     'mm', verdict(result.AT, acc_target)))
        rows.append(('RT  path repeatability', result.RT * 1000,
                     rep_target * 1000, 'mm', verdict(result.RT, rep_target)))
    for label, value, limit, unit, mark in rows:
        lines.append(f'{label:<34}{value:>10.2f} {unit:<3}{limit:>10.2f} {unit:<3}'
                     f'   {mark}')
    lines.append('=' * W)
    lines.append('  AP is systematic and therefore correctable by calibration or a')
    lines.append('  vision loop; RP is random scatter and is not.')
    return '\n'.join(lines)
