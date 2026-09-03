"""Where in the workspace the arm is close to its limits.

A single worst-case torque number says the arm is adequate; it does not say
*where* it stops being adequate. This maps torque utilisation, electrical power
and proximity to singularity over the working plane and sorts it into zones, so
the parts of the workspace to keep a loaded arm out of are visible rather than
implied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from arm_lab_model.kinematics import ArmModel

from .collision import CollisionChecker
from .ik import IKSolver
from .singularity import metrics
from .workspace import TOOL_DOWN

#: Zone thresholds on torque utilisation.
SAFE, CAUTION, DANGER, FORBIDDEN = 0.60, 0.85, 1.00, math.inf

ZONE_CHARS = {'unreachable': ' ', 'no tool-down': '-', 'self-collision': 'C',
              'safe': '.', 'caution': '+', 'danger': '#', 'over torque': 'X'}


@dataclass
class SafetyMap:
    r_edges: np.ndarray
    z_edges: np.ndarray
    envelope: np.ndarray           # bool, TCP can get there somehow
    reachable: np.ndarray          # bool, and with the tool pointing down
    utilisation: np.ndarray        # float, worst joint torque / limit
    power: np.ndarray              # float, W
    sigma_min: np.ndarray          # float
    collision: np.ndarray          # bool
    limiting_joint: np.ndarray     # int index, -1 if none
    payload: float
    joint_names: List[str]
    notes: List[str] = field(default_factory=list)

    def zone(self, i: int, j: int) -> str:
        """Why a cell is unusable matters: torque and geometry need different
        fixes, so they are not collapsed into one 'forbidden' bucket."""
        if not self.envelope[i, j]:
            return 'unreachable'
        if not self.reachable[i, j]:
            return 'no tool-down'
        if self.collision[i, j]:
            return 'self-collision'
        if self.utilisation[i, j] > DANGER:
            return 'over torque'
        if self.utilisation[i, j] > CAUTION:
            return 'danger'
        if self.utilisation[i, j] > SAFE:
            return 'caution'
        return 'safe'

    def zone_counts(self) -> Dict[str, int]:
        counts = {k: 0 for k in ZONE_CHARS}
        for i in range(self.utilisation.shape[0]):
            for j in range(self.utilisation.shape[1]):
                counts[self.zone(i, j)] += 1
        return counts

    def render(self) -> str:
        r = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        z = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        lines = [f'  zones with {self.payload:.1f} kg at the TCP, tool pointing '
                 'down',
                 f'   "." torque under {SAFE * 100:.0f} %      '
                 f'"+" {SAFE * 100:.0f}-{CAUTION * 100:.0f} %      '
                 f'"#" {CAUTION * 100:.0f}-100 %      "X" over torque limit',
                 '   "C" no collision-free pose   '
                 '"-" tool cannot point down here', '']
        for j in range(len(z) - 1, -1, -1):
            row = ''.join(ZONE_CHARS[self.zone(i, j)] for i in range(len(r)))
            lines.append(f'  z={z[j]:+.2f} |{row}')
        lines.append('        +' + '-' * len(r))
        lines.append(f'         r=0{" " * max(len(r) - 9, 1)}r={r[-1]:.2f} m')
        return '\n'.join(lines)


def build(model: ArmModel, payload: float = 2.0,
          nr: int = 30, nz: int = 22,
          samples: int = 20000,
          solver: Optional[IKSolver] = None,
          collision_checker: Optional[CollisionChecker] = None,
          collision_retries: int = 5,
          seed: int = 0, progress=None) -> SafetyMap:
    """Map torque, power and conditioning over the working plane.

    Reachability is established by forward sampling first; IK is only spent on
    cells that can actually be reached with the tool pointing down, which is the
    orientation a loaded arm is in when it matters.
    """
    cfg = model.cfg
    solver = solver or IKSolver.survey(model)
    rng = np.random.default_rng(seed)
    notes: List[str] = []

    fs0 = model.frames(np.zeros(model.n), light=True)
    base = fs0.joint_origin[0]
    axis = fs0.joint_axis[0]

    points = []
    for _ in range(samples):
        q = rng.uniform(model.lower, model.upper)
        fs = model.frames(q, light=True)
        if collision_checker is not None and not collision_checker.is_free(q, fs):
            continue
        d = fs.tcp - base
        h = float(d @ axis)
        points.append((float(np.linalg.norm(d - h * axis)), h))
    if not points:
        raise RuntimeError('no collision-free samples')
    points = np.array(points)

    r_edges = np.linspace(0.0, float(points[:, 0].max()) * 1.02, nr + 1)
    z_edges = np.linspace(float(points[:, 1].min()) - 0.02,
                          float(points[:, 1].max()) + 0.02, nz + 1)
    reachable = np.zeros((nr, nz), dtype=bool)
    ri = np.clip(np.digitize(points[:, 0], r_edges) - 1, 0, nr - 1)
    zi = np.clip(np.digitize(points[:, 1], z_edges) - 1, 0, nz - 1)
    reachable[ri, zi] = True

    utilisation = np.zeros((nr, nz))
    power = np.zeros((nr, nz))
    sigma = np.zeros((nr, nz))
    collision = np.zeros((nr, nz), dtype=bool)
    limiting = np.full((nr, nz), -1, dtype=int)
    tool_reachable = np.zeros((nr, nz), dtype=bool)

    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_c = 0.5 * (z_edges[:-1] + z_edges[1:])
    seed_q = None
    total = int(reachable.sum())
    done = 0
    # Half the joint speed limits: a representative moving state for power.
    qd = 0.5 * model.velocity_limits

    for i in range(nr):
        for j in range(nz):
            if not reachable[i, j]:
                continue
            target = base + axis * z_c[j] + np.array([r_c[i], 0.0, 0.0])
            # A tool-down target usually has several solutions and the solver
            # has no reason to prefer an uncollided one, so ask a few times and
            # keep a free solution if one exists at all.
            chosen = None
            fallback = None
            for attempt in range(collision_retries):
                res = solver.solve(target, TOOL_DOWN,
                                   seed=seed_q if attempt == 0 else None,
                                   restarts=0, wrist_reseeds=3,
                                   rng=np.random.default_rng(1000 + attempt))
                if not res.success:
                    continue
                fallback = fallback or res
                if (collision_checker is None
                        or collision_checker.is_free(res.q)):
                    chosen = res
                    break
            done += 1
            if progress and done % 25 == 0:
                progress(done, total)
            res = chosen or fallback
            if res is None:
                continue
            seed_q = res.q
            tool_reachable[i, j] = True
            fs = model.frames(res.q)
            tau = model.gravity_torque(res.q, payload=payload, fs=fs)
            use = np.abs(tau) / model.torque_limits
            utilisation[i, j] = float(np.max(use))
            limiting[i, j] = int(np.argmax(use))
            power[i, j] = model.power(res.q, qd, payload=payload,
                                      fs=fs)['total_w']
            sigma[i, j] = metrics(model, res.q, fs).sigma_min_full
            if collision_checker is not None:
                collision[i, j] = not collision_checker.is_free(res.q, fs)

    notes.append(f'{int(reachable.sum())} cells reachable, '
                 f'{int(tool_reachable.sum())} of them with the tool pointing down')
    notes.append(f'torque is the static holding torque with {payload:.1f} kg '
                 'at the TCP')
    notes.append('power is at half joint speed, which is a working estimate, '
                 'not a peak')
    blocked = int(np.sum(collision & tool_reachable))
    if blocked:
        notes.append(
            f'{blocked} of {int(tool_reachable.sum())} tool-down poses have no '
            'collision-free solution at all')
    return SafetyMap(r_edges, z_edges, reachable, tool_reachable, utilisation,
                     power, sigma, collision, limiting, payload,
                     list(cfg.joint_names), notes)


def summarise(model: ArmModel, smap: SafetyMap) -> str:
    cfg = model.cfg
    W = 92
    lines = ['', '=' * W, '  TORQUE AND POWER ZONES', '=' * W]
    for note in smap.notes:
        lines.append(f'  - {note}')
    lines.append('')

    counts = smap.zone_counts()
    working = sum(v for k, v in counts.items() if k != 'unreachable')
    lines.append(f'{"zone":<14}{"cells":>8}{"share of the working plane":>30}')
    lines.append('-' * W)
    for zone in ('safe', 'caution', 'danger', 'over torque',
                 'self-collision', 'no tool-down'):
        share = counts[zone] / working * 100 if working else 0.0
        lines.append(f'{zone:<14}{counts[zone]:>8}{share:>29.1f}%')
    lines.append('-' * W)

    mask = smap.reachable
    if mask.any():
        util = smap.utilisation[mask]
        pw = smap.power[mask]
        lines.append(f'  torque utilisation   median {np.median(util) * 100:.0f} %'
                     f'   worst {util.max() * 100:.0f} %')
        lines.append(f'  power draw           median {np.median(pw):.0f} W'
                     f'   worst {pw.max():.0f} W')
        buses: Dict[float, float] = {}
        for joint in cfg.joints:
            buses.setdefault(joint.actuator.bus_voltage, 0.0)
        lines.append(f'  bus voltages present  '
                     + ', '.join(f'{v:.0f} V' for v in sorted(buses)))
        counts_by_joint: Dict[str, int] = {}
        for idx in smap.limiting_joint[mask]:
            if idx >= 0:
                name = smap.joint_names[int(idx)]
                counts_by_joint[name] = counts_by_joint.get(name, 0) + 1
        if counts_by_joint:
            ordered = sorted(counts_by_joint.items(), key=lambda kv: -kv[1])
            lines.append('  limiting joint        ' + ', '.join(
                f'{name} ({n * 100 // max(sum(counts_by_joint.values()), 1)} %)'
                for name, n in ordered[:3]))
        near = int(np.sum((smap.sigma_min[mask] > 0)
                          & (smap.sigma_min[mask] < 0.02)))
        lines.append(f'  near-singular cells   {near} of {int(mask.sum())}')
    lines.append('')
    lines.append(smap.render())
    lines.append('')
    lines.append('  X marks poses where the torque exceeds an actuator limit.')
    lines.append('  C marks poses the arm can reach but not without hitting')
    lines.append('  itself -- a geometry problem, not a torque one, and the fix')
    lines.append('  is a longer wrist or slimmer links rather than bigger motors.')
    lines.append('  Both sit inside the reachable envelope: reachable does not')
    lines.append('  mean usable.')
    return '\n'.join(lines)
