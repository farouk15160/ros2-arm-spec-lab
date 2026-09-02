"""Reachable and dexterous workspace mapping.

Reach is a single number; workspace is a shape, and the useful part of it is
always smaller than the headline figure. Three volumes are distinguished here:

reachable
    the TCP can get there in *some* orientation
tool-down
    the TCP can get there with the tool pointing at the ground, which is what a
    sampling arm actually has to do
dexterous
    the TCP can get there in *every* sampled orientation

The first joint sweeps the whole arm about the base axis, so the workspace is
very nearly a solid of revolution. Everything is therefore computed in the
(radius, height) half-plane and swept through the first joint's travel, which is
cheaper and far easier to read than a 3-D cloud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from arm_lab_model.kinematics import ArmModel, rpy_to_matrix

from .collision import CollisionChecker
from .ik import IKSolver

#: Tool pointing straight down at the ground.
TOOL_DOWN = rpy_to_matrix([math.pi, 0.0, 0.0])


def sample_orientations(count: int, seed: int = 0) -> List[np.ndarray]:
    """A deterministic spread of orientations, tool-down always first."""
    out = [TOOL_DOWN]
    rng = np.random.default_rng(seed)
    while len(out) < max(count, 1):
        # Uniform random rotations via QR of a Gaussian matrix.
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q @ np.diag(np.sign(np.diag(R)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1.0
        out.append(Q)
    return out[:max(count, 1)]


@dataclass
class WorkspaceMap:
    r_edges: np.ndarray
    z_edges: np.ndarray
    reachable: np.ndarray            # bool  (nr, nz)
    tool_down: np.ndarray            # bool
    dexterity: np.ndarray            # float 0..1
    sweep_angle: float               # radians of first-joint travel
    collision_free: bool
    orientations: int
    notes: List[str] = field(default_factory=list)

    @property
    def dr(self) -> float:
        return float(self.r_edges[1] - self.r_edges[0])

    @property
    def dz(self) -> float:
        return float(self.z_edges[1] - self.z_edges[0])

    def _centres(self) -> Tuple[np.ndarray, np.ndarray]:
        r = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        z = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        return r, z

    def volume(self, mask: np.ndarray) -> float:
        """Pappus: sweep the masked half-plane area about the base axis."""
        r, _ = self._centres()
        area_weight = (r[:, None] * self.dr * self.dz) * mask
        return float(self.sweep_angle * area_weight.sum())

    @property
    def reachable_volume(self) -> float:
        return self.volume(self.reachable)

    @property
    def tool_down_volume(self) -> float:
        return self.volume(self.tool_down)

    @property
    def dexterous_volume(self) -> float:
        return self.volume(self.dexterity >= 0.999)

    def render(self, width: int = 78) -> str:
        """ASCII cross-section: radius across, height up."""
        r, z = self._centres()
        legend = ' unreachable  . reachable  o tool-down  # dexterous'
        lines = [f'  workspace cross-section  ({legend})', '']
        step = max(1, len(r) // width)
        for j in range(len(z) - 1, -1, -1):
            row = []
            for i in range(0, len(r), step):
                if self.dexterity[i, j] >= 0.999:
                    row.append('#')
                elif self.tool_down[i, j]:
                    row.append('o')
                elif self.reachable[i, j]:
                    row.append('.')
                else:
                    row.append(' ')
            lines.append(f'  z={z[j]:+.2f} |' + ''.join(row))
        axis = '        ' + '+' + '-' * len(range(0, len(r), step))
        lines.append(axis)
        lines.append(f'         r=0{" " * (len(range(0, len(r), step)) - 8)}'
                     f'r={r[-1]:.2f} m')
        return '\n'.join(lines)


def build_map(model: ArmModel,
              solver: Optional[IKSolver] = None,
              samples: int = 60000,
              nr: int = 40,
              nz: int = 32,
              orientations: int = 8,
              collision_checker: Optional[CollisionChecker] = None,
              dexterity: bool = True,
              seed: int = 0,
              progress=None) -> WorkspaceMap:
    """Map the workspace.

    Reachability comes from forward sampling, which is thousands of times
    cheaper than IK and cannot report a false positive. Only cells that pass
    that filter are then probed with IK for orientation coverage.
    """
    rng = np.random.default_rng(seed)
    solver = solver or IKSolver.survey(model)
    notes: List[str] = []

    fs0 = model.frames(np.zeros(model.n))
    base_origin = fs0.joint_origin[0]
    base_axis = fs0.joint_axis[0]

    def radial(p: np.ndarray) -> Tuple[float, float]:
        d = p - base_origin
        height = float(d @ base_axis)
        return float(np.linalg.norm(d - height * base_axis)), height

    # ---- pass one: reachability by forward sampling ----------------------
    points = np.empty((samples, 2))
    kept = 0
    for _ in range(samples):
        q = rng.uniform(model.lower, model.upper)
        fs = model.frames(q, light=True)
        if collision_checker is not None and not collision_checker.is_free(q, fs):
            continue
        points[kept] = radial(fs.tcp)
        kept += 1
    points = points[:kept]
    if kept == 0:
        raise RuntimeError('no collision-free samples; check the joint limits')

    r_max = float(points[:, 0].max()) * 1.02
    z_lo = float(points[:, 1].min()) - 0.02
    z_hi = float(points[:, 1].max()) + 0.02
    r_edges = np.linspace(0.0, r_max, nr + 1)
    z_edges = np.linspace(z_lo, z_hi, nz + 1)

    reachable = np.zeros((nr, nz), dtype=bool)
    ri = np.clip(np.digitize(points[:, 0], r_edges) - 1, 0, nr - 1)
    zi = np.clip(np.digitize(points[:, 1], z_edges) - 1, 0, nz - 1)
    reachable[ri, zi] = True

    sweep = float(min(model.upper[0] - model.lower[0], 2.0 * math.pi))
    notes.append(f'{kept} of {samples} samples kept'
                 + (' after collision filtering' if collision_checker else ''))
    notes.append(f'first joint sweeps {math.degrees(sweep):.0f} deg')

    tool_down = np.zeros((nr, nz), dtype=bool)
    dex = np.zeros((nr, nz))
    if not dexterity:
        notes.append('orientation coverage not evaluated (--no-dexterity)')
        return WorkspaceMap(r_edges, z_edges, reachable, tool_down, dex,
                            sweep, collision_checker is not None, 0, notes)

    # ---- pass two: orientation coverage, IK only where reachable ---------
    rots = sample_orientations(orientations, seed=seed)
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_c = 0.5 * (z_edges[:-1] + z_edges[1:])
    seed_q: Optional[np.ndarray] = None
    total = int(reachable.sum())
    done = 0
    for i in range(nr):
        for j in range(nz):
            if not reachable[i, j]:
                continue
            target = base_origin + base_axis * z_c[j] + np.array([r_c[i], 0.0, 0.0])
            hits = 0
            for k, R in enumerate(rots):
                res = solver.solve(target, R, seed=seed_q, restarts=0,
                                   wrist_reseeds=2,
                                   rng=np.random.default_rng(1000 + k))
                if res.success and (collision_checker is None
                                    or collision_checker.is_free(res.q)):
                    hits += 1
                    seed_q = res.q
                    if k == 0:
                        tool_down[i, j] = True
            dex[i, j] = hits / len(rots)
            done += 1
            if progress and done % 25 == 0:
                progress(done, total)

    notes.append(f'{len(rots)} orientations tested per cell '
                 '(tool-down plus a uniform SO(3) spread)')
    return WorkspaceMap(r_edges, z_edges, reachable, tool_down, dex,
                        sweep, collision_checker is not None, len(rots), notes)


def summarise(model: ArmModel, wmap: WorkspaceMap) -> str:
    lines = ['', '=' * 72, '  WORKSPACE', '=' * 72]
    for note in wmap.notes:
        lines.append(f'  - {note}')
    lines.append('')
    reach = model.geometric_max_reach
    rows = [
        ('reachable volume', wmap.reachable_volume),
        ('tool-down volume', wmap.tool_down_volume),
        ('dexterous volume', wmap.dexterous_volume),
    ]
    base = wmap.reachable_volume or 1.0
    for label, value in rows:
        lines.append(f'  {label:<22}{value:8.3f} m^3   '
                     f'{value / base * 100:5.1f} % of reachable')
    lines.append('')
    lines.append(f'  {"geometric max reach":<22}{reach * 1000:8.0f} mm')
    lines.append(f'  {"sampled max radius":<22}'
                 f'{wmap.r_edges[-1] * 1000:8.0f} mm from the base axis')
    lines.append('')
    lines.append(wmap.render())
    return '\n'.join(lines)
