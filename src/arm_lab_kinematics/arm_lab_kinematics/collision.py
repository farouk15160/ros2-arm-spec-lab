"""Self-collision and ground-collision checking with capsule proxies.

Each tube link is already a cylinder, so a capsule of the same radius around the
same axis is a tight fit rather than a crude bound. Adjacent links are skipped:
they touch at the joint by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from arm_lab_model.kinematics import ArmModel


def segment_distance(p1: np.ndarray, q1: np.ndarray,
                     p2: np.ndarray, q2: np.ndarray) -> float:
    """Shortest distance between two line segments."""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) \
                if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


@dataclass
class Capsule:
    name: str
    start: np.ndarray
    end: np.ndarray
    radius: float


@dataclass
class CollisionReport:
    in_collision: bool
    pairs: List[Tuple[str, str, float]]      # name, name, penetration depth
    ground: List[Tuple[str, float]]
    min_clearance: float


class CollisionChecker:
    """Capsule self-collision checker with an allowed-collision matrix.

    Short wrist links can have radii larger than the spacing between them, so
    their capsules overlap in every reachable pose. Those pairs are not
    collisions, they are the model being conservative, and flagging them would
    make the checker useless. The ACM is built the way MoveIt builds its SRDF
    disable list: sample the joint space and permanently disable any pair that
    is in contact essentially always.
    """

    def __init__(self, model: ArmModel, margin: float = 0.005,
                 ground_z: Optional[float] = 0.0,
                 skip_adjacent: int = 1,
                 allowed: Optional[Set[Tuple[str, str]]] = None):
        self.model = model
        self.margin = margin
        self.ground_z = ground_z
        self.skip_adjacent = skip_adjacent
        self.allowed: Set[Tuple[str, str]] = allowed or set()

    @staticmethod
    def _key(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def allow(self, a: str, b: str) -> None:
        self.allowed.add(self._key(a, b))

    def capsules(self, q: Sequence[float], fs=None) -> List[Capsule]:
        fs = fs or self.model.frames(q)
        out: List[Capsule] = []
        cfg = self.model.cfg
        for i, joint in enumerate(cfg.joints):
            link = joint.link
            out.append(Capsule(link.name, fs.joint_origin[i].copy(),
                               fs.distal[i].copy(), link.outer_radius))
        ee = cfg.end_effector
        tip = fs.distal[-1] if len(fs.distal) else fs.base_distal
        direction = fs.link_dir[-1]
        # Use the URDF link name so the generated SRDF disable_collisions list
        # refers to links that actually exist in the robot description.
        out.append(Capsule(f'{ee.name}_base_link', tip.copy(),
                           tip + direction * ee.body_length,
                           max(ee.body_width, ee.body_height) / 2.0))
        return out

    def check(self, q: Sequence[float], fs=None) -> CollisionReport:
        caps = self.capsules(q, fs)
        pairs: List[Tuple[str, str, float]] = []
        ground: List[Tuple[str, float]] = []
        min_clearance = float('inf')

        for i in range(len(caps)):
            for j in range(i + 1 + self.skip_adjacent, len(caps)):
                a, b = caps[i], caps[j]
                if self._key(a.name, b.name) in self.allowed:
                    continue
                gap = (segment_distance(a.start, a.end, b.start, b.end)
                       - a.radius - b.radius)
                min_clearance = min(min_clearance, gap)
                if gap < self.margin:
                    pairs.append((a.name, b.name, self.margin - gap))

        if self.ground_z is not None:
            for cap in caps:
                lowest = min(float(cap.start[2]), float(cap.end[2])) - cap.radius
                gap = lowest - self.ground_z
                min_clearance = min(min_clearance, gap)
                if gap < self.margin:
                    ground.append((cap.name, self.margin - gap))

        return CollisionReport(
            in_collision=bool(pairs or ground),
            pairs=pairs, ground=ground,
            min_clearance=(0.0 if min_clearance == float('inf')
                           else min_clearance))

    def is_free(self, q: Sequence[float], fs=None) -> bool:
        return not self.check(q, fs).in_collision


def build_allowed_collisions(model: ArmModel, samples: int = 1500,
                             always_fraction: float = 0.95,
                             seed: int = 0,
                             margin: float = 0.005) -> Dict[str, object]:
    """Find link pairs that are in contact in (almost) every reachable pose.

    Mirrors how MoveIt generates its SRDF `disable_collisions` list, and the
    result is used for exactly that when the MoveIt config is generated.
    """
    checker = CollisionChecker(model, margin=margin, ground_z=None)
    rng = np.random.default_rng(seed)
    caps = checker.capsules(np.zeros(model.n))
    names = [c.name for c in caps]

    counts: Dict[Tuple[str, str], int] = {}
    for i in range(len(names)):
        for j in range(i + 1 + checker.skip_adjacent, len(names)):
            counts[checker._key(names[i], names[j])] = 0

    for _ in range(samples):
        q = rng.uniform(model.lower, model.upper)
        for name_a, name_b, _ in checker.check(q).pairs:
            counts[checker._key(name_a, name_b)] += 1

    always, never, sometimes = set(), set(), set()
    for pair, hits in counts.items():
        if hits >= always_fraction * samples:
            always.add(pair)
        elif hits == 0:
            never.add(pair)
        else:
            sometimes.add(pair)
    return {'always': always, 'never': never, 'sometimes': sometimes,
            'samples': samples, 'counts': counts}
