"""Error injection: what actually stands between commanded and attained pose.

The distinction that makes the ISO 9283 numbers mean anything is the one
between *systematic* and *random* error:

systematic
    fixed for a given built arm -- link machining tolerances, joint zero offsets
    from assembly and calibration, gravity droop, and backlash when every pose
    is approached from the same direction. These move the mean, so they set
    **accuracy** and are exactly what a calibration or a vision correction can
    take out.

random
    different on every approach -- stiction and control deadband, encoder noise.
    These scatter the samples, so they set **repeatability**, and no amount of
    calibration removes them.

ISO 9283 mandates approaching every pose from the same direction precisely so
that backlash lands in the first bucket rather than the second. That is modelled
here rather than assumed away.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from arm_lab_model.config import ArmConfig
from arm_lab_model.kinematics import ArmModel

ARCMIN = math.pi / (180.0 * 60.0)


@dataclass
class ErrorSpec:
    """Error sources, all output-side unless stated otherwise."""

    backlash_arcmin: float = 3.0            # gearbox lost motion, peak to peak
    encoder_bits: int = 17                  # motor-side absolute encoder
    joint_zero_offset_arcmin: float = 2.0   # assembly + calibration, 1 sigma
    link_length_tolerance: float = 0.0005   # metres, 1 sigma
    stiction_deadband_arcmin: float = 0.5   # random per approach, 1 sigma
    include_compliance: bool = True         # gravity droop through the gearboxes
    thermal_drift_arcmin: float = 0.0       # slow systematic, 1 sigma

    @staticmethod
    def from_config(cfg: ArmConfig) -> 'ErrorSpec':
        raw = cfg.raw.get('errors', {}) or {}
        base = ErrorSpec()
        return ErrorSpec(
            backlash_arcmin=float(raw.get('backlash_arcmin', base.backlash_arcmin)),
            encoder_bits=int(raw.get('encoder_bits', base.encoder_bits)),
            joint_zero_offset_arcmin=float(
                raw.get('joint_zero_offset_arcmin', base.joint_zero_offset_arcmin)),
            link_length_tolerance=float(
                raw.get('link_length_tolerance', base.link_length_tolerance)),
            stiction_deadband_arcmin=float(
                raw.get('stiction_deadband_arcmin', base.stiction_deadband_arcmin)),
            include_compliance=bool(
                raw.get('include_compliance', base.include_compliance)),
            thermal_drift_arcmin=float(
                raw.get('thermal_drift_arcmin', base.thermal_drift_arcmin)),
        )


@dataclass
class ErrorBudget:
    """Per-source contribution to TCP error, in metres."""

    contributions: Dict[str, float] = field(default_factory=dict)

    def total(self) -> float:
        return float(sum(self.contributions.values()))


class BuiltArm:
    """One physical unit: systematic errors drawn once, then held fixed.

    A different `seed` is a different arm off the same drawing.
    """

    def __init__(self, cfg: ArmConfig, spec: Optional[ErrorSpec] = None,
                 seed: int = 0):
        self.spec = spec or ErrorSpec.from_config(cfg)
        self.nominal = ArmModel(cfg)
        rng = np.random.default_rng(seed)
        n = self.nominal.n

        # --- systematic: link machining tolerances -> a different real arm ---
        perturbed = copy.deepcopy(cfg)
        self.link_deltas = rng.normal(0.0, self.spec.link_length_tolerance, n)
        for i, joint in enumerate(perturbed.joints):
            joint.link.length = max(joint.link.length + float(self.link_deltas[i]),
                                    1e-4)
        self.actual = ArmModel(perturbed)

        # --- systematic: joint zero offsets from assembly and calibration ---
        self.zero_offsets = rng.normal(
            0.0, self.spec.joint_zero_offset_arcmin * ARCMIN, n)
        self.thermal = rng.normal(
            0.0, self.spec.thermal_drift_arcmin * ARCMIN, n)

        # --- per-joint constants -------------------------------------------
        self.backlash = np.full(n, self.spec.backlash_arcmin * ARCMIN)
        self.encoder_lsb = np.array([
            2.0 * math.pi / (2 ** self.spec.encoder_bits * j.actuator.gear_ratio)
            for j in cfg.joints])
        self.stiffness = np.array([j.actuator.joint_stiffness for j in cfg.joints])

    # ------------------------------------------------------------------ core
    def attained(self, q_command: Sequence[float],
                 approach_sign: float = 1.0,
                 payload: float = 0.0,
                 rng: Optional[np.random.Generator] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where the arm really ends up when told to go to `q_command`.

        Returns (tcp position, tcp rotation, actual joint angles).
        `approach_sign` is the direction the joints were last moving; holding it
        constant is what makes backlash systematic instead of random.
        """
        rng = rng or np.random.default_rng()
        q = np.asarray(q_command, dtype=float).copy()

        # Systematic, fixed for this unit.
        q = q + self.zero_offsets + self.thermal
        # Backlash: the output sits at one end of the lost-motion band, and
        # which end depends only on the approach direction.
        q = q + approach_sign * 0.5 * self.backlash
        # Encoder quantisation: deterministic given the commanded angle.
        q = np.round(q / self.encoder_lsb) * self.encoder_lsb
        # Random, redrawn on every approach.
        q = q + rng.normal(0.0, self.spec.stiction_deadband_arcmin * ARCMIN,
                           len(q))

        # Gravity droop through the gearboxes, at the pose actually held.
        if self.spec.include_compliance:
            tau = self.actual.gravity_torque(q, payload=payload)
            q = q - tau / np.maximum(self.stiffness, 1e-9)

        fs = self.actual.frames(q, light=True)
        return fs.tcp.copy(), fs.tcp_R.copy(), q

    def commanded(self, q_command: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """The pose the controller believes it is at: nominal model, no errors."""
        fs = self.nominal.frames(q_command, light=True)
        return fs.tcp.copy(), fs.tcp_R.copy()

    # -------------------------------------------------------------- analysis
    def budget(self, q: Sequence[float], payload: float = 0.0) -> ErrorBudget:
        """Break the TCP error at one pose down by source.

        Each source is switched on alone against the nominal arm, so the numbers
        say which one to spend money on.
        """
        nominal_tcp, _ = self.commanded(q)
        q = np.asarray(q, dtype=float)
        out = ErrorBudget()

        def displacement(model: ArmModel, q_used: np.ndarray) -> float:
            return float(np.linalg.norm(
                model.frames(q_used, light=True).tcp - nominal_tcp))

        out.contributions['link tolerance'] = displacement(self.actual, q)
        out.contributions['joint zero offset'] = displacement(
            self.nominal, q + self.zero_offsets)
        out.contributions['backlash'] = displacement(
            self.nominal, q + 0.5 * self.backlash)
        out.contributions['encoder resolution'] = displacement(
            self.nominal, np.round(q / self.encoder_lsb) * self.encoder_lsb)
        out.contributions['stiction'] = displacement(
            self.nominal,
            q + self.spec.stiction_deadband_arcmin * ARCMIN * np.ones(len(q)))
        if self.spec.include_compliance:
            tau = self.nominal.gravity_torque(q, payload=payload)
            out.contributions['gravity droop'] = displacement(
                self.nominal, q - tau / np.maximum(self.stiffness, 1e-9))
        if self.spec.thermal_drift_arcmin > 0:
            out.contributions['thermal drift'] = displacement(
                self.nominal, q + self.thermal)
        return out
