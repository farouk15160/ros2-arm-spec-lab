"""Structural checks the beam-deflection model does not cover.

Bending stress says whether a tube survives one load. It says nothing about a
tube in compression folding up, or about a tube that survives every single load
and still cracks after a hundred thousand of them.

buckling
    A thin tube fails in compression long before its material yields, and it can
    fail two ways: as a column (Euler) or by the wall dimpling locally. Local
    shell buckling is the one that catches people out, because it depends on the
    wall-thickness-to-radius ratio rather than on length.

fatigue
    A rover arm doing a sampling cycle sees the same load reversal thousands of
    times. Fatigue strength is roughly half the ultimate strength for steel, and
    aluminium has no true endurance limit at all -- it keeps degrading, so a
    life has to be quoted with a cycle count attached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

#: Knockdown on the classical shell-buckling stress. The classical formula
#: overpredicts real thin-shell strength badly because it assumes a perfect
#: cylinder; test data sits well below it, and 0.3 is a standard conservative
#: design factor for the imperfection sensitivity of thin tubes.
SHELL_KNOCKDOWN = 0.30

#: End-fixity factors for the Euler column formula.
END_FIXITY = {'pinned-pinned': 1.0, 'fixed-pinned': 0.7,
              'fixed-fixed': 0.5, 'cantilever': 2.0}


@dataclass
class BucklingResult:
    link: str
    applied: float               # N, compressive
    euler_critical: float        # N
    shell_critical: float        # N
    critical: float              # N, the governing one
    mode: str
    margin: float                # critical / applied

    @property
    def safe(self) -> bool:
        return self.applied <= 0.0 or self.margin >= 1.0


def buckling(link, applied_compression: float,
             end_fixity: str = 'fixed-pinned') -> BucklingResult:
    """Critical compressive load of a hollow tube, both modes."""
    e = link.material.youngs_modulus
    i = link.second_moment_area
    length = max(link.length, 1e-6)
    k = END_FIXITY.get(end_fixity, 0.7)

    euler = math.pi ** 2 * e * i / (k * length) ** 2

    # Classical local buckling stress for a thin cylinder, with a knockdown for
    # the imperfections a real tube has.
    thickness = link.wall_thickness
    radius = max(link.outer_radius - thickness / 2.0, 1e-9)
    sigma_cr = SHELL_KNOCKDOWN * 0.605 * e * thickness / radius
    shell = sigma_cr * link.section_area

    critical = min(euler, shell)
    mode = 'column (Euler)' if euler <= shell else 'local wall buckling'
    margin = (critical / applied_compression
              if applied_compression > 1e-9 else math.inf)
    return BucklingResult(link.name, applied_compression, euler, shell,
                          critical, mode, margin)


@dataclass
class FatigueResult:
    link: str
    alternating_stress: float    # Pa
    mean_stress: float           # Pa
    endurance_limit: float       # Pa, corrected
    safety_factor: float         # Goodman
    cycles_to_failure: float
    target_cycles: float

    @property
    def safe(self) -> bool:
        return self.cycles_to_failure >= self.target_cycles


def endurance_limit(material, diameter: float,
                    surface_factor: float = 0.75,
                    reliability_factor: float = 0.814,
                    temperature_factor: float = 1.0) -> float:
    """Corrected endurance limit via the usual Marin factors.

    Defaults: machined surface, 99 % reliability, room temperature. The
    uncorrected value is half the ultimate strength, which is the standard
    first estimate for steel and a reasonable stand-in for a quoted aluminium
    fatigue strength.
    """
    base = 0.5 * material.yield_strength * 1.2   # approximate Su from Sy
    size_factor = 1.24 * (diameter * 1000.0) ** -0.107 \
        if 2.79e-3 <= diameter <= 51e-3 else 0.9
    return base * surface_factor * size_factor * reliability_factor \
        * temperature_factor


def fatigue(link, stress_max: float, stress_min: float = 0.0,
            target_cycles: float = 1e6, **kwargs) -> FatigueResult:
    """Goodman check plus an S-N life estimate for a fully reversed component.

    A sampling arm's stress cycle runs from unloaded to loaded and back, so mean
    stress is not zero and Goodman is the right correction.
    """
    ultimate = link.material.yield_strength * 1.2
    alternating = abs(stress_max - stress_min) / 2.0
    mean = (stress_max + stress_min) / 2.0
    se = endurance_limit(link.material, 2.0 * link.outer_radius, **kwargs)

    denominator = alternating / max(se, 1.0) + mean / max(ultimate, 1.0)
    safety = 1.0 / denominator if denominator > 0 else math.inf

    # Basquin between 1e3 cycles at 0.9*Su and 1e6 at Se.
    s1000 = 0.9 * ultimate
    if alternating <= 0.0:
        cycles = math.inf
    elif alternating >= s1000:
        cycles = 1e3
    elif alternating <= se:
        cycles = math.inf
    else:
        b = -math.log10(s1000 / se) / 3.0
        a = s1000 / (1e3 ** b)
        cycles = (alternating / a) ** (1.0 / b)
    return FatigueResult(link.name, alternating, mean, se, safety, cycles,
                         target_cycles)


@dataclass
class BearingLife:
    joint: str
    dynamic_load: float          # N, equivalent
    rating: float                # N, basic dynamic load rating C
    revolutions: float
    hours_at_duty: float
    target_hours: float

    @property
    def safe(self) -> bool:
        return self.hours_at_duty >= self.target_hours


def bearing_l10(joint_name: str, equivalent_load: float, rating: float,
                revolutions_per_hour: float, target_hours: float = 20000.0,
                exponent: float = 3.0) -> BearingLife:
    """Standard L10 life: 90 % of bearings survive this many revolutions."""
    if equivalent_load <= 0.0:
        return BearingLife(joint_name, 0.0, rating, math.inf, math.inf,
                           target_hours)
    revolutions = (rating / equivalent_load) ** exponent * 1e6
    hours = revolutions / max(revolutions_per_hour, 1e-9)
    return BearingLife(joint_name, equivalent_load, rating, revolutions, hours,
                       target_hours)


@dataclass
class MassCorrection:
    """What the idealised tube model leaves out.

    A tube calculation counts the tube. It does not count end fittings, bosses,
    fastener heads, cable clips, connector brackets or the harness itself, and
    those land mostly at the joints where they cost the most torque. A blanket
    factor is crude, but a crude correction that is stated beats an implicit
    factor of 1.0.
    """

    structure_factor: float = 1.25
    harness_per_joint: float = 0.08     # kg
    fastener_fraction: float = 0.03

    def corrected_link_mass(self, tube_mass: float, lumped_mass: float) -> float:
        return (tube_mass * self.structure_factor * (1.0 + self.fastener_fraction)
                + lumped_mass + self.harness_per_joint)

    def corrected_arm_mass(self, cfg) -> float:
        total = self.corrected_link_mass(cfg.pedestal.tube_mass,
                                         cfg.pedestal.lumped_mass)
        for joint in cfg.joints:
            total += self.corrected_link_mass(joint.link.tube_mass,
                                              joint.link.lumped_mass)
        return total + cfg.end_effector.mass

    @staticmethod
    def from_config(cfg) -> 'MassCorrection':
        raw = cfg.raw.get('mass_model', {}) or {}
        base = MassCorrection()
        return MassCorrection(
            structure_factor=float(raw.get('structure_factor',
                                           base.structure_factor)),
            harness_per_joint=float(raw.get('harness_per_joint',
                                            base.harness_per_joint)),
            fastener_fraction=float(raw.get('fastener_fraction',
                                            base.fastener_fraction)))
