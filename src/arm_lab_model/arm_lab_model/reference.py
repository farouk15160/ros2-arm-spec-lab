"""Measured reference data from manufacturer catalogues and published work.

Every number here has a source. The point is that the defaults elsewhere in this
workspace stop being invented and start being traceable, and that a design can
be checked against what real components actually do.

Sources
-------
[HD]   Harmonic Drive LLC, "Cup Type Component Sets & Housed Units, CSF & CSG
       Series", Tables 42 and 43 (torsional stiffness for ratio 50:1 and for
       ratio 80:1 and up).
       https://www.harmonicdrive.net/_hd/content/documents/csf-csg.pdf
[CM]   CubeMars AK80-9 KV100 robotic actuator data sheet.
       https://www.cubemars.com/goods.php?id=982
[JS]   Reported experimental identification of a robot joint test bench giving
       891 N.m/rad for the coupling and reducer in series -- an order of
       magnitude below the gearbox alone, which is the point of
       `structural_series_stiffness`.
[TH]   Motion Control Tips, "Thermal time constants and managing PMAC servo
       motor overloads"; Machine Design, "Thermal Safety Margins for
       Servomotors". Winding time constants of tens of seconds, and winding
       resistance rising by up to ~50 % from 25 C to 155 C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

#: Copper resistivity temperature coefficient, per kelvin.
COPPER_ALPHA = 0.00393


@dataclass
class GearboxSize:
    """One frame size of a strain-wave gear, from the catalogue. [HD]"""

    size: int
    t1: float          # N.m, end of the K1 region
    t2: float          # N.m, end of the K2 region
    k1: float          # N.m/rad
    k2: float          # N.m/rad
    k3: float          # N.m/rad
    phi1_arcmin: float
    phi2_arcmin: float

    def stiffness_at(self, torque: float) -> float:
        """Torsional stiffness at a given output torque.

        Strain-wave gears stiffen with load: the catalogue approximates the
        torque/deflection curve with three straight segments, and quoting the
        high-load K3 for a lightly loaded joint overstates its stiffness by a
        factor of two or more.
        """
        t = abs(torque)
        if t <= self.t1:
            return self.k1
        if t <= self.t2:
            return self.k2
        return self.k3

    def windup(self, torque: float) -> float:
        """Output deflection in radians at a given torque, using all segments."""
        t = abs(torque)
        if t <= self.t1:
            return t / self.k1
        angle = self.t1 / self.k1
        if t <= self.t2:
            return angle + (t - self.t1) / self.k2
        angle += (self.t2 - self.t1) / self.k2
        return angle + (t - self.t2) / self.k3


def _table(sizes, t1, t2, k1, k2, k3, phi1, phi2) -> Dict[int, GearboxSize]:
    return {s: GearboxSize(s, a, b, c * 1e4, d * 1e4, e * 1e4, f, g)
            for s, a, b, c, d, e, f, g in
            zip(sizes, t1, t2, k1, k2, k3, phi1, phi2)}


_SIZES = [8, 11, 14, 17, 20, 25, 32, 40, 45, 50, 58, 65, 80, 90, 100]
_T1 = [0.29, 0.80, 2.0, 3.9, 7.0, 14, 29, 54, 76, 108, 168, 235, 430, 618, 843]
_T2 = [0.75, 2.0, 6.9, 12, 25, 48, 108, 196, 275, 382, 598, 843, 1570, 2260, 3040]

#: Harmonic Drive CSF/CSG, ratio 50:1. Stiffness columns are x10^4 N.m/rad. [HD]
CSF_RATIO_50 = _table(
    _SIZES, _T1, _T2,
    [0.044, 0.22, 0.34, 0.81, 1.3, 2.5, 5.4, 10, 15, 20, 31, 44, 81, 118, 162],
    [0.067, 0.30, 0.47, 1.1, 1.8, 3.4, 7.8, 14, 20, 28, 44, 61, 115, 162, 222],
    [0.084, 0.32, 0.57, 1.3, 2.3, 4.4, 9.8, 18, 26, 34, 54, 78, 145, 206, 283],
    [2.3, 1.2, 2.0, 1.7, 1.8, 1.9, 1.9, 1.8, 1.8, 1.9, 1.8, 1.8, 1.8, 1.8, 1.8],
    [4.7, 2.6, 5.6, 4.2, 5.3, 5.4, 5.4, 5.3, 5.2, 5.3, 5.2, 5.2, 5.2, 5.3, 5.2])

#: Harmonic Drive CSF/CSG, ratio 80:1 and up. [HD]
CSF_RATIO_80_PLUS = _table(
    _SIZES, _T1, _T2,
    [0.091, 0.27, 0.47, 1.0, 1.6, 3.1, 6.7, 13, 18, 25, 40, 54, 100, 145, 200],
    [0.10, 0.34, 0.61, 1.4, 2.5, 5.0, 11, 20, 29, 40, 61, 88, 162, 230, 310],
    [0.12, 0.44, 0.71, 1.6, 2.9, 5.7, 12, 23, 33, 44, 71, 98, 185, 263, 370],
    [1.1, 1.0, 1.4, 1.3, 1.5, 1.5, 1.5, 1.4, 1.4, 1.5, 1.4, 1.5, 1.5, 1.5, 1.5],
    [2.6, 2.2, 4.2, 3.3, 3.9, 3.8, 4.0, 3.8, 3.8, 3.8, 3.8, 3.9, 3.9, 4.0, 3.9])


def gearbox_table(ratio: float) -> Dict[int, GearboxSize]:
    return CSF_RATIO_50 if ratio < 80 else CSF_RATIO_80_PLUS


def select_gearbox(peak_torque: float, ratio: float = 100.0,
                   margin: float = 1.0) -> Optional[GearboxSize]:
    """Smallest catalogue frame size whose K2 breakpoint covers the load.

    T2 is used as the sizing anchor rather than a rated-torque column: it is the
    torque above which the gear is running on its stiffest segment, which is
    where a positioning joint wants to be.
    """
    table = gearbox_table(ratio)
    for size in sorted(table):
        if table[size].t2 >= peak_torque * margin:
            return table[size]
    return None


def structural_series_stiffness(gearbox_k: float,
                                bracket_k: float,
                                bearing_k: float,
                                shaft_k: float = 0.0) -> float:
    """Combine compliances in series -- the softest one dominates.

    A joint is not as stiff as its gearbox. Published identification of a joint
    test bench gave 891 N.m/rad where the reducer alone is orders of magnitude
    stiffer [JS], because the coupling, bearings and mounting structure sit in
    series with it. Quoting the catalogue K3 as the joint stiffness is the
    single most optimistic assumption available in arm design.
    """
    total = 0.0
    for k in (gearbox_k, bracket_k, bearing_k, shaft_k):
        if k and k > 0:
            total += 1.0 / k
    return 1.0 / total if total > 0 else float('inf')


@dataclass
class ActuatorReference:
    """A real actuator, for sanity-checking a configured one. [CM]"""

    name: str
    rated_torque: float          # N.m at the output
    peak_torque: float
    gear_ratio: float
    torque_constant: float       # N.m/A, motor side
    phase_resistance: float      # ohm, phase to phase
    phase_inductance: float      # H
    bus_voltage: float
    mass: float
    source: str


#: Measured data sheet values, not estimates.
ACTUATOR_CATALOGUE: List[ActuatorReference] = [
    ActuatorReference('CubeMars AK80-9 (KV100)', 9.0, 18.0, 9.0, 0.105, 0.170,
                      57e-6, 48.0, 0.485, '[CM]'),
    ActuatorReference('CubeMars AK80-9 V3.0', 9.0, 18.0, 9.0, 0.095, 0.160,
                      116e-6, 48.0, 0.485, '[CM]'),
]

#: Insulation class limits, degrees C (winding hot spot). [TH]
INSULATION_CLASS = {'B': 130.0, 'F': 155.0, 'H': 180.0}


def resistance_at(resistance_20c: float, temperature_c: float) -> float:
    """Copper resistance rises with temperature -- about +50 % by 155 C. [TH]

    This matters twice over: a hot motor makes less torque per watt, and the
    extra loss heats it further.
    """
    return resistance_20c * (1.0 + COPPER_ALPHA * (temperature_c - 20.0))
