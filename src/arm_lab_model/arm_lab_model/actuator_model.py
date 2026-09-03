"""Electrical and thermal model of an actuator.

"Peak torque" on a data sheet is a stall number. Two things take it away:

voltage
    torque needs current, current needs volts, and the back-EMF eats the supply
    as speed rises. Above the corner speed the available torque falls linearly
    to zero at the no-load speed, so an arm can have the torque OR the speed at
    a given operating point and not necessarily both.

heat
    holding a load draws current whether or not anything moves, and copper loss
    goes as the square of it. An arm with no brakes holds position on motor
    current alone, which makes the thermal limit -- not the torque limit -- the
    thing that decides whether it can hold a sample at full reach.

Conventions: `phase_resistance` is the phase-to-phase value quoted on data
sheets, and copper loss is taken as I^2 R with the same current the torque
constant refers to. That is the usual field-oriented-control bookkeeping and is
consistent with how Kt and R are published together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .reference import INSULATION_CLASS, resistance_at


@dataclass
class ActuatorElectrical:
    torque_constant: float       # N.m/A, motor side (= back-EMF constant in SI)
    phase_resistance: float      # ohm at 20 C
    bus_voltage: float
    max_phase_current: float
    gear_ratio: float
    efficiency: float
    phase_inductance: float = 0.0
    peak_output_torque: float = math.inf   # the actuator's own rating

    @property
    def back_emf_constant(self) -> float:
        """V.s/rad. Numerically equal to Kt in SI units."""
        return self.torque_constant

    def current_for_output_torque(self, torque: float) -> float:
        denominator = self.torque_constant * self.gear_ratio * self.efficiency
        return abs(torque) / max(denominator, 1e-12)

    def output_torque_for_current(self, current: float) -> float:
        return current * self.torque_constant * self.gear_ratio * self.efficiency

    @property
    def no_load_speed(self) -> float:
        """Output rad/s where back-EMF alone consumes the bus."""
        return self.bus_voltage / max(self.torque_constant, 1e-12) / self.gear_ratio

    def available_torque(self, output_speed: float,
                         winding_temperature: float = 20.0) -> float:
        """Output torque available at a given output speed.

        Limited by the current the drive can push, and above the corner speed by
        what the bus has left after the back-EMF.
        """
        resistance = resistance_at(self.phase_resistance, winding_temperature)
        motor_speed = abs(output_speed) * self.gear_ratio
        headroom = self.bus_voltage - self.torque_constant * motor_speed
        if headroom <= 0.0:
            return 0.0
        current = min(self.max_phase_current, headroom / max(resistance, 1e-9))
        # The drive may be able to push more current than the actuator is rated
        # for; the mechanical rating still governs.
        return min(self.output_torque_for_current(current),
                   self.peak_output_torque)

    def corner_speed(self, winding_temperature: float = 20.0) -> float:
        """Output speed above which torque starts to fall away."""
        resistance = resistance_at(self.phase_resistance, winding_temperature)
        rated_current = min(
            self.max_phase_current,
            self.peak_output_torque / max(self.torque_constant * self.gear_ratio
                                          * self.efficiency, 1e-12))
        volts_for_current = rated_current * resistance
        headroom = self.bus_voltage - volts_for_current
        if headroom <= 0.0:
            return 0.0
        return headroom / self.torque_constant / self.gear_ratio

    def envelope(self, points: int = 60,
                 winding_temperature: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        speeds = np.linspace(0.0, self.no_load_speed, points)
        torques = np.array([self.available_torque(s, winding_temperature)
                            for s in speeds])
        return speeds, torques

    def copper_loss(self, torque: float, winding_temperature: float = 20.0) -> float:
        current = self.current_for_output_torque(torque)
        return current ** 2 * resistance_at(self.phase_resistance,
                                            winding_temperature)


@dataclass
class ActuatorThermal:
    thermal_resistance: float    # K/W, winding to ambient
    thermal_capacity: float      # J/K
    max_winding_temp: float      # C
    ambient_temp: float = 25.0
    insulation_class: str = 'H'

    @property
    def time_constant(self) -> float:
        return self.thermal_resistance * self.thermal_capacity

    @property
    def limit(self) -> float:
        return min(self.max_winding_temp,
                   INSULATION_CLASS.get(self.insulation_class, 180.0))

    def steady_state_temperature(self, power: float) -> float:
        return self.ambient_temp + power * self.thermal_resistance

    def temperature_after(self, power: float, seconds: float,
                          start: Optional[float] = None) -> float:
        start = self.ambient_temp if start is None else start
        final = self.steady_state_temperature(power)
        return final + (start - final) * math.exp(-seconds / self.time_constant)

    def time_to_limit(self, power: float,
                      start: Optional[float] = None) -> float:
        """Seconds of continuous operation before the winding limit is hit.

        Infinite when the steady-state temperature is below the limit, which is
        the definition of a continuous rating.
        """
        start = self.ambient_temp if start is None else start
        final = self.steady_state_temperature(power)
        if final <= self.limit:
            return math.inf
        ratio = (final - self.limit) / max(final - start, 1e-9)
        if ratio <= 0.0:
            return 0.0
        return -self.time_constant * math.log(ratio)


@dataclass
class ThermalVerdict:
    joint: str
    hold_torque: float
    current: float
    loss_w: float
    steady_temp: float
    limit: float
    hold_seconds: float
    continuous: bool
    duty_cycle: float            # fraction of time this load is sustainable


class ActuatorModel:
    """Electrical plus thermal behaviour of one actuator."""

    def __init__(self, electrical: ActuatorElectrical, thermal: ActuatorThermal,
                 quiescent_power: float = 0.0):
        self.electrical = electrical
        self.thermal = thermal
        self.quiescent_power = quiescent_power

    def holding(self, torque: float, name: str = '') -> ThermalVerdict:
        """Can this actuator hold this torque, and for how long?

        Iterates once on winding temperature, because the resistance rise makes
        the loss higher than a cold calculation suggests.
        """
        loss = self.electrical.copper_loss(torque, self.thermal.ambient_temp)
        for _ in range(6):
            temperature = self.thermal.steady_state_temperature(
                loss + self.quiescent_power)
            temperature = min(temperature, 400.0)
            loss = self.electrical.copper_loss(torque, temperature)
        power = loss + self.quiescent_power
        steady = self.thermal.steady_state_temperature(power)
        seconds = self.thermal.time_to_limit(power)
        headroom = self.thermal.limit - self.thermal.ambient_temp
        duty = min(headroom / max(power * self.thermal.thermal_resistance, 1e-9),
                   1.0)
        return ThermalVerdict(
            joint=name,
            hold_torque=abs(torque),
            current=self.electrical.current_for_output_torque(torque),
            loss_w=loss,
            steady_temp=steady,
            limit=self.thermal.limit,
            hold_seconds=seconds,
            continuous=math.isinf(seconds),
            duty_cycle=duty)

    def duty_cycle_temperature(self, profile: List[Tuple[float, float]],
                               cycles: int = 20) -> float:
        """Peak winding temperature after repeating a (torque, seconds) profile."""
        temperature = self.thermal.ambient_temp
        peak = temperature
        for _ in range(cycles):
            for torque, seconds in profile:
                loss = self.electrical.copper_loss(torque, temperature)
                temperature = self.thermal.temperature_after(
                    loss + self.quiescent_power, seconds, temperature)
                peak = max(peak, temperature)
        return peak


def from_config_actuator(actuator, control: Optional[Dict] = None
                         ) -> Optional[ActuatorModel]:
    """Build the model from an `Actuator`, when it carries electrical data.

    Returns None for an actuator described only by torque and ratio, so the
    report can say the data is missing rather than invent it.
    """
    if not getattr(actuator, 'torque_constant', 0.0):
        return None
    electrical = ActuatorElectrical(
        torque_constant=actuator.torque_constant,
        phase_resistance=actuator.phase_resistance,
        bus_voltage=actuator.bus_voltage,
        max_phase_current=actuator.max_phase_current,
        gear_ratio=actuator.gear_ratio,
        efficiency=actuator.efficiency,
        phase_inductance=getattr(actuator, 'phase_inductance', 0.0),
        peak_output_torque=actuator.output_peak_torque)
    thermal = ActuatorThermal(
        thermal_resistance=actuator.thermal_resistance,
        thermal_capacity=actuator.thermal_capacity,
        max_winding_temp=actuator.max_winding_temp,
        ambient_temp=actuator.ambient_temp,
        insulation_class=actuator.insulation_class)
    return ActuatorModel(electrical, thermal, actuator.quiescent_power)
