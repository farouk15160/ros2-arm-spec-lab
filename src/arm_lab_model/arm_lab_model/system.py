"""System-level checks: timing, sensing and the power bus.

These are the parts a rigid-body simulator structurally cannot tell you about.
Gazebo runs on an idealised clock, reads sensors with no quantisation and never
loses power, so none of the following shows up there no matter how long you run
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------- real time
@dataclass
class BusTiming:
    bitrate: float
    bits_per_frame: float
    frames_per_cycle: int
    nodes: int
    control_rate: float

    @property
    def frame_time(self) -> float:
        return self.bits_per_frame / self.bitrate

    @property
    def cycle_frames(self) -> int:
        return self.nodes * self.frames_per_cycle

    @property
    def bus_time_per_cycle(self) -> float:
        return self.cycle_frames * self.frame_time

    @property
    def control_period(self) -> float:
        return 1.0 / self.control_rate

    @property
    def utilisation(self) -> float:
        return self.bus_time_per_cycle / self.control_period

    @property
    def worst_case_latency(self) -> float:
        """Latency of the last frame in a cycle, classical CAN.

        The lowest-priority message waits for every other frame in the cycle,
        plus one already-in-progress frame it cannot pre-empt. This is the
        number that sets how stale the joint data is by the time the controller
        acts on it.
        """
        return self.bus_time_per_cycle + self.frame_time

    def jitter_position_error(self, joint_speed: float) -> float:
        """Position error contributed by timing uncertainty, radians.

        A command that lands one bus period late is a command applied at the
        wrong place, and the faster the joint is moving the further wrong it is.
        """
        return joint_speed * self.worst_case_latency

    def phase_lag_deg(self, bandwidth_hz: float) -> float:
        """Phase lost to transport delay at a given loop bandwidth.

        Beyond roughly 60 degrees a position loop is on its way to instability,
        which is what puts a ceiling on the achievable stiffness.
        """
        return 360.0 * bandwidth_hz * self.worst_case_latency


@dataclass
class TimingVerdict:
    timing: BusTiming
    tcp_speed: float
    tcp_error: float             # metres of lag at the TCP
    jitter_budget: float
    within_budget: bool
    notes: List[str] = field(default_factory=list)


def timing_analysis(cfg, model, tcp_speed: Optional[float] = None
                    ) -> TimingVerdict:
    can = cfg.raw.get('can_bus', {}) or {}
    rt = cfg.raw.get('realtime', {}) or {}
    rate = float(cfg.control.get('update_rate', 200))
    nodes = model.n + (1 if cfg.end_effector.finger_joint_names else 0)
    timing = BusTiming(
        bitrate=float(can.get('bitrate', 1e6)),
        bits_per_frame=float(can.get('bits_per_frame', 130)),
        frames_per_cycle=int(can.get('frames_per_joint_per_cycle', 2)),
        nodes=nodes,
        control_rate=rate)

    tcp_speed = tcp_speed if tcp_speed is not None else float(
        cfg.control.get('tcp_speed_limit', 0.2))
    tcp_error = tcp_speed * timing.worst_case_latency
    budget = float(rt.get('tcp_lag_budget', 0.001))

    notes = []
    if timing.utilisation > 1.0:
        notes.append('the bus cannot carry one full exchange inside a control '
                     'period: the loop rate is not achievable on this bus')
    elif timing.utilisation > 0.6:
        notes.append('bus load above 60 %: latency becomes sensitive to any '
                     'additional traffic')
    lag = timing.phase_lag_deg(float(rt.get('loop_bandwidth', 10.0)))
    if lag > 60.0:
        notes.append(f'{lag:.0f} deg of phase lost to bus latency at the '
                     'assumed loop bandwidth; the position loop will not be '
                     'able to run stiff')
    return TimingVerdict(timing, tcp_speed, tcp_error, budget,
                         tcp_error <= budget, notes)


# --------------------------------------------------------- contact sensing
@dataclass
class ContactVerdict:
    method: str
    torque_resolution: float     # N.m at the joint output
    force_resolution: float      # N at the TCP
    friction_floor: float        # N at the TCP, set by joint friction
    detectable: float            # N, the larger of the two
    target: float
    adequate: bool
    note: str = ''


def contact_detection(cfg, model, joint_index: int = 1,
                      lever_arm: Optional[float] = None) -> List[ContactVerdict]:
    """What contact force can actually be detected, and how.

    Two independent limits. Current sensing resolves a torque set by the ADC and
    the shunt; joint friction then hides anything below its own band, and on a
    geared joint friction is the bigger of the two by a wide margin.
    """
    sensing = cfg.raw.get('sensing', {}) or {}
    target = float(cfg.spec_targets.get('contact_force_limit', 20.0))
    joint = cfg.joints[min(joint_index, len(cfg.joints) - 1)]
    if lever_arm is None:
        q = model.resolve_pose('full_reach')
        lever_arm = model.reach(q)
    lever_arm = max(lever_arm, 1e-6)

    bits = int(sensing.get('current_sense_bits', 12))
    full_scale = float(sensing.get('current_sense_full_scale',
                                   joint.actuator.peak_torque
                                   / max(joint.actuator.gear_ratio, 1) * 40.0))
    current_resolution = full_scale / (2 ** bits)
    kt = getattr(joint.actuator, 'torque_constant', 0.0) or 0.1
    torque_resolution = (current_resolution * kt * joint.actuator.gear_ratio
                         * joint.actuator.efficiency)
    friction_floor = joint.actuator.friction / lever_arm

    out = [ContactVerdict(
        method=f'motor current on {joint.name}',
        torque_resolution=torque_resolution,
        force_resolution=torque_resolution / lever_arm,
        friction_floor=friction_floor,
        detectable=max(torque_resolution / lever_arm, friction_floor),
        target=target,
        adequate=max(torque_resolution / lever_arm, friction_floor) <= target / 3.0,
        note='joint friction, not the ADC, sets this floor'
        if friction_floor > torque_resolution / lever_arm else
        'current resolution dominates')]

    ft_resolution = float(sensing.get('ft_sensor_resolution', 0.1))
    if ft_resolution > 0:
        out.append(ContactVerdict(
            method='wrist force/torque sensor',
            torque_resolution=ft_resolution * lever_arm,
            force_resolution=ft_resolution,
            friction_floor=0.0,
            detectable=ft_resolution,
            target=target,
            adequate=ft_resolution <= target / 3.0,
            note='sits outboard of the gearbox, so joint friction does not '
                 'mask it'))
    return out


# ------------------------------------------------------------- power bus
@dataclass
class BusVerdict:
    voltage: float
    peak_current: float
    peak_power: float
    regen_power: float
    regen_current: float
    notes: List[str] = field(default_factory=list)


def bus_analysis(cfg, model, payload: float = 0.0,
                 descent_speed: Optional[float] = None) -> Dict[float, BusVerdict]:
    """Peak draw per bus, and what comes back when a load is lowered.

    Lowering a payload is a generator. With no brake resistor the energy has to
    go into the bus capacitance or back through the supply, and a supply that
    cannot sink it will simply rise in voltage until something trips.
    """
    descent_speed = descent_speed if descent_speed is not None else float(
        cfg.raw.get('motion', {}).get('cartesian_speed', 0.2))
    q = model.resolve_pose('full_reach')
    fs = model.frames(q)
    qd = 0.5 * model.velocity_limits
    power = model.power(q, qd, payload=payload, fs=fs)

    buses: Dict[float, BusVerdict] = {}
    for voltage, watts in power['per_bus_w'].items():
        buses[voltage] = BusVerdict(voltage=voltage,
                                    peak_current=watts / voltage,
                                    peak_power=watts,
                                    regen_power=0.0, regen_current=0.0)

    # Regeneration: lowering the whole moving mass plus the payload.
    lowered = float(np.sum(fs.link_mass)) + fs.ee_mass + payload
    mechanical = lowered * cfg.gravity * descent_speed
    main = max(buses) if buses else 48.0
    if main in buses:
        recovered = mechanical * min(
            float(np.mean([j.actuator.efficiency for j in cfg.joints])), 1.0)
        buses[main].regen_power = recovered
        buses[main].regen_current = recovered / main
        buses[main].notes.append(
            f'lowering {lowered:.1f} kg at {descent_speed:.2f} m/s returns about '
            f'{recovered:.0f} W to the {main:.0f} V bus')
        buses[main].notes.append(
            'with no brake resistor this has to be absorbed by the bus '
            'capacitance or the supply; size for it or limit descent speed')
    return buses


# ----------------------------------------------- holding without a brake
@dataclass
class HoldingVerdict:
    joint: str
    hold_torque: float
    backdrivable: bool
    backdrive_threshold: float   # N.m at the output
    dynamic_brake_torque: float  # N.m at the output, at `drop_speed`
    falls_on_power_loss: bool
    descent_speed: float = 0.0   # rad/s the joint settles at, phases shorted
    note: str = ''


def holding_analysis(cfg, model, payload: float = 0.0,
                     drop_speed: float = 0.5) -> List[HoldingVerdict]:
    """What happens at the moment the power goes away.

    With no brake, holding is done entirely by motor current, and when that
    stops the only things left are gearbox friction and whatever the windings
    do if the drive shorts them. A high-ratio strain-wave gear is usually not
    backdrivable, which is what makes a brakeless design defensible -- but that
    is a property to verify, not to assume.
    """
    q = model.resolve_pose('full_reach')
    tau = model.gravity_torque(q, payload=payload)
    out: List[HoldingVerdict] = []
    for i, joint in enumerate(cfg.joints):
        act = joint.actuator
        forward = act.efficiency
        # Standard approximation: a drive with forward efficiency at or below
        # 0.5 cannot be driven from the output at all.
        reverse = 2.0 - 1.0 / max(forward, 1e-6)
        friction_at_output = act.friction
        if reverse <= 0.0:
            backdrivable = False
            threshold = math.inf
        else:
            threshold = friction_at_output / max(reverse, 1e-6)
            backdrivable = abs(tau[i]) > threshold

        kt = getattr(act, 'torque_constant', 0.0)
        resistance = getattr(act, 'phase_resistance', 0.0)
        # Shorting the windings makes the motor a brake whose torque is
        # proportional to speed, so it does not hold: it settles at whatever
        # speed balances gravity. That is a controlled descent, not a stop.
        if kt and resistance:
            gain = (kt * kt / resistance) * act.gear_ratio ** 2 * forward
            brake = gain * drop_speed
            excess = max(abs(tau[i]) - threshold, 0.0)
            descent = excess / gain if gain > 0 else math.inf
        else:
            gain = 0.0
            brake = 0.0
            descent = math.inf

        falls = backdrivable and abs(tau[i]) > threshold
        if not backdrivable:
            note = 'self-locking at this ratio and efficiency: it stays put'
        elif not falls:
            note = 'gearbox friction alone holds it'
        elif gain > 0:
            note = (f'backdrives; shorting the phases limits it to '
                    f'{descent:.3f} rad/s, a slow descent rather than a hold')
        else:
            note = 'backdrives and there is nothing to stop it'
        out.append(HoldingVerdict(joint.name, abs(tau[i]), backdrivable,
                                  threshold, brake, falls, descent, note))
    return out


# ------------------------------------------------- second output encoder
@dataclass
class EncoderVerdict:
    joint: str
    motor_resolution: float      # rad at the output, via the gearbox
    output_resolution: float     # rad, from the second encoder
    windup: float                # rad, deflection under load
    windup_measurable: bool
    corrected_windup: float      # rad remaining after correction
    tcp_error_before: float      # m
    tcp_error_after: float       # m


def encoder_analysis(cfg, model, payload: float = 0.0) -> List[EncoderVerdict]:
    """What a second, output-side encoder buys.

    A motor-side encoder measures the motor. Everything between the motor and
    the link -- gearbox wind-up, backlash, bracket flex -- is invisible to it,
    and on this arm that hidden term is most of the position error. An encoder
    on the output measures the joint where it actually is, so the deflection
    stops being an error and becomes a reading, and the controller can take it
    out. It also gives absolute position after a power loss, which is what
    replaces the homing routine a brakeless arm would otherwise need.
    """
    q = model.resolve_pose('full_reach')
    fs = model.frames(q)
    tau = model.gravity_torque(q, payload=payload, fs=fs)
    sensing = cfg.raw.get('sensing', {}) or {}
    out: List[EncoderVerdict] = []
    for i, joint in enumerate(cfg.joints):
        act = joint.actuator
        motor_bits = int(getattr(act, 'encoder_bits', 17))
        output_bits = int(getattr(act, 'output_encoder_bits', 0)
                          or sensing.get('output_encoder_bits', 0))
        motor_res = 2.0 * math.pi / (2 ** motor_bits) / max(act.gear_ratio, 1.0)
        output_res = (2.0 * math.pi / (2 ** output_bits)) if output_bits else 0.0

        windup = tau[i] / max(act.joint_stiffness, 1e-9)
        measurable = bool(output_bits) and abs(windup) > 3.0 * output_res
        remaining = (max(abs(windup) - abs(windup) * 0.9, output_res)
                     if measurable else abs(windup))

        z = fs.joint_axis[i]
        r = fs.tcp - fs.joint_origin[i]
        lever = float(np.linalg.norm(np.cross(z, r)))
        out.append(EncoderVerdict(
            joint=joint.name,
            motor_resolution=motor_res,
            output_resolution=output_res,
            windup=windup,
            windup_measurable=measurable,
            corrected_windup=remaining,
            tcp_error_before=abs(windup) * lever,
            tcp_error_after=remaining * lever))
    return out
