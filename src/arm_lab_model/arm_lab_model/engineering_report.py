"""The engineering report: everything a rigid-body simulator cannot tell you.

    ros2 run arm_lab_model engineering_report
    ros2 run arm_lab_model engineering_report --payload 3.0 --stiffness-sweep
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional

import numpy as np

from . import structures, system
from .actuator_model import from_config_actuator
from .config import load_config
from .kinematics import ArmModel
from .reference import ACTUATOR_CATALOGUE, gearbox_table, select_gearbox

W = 100


def _rule(char: str = '-') -> str:
    return char * W


def actuator_section(cfg, model, payload: float) -> List[str]:
    out = ['', '=' * W, '  ACTUATORS: TORQUE, SPEED AND HEAT', '=' * W]
    q = model.resolve_pose('full_reach')
    hold = model.gravity_torque(q, payload=payload)

    out.append(f'{"joint":<10}{"actuator":<14}{"hold N.m":>10}{"amps":>8}'
               f'{"loss W":>9}{"T_wind C":>10}{"hold for":>12}{"duty":>8}'
               f'{"corner rad/s":>14}')
    out.append(_rule())
    missing = []
    for i, joint in enumerate(cfg.joints):
        act_model = from_config_actuator(joint.actuator)
        if act_model is None:
            missing.append(joint.actuator.name)
            continue
        verdict = act_model.holding(hold[i], joint.name)
        corner = act_model.electrical.corner_speed()
        span = ('continuous' if verdict.continuous
                else f'{verdict.hold_seconds:.0f} s')
        flag = '' if verdict.continuous else '  <-- thermally limited'
        out.append(f'{joint.name:<10}{joint.actuator.name:<14}'
                   f'{verdict.hold_torque:>10.1f}{verdict.current:>8.1f}'
                   f'{verdict.loss_w:>9.1f}{verdict.steady_temp:>10.0f}'
                   f'{span:>12}{verdict.duty_cycle * 100:>7.0f}%'
                   f'{corner:>14.2f}{flag}')
    out.append(_rule())
    if missing:
        out.append(f'  no electrical data for: {", ".join(sorted(set(missing)))} '
                   '-- add torque_constant and phase_resistance to model them')
    out.append('  "hold for" is how long the winding takes to reach its limit '
               'holding that torque')
    out.append('  from cold. This arm has no brakes, so holding is a continuous '
               'current draw and')
    out.append('  the thermal limit, not the torque limit, is what ends a hold.')

    out.append('')
    out.append('  TORQUE AVAILABLE VS SPEED (the data sheet peak is a stall '
               'number)')
    out.append(_rule())
    out.append(f'{"joint":<10}{"peak N.m":>10}{"at 50 %":>10}{"at 80 %":>10}'
               f'{"at 95 %":>10}{"corner rad/s":>14}{"no-load rad/s":>15}'
               f'{"limit needs":>13}')
    for i, joint in enumerate(cfg.joints):
        act_model = from_config_actuator(joint.actuator)
        if act_model is None:
            continue
        no_load = act_model.electrical.no_load_speed
        corner = act_model.electrical.corner_speed()
        values = [act_model.electrical.available_torque(f * no_load)
                  for f in (0.5, 0.8, 0.95)]
        need = joint.usable_speed
        flag = '' if need <= corner else '  <-- speed limit is past the corner'
        out.append(f'{joint.name:<10}{joint.effort_limit:>10.1f}'
                   f'{values[0]:>10.1f}{values[1]:>10.1f}{values[2]:>10.1f}'
                   f'{corner:>14.2f}{no_load:>15.2f}{need:>13.2f}{flag}')
    out.append('  Below the corner speed the drive current governs; above it '
               'the back-EMF does,')
    out.append('  and torque falls linearly to zero at no-load. A joint whose '
               'commanded speed')
    out.append('  sits past its corner cannot deliver its rated torque there.')
    return out


def structure_section(cfg, model, payload: float) -> List[str]:
    out = ['', '=' * W, '  STRUCTURE: BUCKLING, FATIGUE, BEARINGS, REAL MASS',
           '=' * W]
    st = cfg.raw.get('structure', {}) or {}
    q = model.resolve_pose('full_reach')
    fs = model.frames(q)
    defl = model.deflection(q, payload=payload, fs=fs)
    stress = defl['bending_stress']

    out.append('  Buckling (compressive load taken as the axial component of '
               'the held load)')
    out.append(_rule())
    out.append(f'{"link":<10}{"applied N":>11}{"Euler N":>12}{"local N":>12}'
               f'{"critical N":>13}{"margin":>9}   governing mode')
    weight = (float(np.sum(fs.link_mass)) + fs.ee_mass + payload) * cfg.gravity
    for joint in cfg.joints:
        result = structures.buckling(joint.link, weight * 0.5,
                                     st.get('end_fixity', 'fixed-pinned'))
        flag = '' if result.safe else '  <-- BUCKLES'
        margin = ('>1000' if result.margin > 1000 else f'{result.margin:.0f}')
        out.append(f'{result.link:<10}{result.applied:>11.0f}'
                   f'{result.euler_critical:>12.3g}{result.shell_critical:>12.3g}'
                   f'{result.critical:>13.3g}{margin:>9}   '
                   f'{result.mode}{flag}')
    out.append('  Local wall buckling is the mode that catches thin tubes; it '
               'depends on wall')
    out.append('  thickness over radius, not on length, so a short fat tube is '
               'not automatically safe.')
    out.append('  The applied load here is a crude axial share of the held '
               'weight. This arm is')
    out.append('  loaded in bending, not compression, so the margins are large '
               'and the check is')
    out.append('  a guard rather than a driver -- it would bite on a much '
               'thinner wall.')

    out.append('')
    out.append(f'  Fatigue at {float(st.get("target_cycles", 1e6)):.0g} cycles '
               '(unloaded to loaded and back)')
    out.append(_rule())
    out.append(f'{"link":<10}{"alt MPa":>10}{"mean MPa":>10}{"Se MPa":>9}'
               f'{"Goodman n":>12}{"life cycles":>15}')
    for i, joint in enumerate(cfg.joints):
        result = structures.fatigue(
            joint.link, stress[i], 0.0,
            target_cycles=float(st.get('target_cycles', 1e6)),
            surface_factor=float(st.get('surface_factor', 0.75)))
        life = 'infinite' if math.isinf(result.cycles_to_failure) \
            else f'{result.cycles_to_failure:.3g}'
        flag = '' if result.safe else '  <-- SHORT LIFE'
        out.append(f'{result.link:<10}{result.alternating_stress / 1e6:>10.1f}'
                   f'{result.mean_stress / 1e6:>10.1f}'
                   f'{result.endurance_limit / 1e6:>9.1f}'
                   f'{result.safety_factor:>12.1f}{life:>15}{flag}')

    out.append('')
    out.append('  Bearing life and real mass')
    out.append(_rule())
    load = weight * 0.5
    life = structures.bearing_l10(
        'shoulder', load, float(st.get('bearing_rating', 12000.0)),
        float(st.get('duty_revolutions_per_hour', 600.0)),
        float(st.get('target_hours', 20000.0)))
    hours = ('>1e6' if life.hours_at_duty > 1e6 else f'{life.hours_at_duty:.0f}')
    out.append(f'  shoulder bearing L10   {hours} h at the assumed duty '
               f'(target {life.target_hours:.0f} h)'
               f'{"" if life.safe else "   <-- SHORT"}')
    correction = structures.MassCorrection.from_config(cfg)
    corrected = correction.corrected_arm_mass(cfg)
    target = float(cfg.spec_targets.get('arm_mass_target', math.inf))
    out.append(f'  idealised tube mass    {cfg.arm_mass:.2f} kg')
    out.append(f'  with real hardware     {corrected:.2f} kg   '
               f'(x{correction.structure_factor:.2f} on tube mass, '
               f'+{correction.harness_per_joint * 1000:.0f} g harness per joint)'
               f'{"" if corrected <= target else "   <-- OVER TARGET"}')
    out.append('  The second number is the one to compare against a mass '
               'budget. The tube')
    out.append('  calculation counts the tube and nothing that attaches to it.')
    return out


def system_section(cfg, model, payload: float) -> List[str]:
    out = ['', '=' * W, '  SYSTEM: TIMING, SENSING, POWER, HOLDING', '=' * W]

    timing = system.timing_analysis(cfg, model)
    t = timing.timing
    out.append('  Real-time behaviour on classical CAN')
    out.append(_rule())
    out.append(f'  frame time                {t.frame_time * 1e6:.0f} us')
    out.append(f'  bus time per cycle        {t.bus_time_per_cycle * 1e6:.0f} us '
               f'of a {t.control_period * 1e6:.0f} us control period '
               f'({t.utilisation * 100:.0f} %)')
    out.append(f'  worst-case latency        {t.worst_case_latency * 1e6:.0f} us '
               '(lowest-priority frame)')
    out.append(f'  TCP lag from that         {timing.tcp_error * 1000:.2f} mm at '
               f'{timing.tcp_speed:.2f} m/s'
               f'{"" if timing.within_budget else "   <-- over budget"}')
    out.append(f'  phase lost at 10 Hz       {t.phase_lag_deg(10.0):.1f} deg')
    for note in timing.notes:
        out.append(f'  - {note}')

    out.append('')
    out.append('  Contact detection against the '
               f'{cfg.spec_targets.get("contact_force_limit", 20):.0f} N limit')
    out.append(_rule())
    for verdict in system.contact_detection(cfg, model):
        flag = 'adequate' if verdict.adequate else 'NOT ADEQUATE'
        out.append(f'  {verdict.method:<32}{verdict.detectable:>8.1f} N '
                   f'detectable   {flag}')
        out.append(f'      {verdict.note}')
    out.append('  A limit you cannot measure is not a limit. Detecting a third '
               'of the')
    out.append('  threshold is the usual rule for acting before it is reached.')

    out.append('')
    out.append('  Power bus')
    out.append(_rule())
    for voltage, verdict in sorted(system.bus_analysis(cfg, model, payload).items()):
        out.append(f'  {voltage:.0f} V bus   {verdict.peak_power:>6.0f} W draw, '
                   f'{verdict.peak_current:>5.1f} A')
        for note in verdict.notes:
            out.append(f'      {note}')

    out.append('')
    out.append('  Holding without brakes, and what a power loss does')
    out.append(_rule())
    out.append(f'{"joint":<10}{"hold N.m":>10}{"backdrive at":>14}'
               f'   behaviour on power loss')
    verdicts = system.holding_analysis(cfg, model, payload)
    for verdict in verdicts:
        threshold = ('self-locking' if math.isinf(verdict.backdrive_threshold)
                     else f'{verdict.backdrive_threshold:.1f} N.m')
        out.append(f'{verdict.joint:<10}{verdict.hold_torque:>10.1f}'
                   f'{threshold:>14}   {verdict.note}')
    falling = [v for v in verdicts if v.falls_on_power_loss]
    if falling:
        out.append('')
        out.append('  With no brake fitted, a power loss is NOT a stop. '
                   f'{len(falling)} of {len(verdicts)} joints')
        out.append('  backdrive under their own holding load. Shorting the '
                   'motor phases turns the')
        out.append('  fall into a slow descent, so the drive must do that on '
                   'power loss rather than')
        out.append('  simply going high-impedance -- and the arm should be '
                   'parked where a slow')
        out.append('  descent is safe. The second encoder is what lets it come '
                   'back knowing where')
        out.append('  it ended up, without a homing move.')

    out.append('')
    out.append('  What the second, output-side encoder buys')
    out.append(_rule())
    out.append(f'{"joint":<10}{"wind-up deg":>13}{"measurable":>12}'
               f'{"TCP err before":>16}{"after":>10}')
    before = after = 0.0
    for verdict in system.encoder_analysis(cfg, model, payload):
        before += verdict.tcp_error_before
        after += verdict.tcp_error_after
        out.append(f'{verdict.joint:<10}{math.degrees(verdict.windup):>13.4f}'
                   f'{str(verdict.windup_measurable):>12}'
                   f'{verdict.tcp_error_before * 1000:>15.2f}mm'
                   f'{verdict.tcp_error_after * 1000:>9.2f}mm')
    out.append(_rule())
    out.append(f'  total TCP droop  {before * 1000:.2f} mm  ->  '
               f'{after * 1000:.2f} mm once the output encoders correct it')
    out.append('  A motor-side encoder cannot see gearbox wind-up: it measures '
               'the motor.')
    out.append('  An output encoder measures the joint where it actually is, so '
               'the droop')
    out.append('  becomes a reading instead of an error. It also gives absolute '
               'position')
    out.append('  back after a power loss, which is what a brakeless arm needs '
               'instead of homing.')
    return out


def stiffness_sweep(cfg, model, payload: float,
                    factors=(0.25, 0.5, 1.0, 2.0, 4.0)) -> List[str]:
    """How the accuracy specification moves with joint stiffness."""
    out = ['', '=' * W, '  STIFFNESS SENSITIVITY', '=' * W,
           '  Joint stiffness is the least-known number in the model and the '
           'one the accuracy',
           '  budget rests on. This is what to demand from a supplier, and what '
           'to measure',
           '  on the bench before believing any of it.', '']
    target = float(cfg.spec_targets.get('raw_positioning_accuracy', 0.01))
    out.append(f'{"stiffness":>12}{"joint N.m/rad":>16}{"TCP droop":>12}'
               f'{"vs 10 mm spec":>16}')
    out.append(_rule())
    baseline = [j.actuator.joint_stiffness for j in cfg.joints]
    q = model.resolve_pose('full_reach')
    try:
        for factor in factors:
            for joint, base in zip(cfg.joints, baseline):
                joint.actuator.joint_stiffness = base * factor
            droop = model.deflection(q, payload=payload)['total']
            verdict = 'pass' if droop <= target else 'FAIL'
            out.append(f'{factor:>11.2f}x{np.mean([b * factor for b in baseline]):>16.0f}'
                       f'{droop * 1000:>11.2f}mm{verdict:>16}')
    finally:
        for joint, base in zip(cfg.joints, baseline):
            joint.actuator.joint_stiffness = base
    out.append(_rule())
    out.append('  Doubling the joint stiffness roughly halves the droop, and '
               'nothing else in')
    out.append('  the design moves it nearly as much. Thicker tubes do not: '
               'the tubes are')
    out.append('  already responsible for well under a millimetre of it.')
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog='engineering_report',
        description='Thermal, structural, timing, sensing and power analysis '
                    'beyond rigid-body dynamics.')
    p.add_argument('--config', default=None)
    p.add_argument('--payload', type=float, default=None)
    p.add_argument('--stiffness-sweep', action='store_true')
    p.add_argument('--section', choices=['actuator', 'structure', 'system'],
                   default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    model = ArmModel(cfg)
    payload = (args.payload if args.payload is not None
               else float(cfg.spec_targets.get('payload_at_full_reach', 2.0)))

    lines = ['=' * W,
             f'  ENGINEERING REPORT  --  {cfg.name}',
             f'  config: {cfg.source_path}',
             f'  holding {payload:.1f} kg at full reach, gravity '
             f'{cfg.gravity:.2f} m/s^2']
    if args.section in (None, 'actuator'):
        lines += actuator_section(cfg, model, payload)
    if args.section in (None, 'structure'):
        lines += structure_section(cfg, model, payload)
    if args.section in (None, 'system'):
        lines += system_section(cfg, model, payload)
    if args.stiffness_sweep:
        lines += stiffness_sweep(cfg, model, payload)
    lines.append('=' * W)
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
