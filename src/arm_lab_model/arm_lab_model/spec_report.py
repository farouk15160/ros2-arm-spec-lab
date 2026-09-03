"""Grade a candidate arm configuration against the target specification.

Run it before you cut any metal::

    ros2 run arm_lab_model spec_report
    ros2 run arm_lab_model spec_report --config my_variant.yaml --ee-mass 1.4

Every check states the requirement, what the configured arm actually achieves,
and where the number came from, so a FAIL points at the parameter to change.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ArmConfig, load_config
from .kinematics import ArmModel

PASS, WARN, FAIL, INFO = 'PASS', 'WARN', 'FAIL', 'INFO'

_COLOR = {PASS: '\033[32m', WARN: '\033[33m', FAIL: '\033[31m', INFO: '\033[36m'}
_RESET = '\033[0m'


@dataclass
class Check:
    parameter: str
    requirement: str
    achieved: str
    status: str
    note: str = ''
    margin: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            'parameter': self.parameter,
            'requirement': self.requirement,
            'achieved': self.achieved,
            'status': self.status,
            'margin': self.margin,
            'note': self.note,
        }


def _status(value: float, limit: float, higher_is_better: bool,
            warn_band: float = 0.10) -> str:
    """PASS/WARN/FAIL with a warning band near the limit."""
    if limit == 0:
        return INFO
    if higher_is_better:
        if value >= limit:
            return PASS if value >= limit * (1.0 + warn_band) else WARN
        return FAIL
    if value <= limit:
        return PASS if value <= limit * (1.0 - warn_band) else WARN
    return FAIL


class SpecReport:
    def __init__(self, cfg: ArmConfig, payload_override: Optional[float] = None):
        self.cfg = cfg
        self.model = ArmModel(cfg)
        self.t = cfg.spec_targets
        self.payload_override = payload_override
        self.checks: List[Check] = []
        self.detail: Dict[str, Any] = {}

    # ------------------------------------------------------------- helpers
    def _pose(self, name: str) -> np.ndarray:
        spec = self.cfg.test_poses.get(name, name)
        return self.model.resolve_pose(spec)

    def _tcp_accel_qdd(self, q, accel: float) -> np.ndarray:
        """Joint accelerations that give the TCP a horizontal linear accel."""
        m = self.model
        J = m.jacobian(q)[:3, :]
        a = np.array([accel, 0.0, 0.0])
        return J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), a)

    def _worst_case_torque(self, payload: float) -> Dict[str, Any]:
        """Scan the vertical work plane for the highest torque each joint sees.

        Poses come from an IK sweep over reach radius and TCP height, which is
        how the arm is actually used, rather than from random joint angles.
        """
        m = self.model
        cfg = self.cfg
        accel = float(cfg.control.get('tcp_acceleration', 0.5))
        r_max = m.geometric_max_reach
        peak = np.zeros(m.n)
        peak_pose = [None] * m.n
        origin = m.frames(np.zeros(m.n)).reach_origin

        radii = np.linspace(0.25 * r_max, r_max, 10)
        heights = np.linspace(-0.45 * r_max, 0.45 * r_max, 7)
        for r in radii:
            for h in heights:
                if math.hypot(r, h) > r_max:
                    continue
                target = origin + np.array([r, 0.0, h])
                q = m.ik_position(target)
                fs = m.frames(q)
                if np.linalg.norm(fs.tcp - target) > 0.02:
                    continue
                qdd = self._tcp_accel_qdd(q, accel)
                tau = m.inverse_dynamics(q, qdd=qdd, payload=payload, fs=fs)
                for i in range(m.n):
                    if abs(tau[i]) > peak[i]:
                        peak[i] = abs(tau[i])
                        peak_pose[i] = (float(r), float(h))
        return {'peak': peak, 'pose': peak_pose, 'accel': accel}

    # -------------------------------------------------------------- checks
    def run(self) -> List[Check]:
        cfg, m, t = self.cfg, self.model, self.t
        add = self.checks.append

        # ---- geometry --------------------------------------------------
        add(Check('Degrees of freedom', f"{t.get('dof', m.n)} revolute",
                  f'{m.n} joints',
                  PASS if m.n == t.get('dof', m.n) else FAIL,
                  'from the `joints:` list'))

        q_full = self._pose('full_reach')
        fs_full = m.frames(q_full)
        reach = m.reach(q_full, fs_full)
        r_lo = float(t.get('reach_min', 0.0))
        r_hi = float(t.get('reach_max', math.inf))
        reach_status = PASS if r_lo <= reach <= r_hi else FAIL
        add(Check('Reach from shoulder axis',
                  f'{r_lo * 1000:.0f}-{r_hi * 1000:.0f} mm',
                  f'{reach * 1000:.0f} mm',
                  reach_status,
                  f'geometric limit {m.geometric_max_reach * 1000:.0f} mm '
                  f'(sum of links distal of {cfg.reach_reference_joint} + TCP offset)'))
        self.detail['full_reach_pose'] = q_full.tolist()

        # ---- mass ------------------------------------------------------
        mass = cfg.arm_mass
        m_target = float(t.get('arm_mass_target', math.inf))
        m_max = float(t.get('arm_mass_max', math.inf))
        mass_status = PASS if mass <= m_target else (WARN if mass <= m_max else FAIL)
        add(Check('Arm mass incl. gripper',
                  f'<= {m_target:.1f} kg (hard max {m_max:.1f})',
                  f'{mass:.2f} kg', mass_status,
                  f'structure {cfg.structure_mass:.2f} kg + gripper '
                  f'{mass - cfg.structure_mass:.2f} kg',
                  margin=m_target - mass))

        ee_total = cfg.end_effector.mass
        allow = float(t.get('ee_mass_allowance', math.inf))
        add(Check('End-effector mass', f'<= {allow:.2f} kg', f'{ee_total:.2f} kg',
                  _status(ee_total, allow, higher_is_better=False),
                  'gripper body plus jaws; override at launch with '
                  'ee_mass:=<kg> to see what a heavier tool costs you.',
                  margin=allow - ee_total))

        # ---- payload ---------------------------------------------------
        reserve = float(cfg.control.get('dynamic_torque_reserve', 0.0))
        cap_full, lim_full = m.payload_capacity(q_full, fs_full)
        want_full = float(t.get('payload_at_full_reach', 0.0))
        add(Check('Payload at full reach', f'{want_full:.1f} kg continuous',
                  f'{cap_full:.2f} kg',
                  _status(cap_full, want_full, higher_is_better=True),
                  f'limited by {cfg.joint_names[lim_full] if lim_full >= 0 else "nothing"}'
                  f'; {reserve * 100:.0f} % torque held in reserve for acceleration',
                  margin=cap_full - want_full))

        q_700 = self._pose('reach_700')
        fs_700 = m.frames(q_700)
        cap_700, lim_700 = m.payload_capacity(q_700, fs_700)
        want_700 = float(t.get('payload_at_700mm', 0.0))
        add(Check('Payload at 700 mm', f'{want_700:.1f} kg',
                  f'{cap_700:.2f} kg',
                  _status(cap_700, want_700, higher_is_better=True),
                  f'actual reach at this pose {m.reach(q_700, fs_700) * 1000:.0f} mm, '
                  f'limited by '
                  f'{cfg.joint_names[lim_700] if lim_700 >= 0 else "nothing"}',
                  margin=cap_700 - want_700))

        # ---- joint torque budget --------------------------------------
        payload = (self.payload_override if self.payload_override is not None
                   else want_full)
        wc = self._worst_case_torque(payload)
        self.detail['worst_case'] = {
            'payload': payload,
            'peak_torque': wc['peak'].tolist(),
            'tcp_accel': wc['accel'],
        }
        worst_util = 0.0
        worst_joint = ''
        cont_over = []
        for i, joint in enumerate(cfg.joints):
            util = wc['peak'][i] / joint.effort_limit if joint.effort_limit else math.inf
            if util > worst_util:
                worst_util, worst_joint = util, joint.name
            if wc['peak'][i] > joint.actuator.output_continuous_torque:
                cont_over.append(joint.name)
        add(Check('Peak joint torque',
                  f'<= actuator peak @ {payload:.1f} kg TCP',
                  f'{worst_util * 100:.0f} % of peak on {worst_joint}',
                  _status(worst_util, 1.0, higher_is_better=False, warn_band=0.20),
                  f'worst case over the reach sweep at '
                  f"{wc['accel']:.2f} m/s^2 TCP acceleration",
                  margin=1.0 - worst_util))
        add(Check('Continuous torque',
                  'peak demand within cont. rating',
                  'exceeded on ' + (', '.join(cont_over) if cont_over else 'none'),
                  WARN if cont_over else PASS,
                  'exceeding it is fine in bursts; check the duty cycle and '
                  'motor thermal limits'))

        # ---- speed -----------------------------------------------------
        v_target = float(t.get('tcp_speed', 0.0))
        v_full, _ = m.max_tcp_speed(q_full, fs_full)
        # Slowest pose in the sweep decides whether the spec holds everywhere.
        v_min = math.inf
        for name in ('home', 'reach_700', 'full_reach', 'overhead'):
            if name in cfg.test_poses:
                qq = self._pose(name)
                v, _ = m.max_tcp_speed(qq)
                v_min = min(v_min, v)
        add(Check('Max TCP speed', f'{v_target:.2f} m/s',
                  f'{v_min:.2f} m/s at the worst test pose',
                  _status(v_min, v_target, higher_is_better=True),
                  f'{v_full:.2f} m/s at full reach. That is the kinematic '
                  'capability implied by the joint speed limits; the controller '
                  'caps the commanded speed at '
                  f"{cfg.control.get('tcp_speed_limit', v_target):.2f} m/s, so "
                  'this spec is a policy choice, not a hardware limit.'))

        # ---- accuracy / stiffness -------------------------------------
        defl = m.deflection(q_full, payload=want_full, fs=fs_full)
        acc_target = float(t.get('raw_positioning_accuracy', math.inf))
        add(Check('Raw positioning accuracy',
                  f'<= {acc_target * 1000:.0f} mm',
                  f'{defl["total"] * 1000:.1f} mm droop at full reach',
                  _status(defl['total'], acc_target, higher_is_better=False),
                  f'tube bending {defl["bending"] * 1000:.2f} mm + torsion '
                  f'{defl["torsion"] * 1000:.2f} mm + gearbox wind-up '
                  f'{defl["joint_compliance"] * 1000:.2f} mm, loaded with '
                  f'{want_full:.1f} kg. Deflection alone; encoder resolution, '
                  'backlash and calibration add to it.',
                  margin=acc_target - defl['total']))

        rep_target = float(t.get('vision_corrected_repeatability', math.inf))
        add(Check('Vision-corr. repeatability', f'<= {rep_target * 1000:.0f} mm',
                  'not simulated', INFO,
                  'repeatability depends on backlash, encoder noise and the '
                  'vision loop, none of which this model covers. The droop above '
                  'is repeatable and therefore correctable.'))

        # Orientation error implied by the same wind-up.
        wind = defl['joint_wind_up']
        ori_err = math.degrees(float(np.sum(np.abs(wind))))
        ori_target = float(t.get('orientation_accuracy_deg', math.inf))
        add(Check('Orientation accuracy', f'<= {ori_target:.1f} deg',
                  f'{ori_err:.2f} deg wind-up at full reach',
                  _status(ori_err, ori_target, higher_is_better=False),
                  'worst case sum of joint deflections under load',
                  margin=ori_target - ori_err))

        # ---- contact force --------------------------------------------
        f_limit = float(t.get('contact_force_limit', 0.0))
        f_cap = m.max_contact_force(q_full, [0.0, 0.0, -1.0], fs_full)
        add(Check('Normal contact force', f'limit {f_limit:.0f} N',
                  f'{f_cap:.0f} N available at full reach',
                  WARN if f_cap < f_limit else PASS,
                  'the arm can push harder than the limit, so the limit has to '
                  'be enforced in software (torque threshold or an F/T sensor); '
                  f'{f_limit:.0f} N at the TCP is '
                  f'{f_limit * reach:.1f} N.m at the shoulder, which is '
                  f'{f_limit * reach / cfg.joints[1].effort_limit * 100:.1f} % of '
                  f'{cfg.joints[1].name} peak torque -- that is the detection '
                  'resolution you need.'))

        # ---- gripper ---------------------------------------------------
        ee = cfg.end_effector
        add(Check('Gripper opening',
                  f"0-{float(t.get('gripper_opening', ee.stroke)) * 1000:.0f} mm",
                  f'0-{ee.stroke * 1000:.0f} mm',
                  PASS if abs(ee.stroke - float(t.get('gripper_opening', ee.stroke)))
                  < 1e-6 else WARN,
                  'modelled as two prismatic jaws, each travelling half the stroke'
                  if ee.simulate_fingers else 'fingers not simulated'))
        start, end = ee.jaw_span
        add(Check('TCP inside the jaws',
                  f'between {start * 1000:.0f} and {end * 1000:.0f} mm',
                  f'{ee.tcp_offset * 1000:.0f} mm from the flange',
                  PASS if ee.tcp_between_jaws else FAIL,
                  'The TCP is the point the arm positions. If it does not lie '
                  'between the jaws, every grasp closes on empty air while the '
                  'object sits outside the gripper -- and it looks like a '
                  'controller fault, not a geometry one.'
                  if not ee.tcp_between_jaws else
                  'the grasp point lies within the jaw travel'))

        add(Check('Gripper force',
                  f"{t.get('gripper_force_min', 0)}-"
                  f"{t.get('gripper_force_max', 0)} N adjustable",
                  f'{ee.grip_force_min:.0f}-{ee.grip_force_max:.0f} N',
                  PASS, 'set as the finger joint effort limit in the URDF'))

        # ---- power -----------------------------------------------------
        qd_fast = np.sign(np.ones(m.n)) * m.velocity_limits * 0.5
        pw = m.power(q_full, qd_fast, payload=want_full, fs=fs_full)
        bus_lines = ', '.join(
            f'{v:.0f} V: {a:.1f} A ({p:.0f} W)'
            for (v, p), (_, a) in zip(pw['per_bus_w'].items(), pw['per_bus_a'].items()))
        add(Check('Power draw',
                  f"main {t.get('main_voltage', 48)} V / "
                  f"secondary {t.get('secondary_voltage', 24)} V",
                  f"{pw['total_w']:.0f} W total",
                  INFO,
                  f'{bus_lines}, at half joint speed with the rated payload. '
                  'Size the harness and fusing from the peak, not this average.'))

        # ---- control rate and bus -------------------------------------
        rate = int(cfg.control.get('update_rate', 200))
        add(Check('ROS control update rate', f"{t.get('control_rate', 200)} Hz",
                  f'{rate} Hz',
                  PASS if rate >= int(t.get('control_rate', 200)) else FAIL,
                  'controller_manager update_rate; validate the real loop jitter '
                  'under load, the simulator will not show it'))

        can = cfg.can_bus
        bitrate = float(can.get('bitrate', 1e6))
        frames = float(can.get('frames_per_joint_per_cycle', 2))
        bits = float(can.get('bits_per_frame', 130))
        n_nodes = m.n + (1 if cfg.end_effector.simulate_fingers else 0)
        load_bps = n_nodes * frames * bits * rate
        util = load_bps / bitrate
        max_util = float(can.get('max_utilisation', 0.6))
        add(Check('CAN bus load',
                  f'<= {max_util * 100:.0f} % of {bitrate / 1e6:.1f} Mbit/s',
                  f'{util * 100:.1f} % ({load_bps / 1000:.0f} kbit/s)',
                  _status(util, max_util, higher_is_better=False),
                  f'{n_nodes} nodes x {frames:.0f} frames x {bits:.0f} bits x '
                  f'{rate} Hz; classical CAN, worst-case bit stuffing',
                  margin=max_util - util))
        self.detail['can'] = {'utilisation': util, 'bits_per_second': load_bps}

        return self.checks

    # --------------------------------------------------------------- output
    def joint_table(self, payload: Optional[float] = None) -> List[Dict[str, Any]]:
        m, cfg = self.model, self.cfg
        payload = (payload if payload is not None
                   else float(self.t.get('payload_at_full_reach', 0.0)))
        wc = self.detail.get('worst_case')
        peak = np.array(wc['peak_torque']) if wc else np.zeros(m.n)
        rows = []
        for i, joint in enumerate(cfg.joints):
            rows.append({
                'joint': joint.name,
                'actuator': joint.actuator.name,
                'link_mass_kg': joint.link.mass,
                'tube_mass_kg': joint.link.tube_mass,
                'peak_torque_nm': float(peak[i]),
                'limit_nm': joint.effort_limit,
                'continuous_nm': joint.actuator.output_continuous_torque,
                'utilisation': float(peak[i] / joint.effort_limit)
                if joint.effort_limit else math.inf,
                'speed_limit_rads': joint.usable_speed,
                'motor_speed_limit_rads': joint.actuator.output_max_speed,
            })
        return rows

    def render(self, color: bool = True) -> str:
        cfg, m = self.cfg, self.model
        lines: List[str] = []
        w = lines.append

        def paint(status: str) -> str:
            if not color:
                return f'{status:<4}'
            return f'{_COLOR.get(status, "")}{status:<4}{_RESET}'

        W = 104
        w('=' * W)
        w(f'  ARM SPEC REPORT  --  {cfg.name}')
        w(f'  config: {cfg.source_path}')
        w(f'  gravity: {cfg.gravity:.2f} m/s^2    dof: {m.n}    '
          f'end effector: {cfg.end_effector.mass:.2f} kg')
        w('=' * W)
        w('')
        w(_cell('PARAMETER', 30) + _cell('REQUIREMENT', 34)
          + _cell('ACHIEVED', 34) + 'STATUS')
        w('-' * W)
        for c in self.checks:
            w(_cell(c.parameter, 30) + _cell(c.requirement, 34)
              + _cell(c.achieved, 34) + paint(c.status))
            if c.note:
                for chunk in _wrap(c.note, 96):
                    w(f'    {chunk}')
        w('')
        w('-' * W)
        payload = self.detail.get('worst_case', {}).get('payload', 0.0)
        w(f'  JOINT BUDGET  (worst case over the reach sweep, {payload:.1f} kg at TCP)')
        w('-' * W)
        w(f'{"joint":<10}{"actuator":<14}{"link kg":>9}{"peak Nm":>10}'
          f'{"cont Nm":>10}{"limit Nm":>10}{"use %":>8}{"speed rad/s":>13}')
        for row in self.joint_table():
            flag = '' if row['utilisation'] <= 1.0 else '  <-- OVER'
            w(f"{row['joint']:<10}{row['actuator']:<14}"
              f"{row['link_mass_kg']:>9.3f}{row['peak_torque_nm']:>10.1f}"
              f"{row['continuous_nm']:>10.1f}{row['limit_nm']:>10.1f}"
              f"{row['utilisation'] * 100:>8.0f}{row['speed_limit_rads']:>13.2f}{flag}")
        w('')

        counts = {s: sum(1 for c in self.checks if c.status == s)
                  for s in (PASS, WARN, FAIL, INFO)}
        w('-' * W)
        w(f"  SUMMARY:  {counts[PASS]} pass   {counts[WARN]} warn   "
          f"{counts[FAIL]} fail   {counts[INFO]} info")
        if counts[FAIL]:
            w('  Failing: ' + ', '.join(c.parameter for c in self.checks
                                        if c.status == FAIL))
        if counts[WARN]:
            w('  Watch:   ' + ', '.join(c.parameter for c in self.checks
                                        if c.status == WARN))
        w('=' * W)
        return '\n'.join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            'config': self.cfg.source_path,
            'robot': self.cfg.name,
            'gravity': self.cfg.gravity,
            'dof': self.model.n,
            'arm_mass_kg': self.cfg.arm_mass,
            'checks': [c.as_dict() for c in self.checks],
            'joints': self.joint_table(),
            'detail': _jsonable(self.detail),
        }


def _cell(text: str, width: int) -> str:
    """Pad or truncate a table cell so the columns never drift."""
    text = str(text)
    if len(text) > width - 1:
        text = text[:width - 2] + '.'
    return text.ljust(width)


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ''
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f'{cur} {word}'.strip()
    if cur:
        lines.append(cur)
    return lines


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog='spec_report',
        description='Check a rover-arm configuration against its target spec.')
    p.add_argument('--config', default=None, help='path to arm_config.yaml')
    p.add_argument('--ee-mass', type=float, default=None,
                   help='override the end-effector mass, kg')
    p.add_argument('--payload', type=float, default=None,
                   help='payload at the TCP for the torque sweep, kg')
    p.add_argument('--gravity', type=float, default=None,
                   help='override gravity, m/s^2 (3.72 Mars, 1.62 Moon)')
    p.add_argument('--json', action='store_true', help='emit JSON instead of a table')
    p.add_argument('--no-color', action='store_true')
    args = p.parse_args(argv)

    cfg = load_config(args.config, ee_mass=args.ee_mass, gravity=args.gravity)
    report = SpecReport(cfg, payload_override=args.payload)
    report.run()
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.render(color=not args.no_color))
    return 1 if any(c.status == FAIL for c in report.checks) else 0


if __name__ == '__main__':
    sys.exit(main())
