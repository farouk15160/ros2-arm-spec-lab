"""Live capability estimate for the arm, driven by /joint_states.

The simulator reports positions and velocities faithfully; the joint torques it
reports depend on the command interface and are often just the commanded value.
So the torque shown here is computed from the measured motion with the same
inverse-dynamics model the spec report uses, and the simulator's own effort
figure is carried alongside for comparison.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from arm_lab_model.config import ArmConfig
from arm_lab_model.kinematics import ArmModel


class LiveState:
    """Turns a stream of joint states into the numbers the dashboard shows."""

    #: How often the expensive checks (deflection, capacity) are recomputed.
    SLOW_PERIOD = 0.2

    def __init__(self, cfg: ArmConfig, payload_mass: float = 0.0,
                 velocity_filter: float = 0.25):
        self.cfg = cfg
        self.model = ArmModel(cfg)
        self.n = self.model.n
        self.payload_mass = float(payload_mass)
        self.alpha = velocity_filter

        self.q = np.zeros(self.n)
        self.qd = np.zeros(self.n)
        self.qdd = np.zeros(self.n)
        self.tau_sim = np.zeros(self.n)
        self.gripper_position = 0.0

        self._qd_prev = np.zeros(self.n)
        self._stamp: Optional[float] = None
        self._have_data = False
        self._slow_at = 0.0
        self._slow: Dict[str, object] = {}
        self.peak_tcp_speed = 0.0
        self.peak_utilisation = 0.0

        self._index: Dict[str, int] = {n: i for i, n in enumerate(cfg.joint_names)}
        self._finger_names = set(cfg.end_effector.finger_joint_names)

    # ---------------------------------------------------------------- input
    def ingest(self, names: List[str], position, velocity, effort,
               stamp: Optional[float] = None) -> None:
        now = stamp if stamp is not None else time.time()
        for k, name in enumerate(names):
            idx = self._index.get(name)
            if idx is None:
                if name in self._finger_names and k < len(position):
                    self.gripper_position = float(position[k])
                continue
            if k < len(position):
                self.q[idx] = float(position[k])
            if k < len(velocity):
                self.qd[idx] = float(velocity[k])
            if k < len(effort):
                self.tau_sim[idx] = float(effort[k])

        if self._stamp is not None:
            dt = now - self._stamp
            if 1e-4 < dt < 0.5:
                raw = (self.qd - self._qd_prev) / dt
                self.qdd = (1.0 - self.alpha) * self.qdd + self.alpha * raw
        self._qd_prev = self.qd.copy()
        self._stamp = now
        self._have_data = True

    @property
    def ready(self) -> bool:
        return self._have_data

    def reset_peaks(self) -> None:
        self.peak_tcp_speed = 0.0
        self.peak_utilisation = 0.0

    # -------------------------------------------------------------- metrics
    def metrics(self) -> Dict[str, object]:
        """Everything the dashboard and the capability topics need."""
        m = self.model
        fs = m.frames(self.q)

        tau = m.inverse_dynamics(self.q, self.qd, self.qdd,
                                 payload=self.payload_mass, fs=fs)
        util = np.abs(tau) / m.torque_limits
        v_tcp = m.tcp_velocity(self.q, self.qd, fs)
        speed = float(np.linalg.norm(v_tcp))
        reach = m.reach(self.q, fs)

        self.peak_tcp_speed = max(self.peak_tcp_speed, speed)
        self.peak_utilisation = max(self.peak_utilisation, float(np.max(util)))

        now = time.monotonic()
        if now - self._slow_at > self.SLOW_PERIOD or not self._slow:
            self._slow_at = now
            capacity, limiting = m.payload_capacity(self.q, fs)
            defl = m.deflection(self.q, payload=self.payload_mass, fs=fs)
            reachable, _ = m.max_tcp_speed(self.q, fs)
            self._slow = {
                'payload_capacity': capacity,
                'limiting_joint': (self.cfg.joint_names[limiting]
                                   if limiting >= 0 else '-'),
                'droop': defl['total'],
                'droop_parts': (defl['bending'], defl['torsion'],
                                defl['joint_compliance']),
                'stress_utilisation': defl['stress_utilisation'],
                'max_tcp_speed': reachable,
                'contact_force_down': m.max_contact_force(self.q, [0, 0, -1], fs),
                'contact_force_fwd': m.max_contact_force(self.q, [1, 0, 0], fs),
            }

        power = m.power(self.q, self.qd, payload=self.payload_mass, fs=fs)

        out: Dict[str, object] = {
            'q': self.q.copy(),
            'qd': self.qd.copy(),
            'qdd': self.qdd.copy(),
            'tau_model': tau,
            'tau_sim': self.tau_sim.copy(),
            'utilisation': util,
            'speed_utilisation': np.abs(self.qd) / m.velocity_limits,
            'tcp': fs.tcp.copy(),
            'tcp_velocity': v_tcp,
            'tcp_speed': speed,
            'reach': reach,
            'peak_tcp_speed': self.peak_tcp_speed,
            'peak_utilisation': self.peak_utilisation,
            'power_w': power['total_w'],
            'power_per_bus': power['per_bus_a'],
            'gripper_opening': 2.0 * self.gripper_position,
            'payload_mass': self.payload_mass,
        }
        out.update(self._slow)
        return out

    # ---------------------------------------------------------- spec checks
    def spec_status(self, mtr: Dict[str, object]) -> List[Dict[str, object]]:
        """Live pass/fail against the target spec, for the scorecard panel."""
        t = self.cfg.spec_targets
        rows: List[Dict[str, object]] = []

        def row(label, value, limit, higher_is_better, fmt='{:.2f}', unit=''):
            ok = value >= limit if higher_is_better else value <= limit
            rows.append({
                'label': label,
                'text': fmt.format(value) + unit,
                'limit': ('>= ' if higher_is_better else '<= ')
                         + fmt.format(limit) + unit,
                'ok': bool(ok),
            })

        row('TCP speed', float(mtr['tcp_speed']),
            float(t.get('tcp_speed', 0.2)), False, '{:.3f}', ' m/s')
        row('Joint torque', float(np.max(mtr['utilisation'])) * 100.0,
            100.0, False, '{:.0f}', ' %')
        row('Payload at TCP', float(mtr['payload_capacity']),
            float(t.get('payload_at_full_reach', 2.0)), True, '{:.2f}', ' kg')
        row('TCP droop', float(mtr['droop']) * 1000.0,
            float(t.get('raw_positioning_accuracy', 0.01)) * 1000.0,
            False, '{:.1f}', ' mm')
        reach = float(mtr['reach'])
        rows.append({
            'label': 'Reach',
            'text': f'{reach * 1000:.0f} mm',
            'limit': f"<= {float(t.get('reach_max', 1.05)) * 1000:.0f} mm",
            'ok': reach <= float(t.get('reach_max', 1.05)) + 1e-3,
        })
        return rows
