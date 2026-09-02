"""Capability dashboard: joint speed, joint torque and payload at the TCP.

    ros2 run arm_lab_gui dashboard
    ros2 run arm_lab_gui dashboard --ros-args -p payload_mass:=2.0

Reads /joint_states, runs the same model the spec report uses, and shows what
the configured arm is doing versus what the specification asks for. The buttons
drive the trajectory controller so the numbers move under real motion.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QColor, QFont
from python_qt_binding.QtWidgets import (QApplication, QCheckBox, QComboBox,
                                         QDialog, QDoubleSpinBox, QFrame,
                                         QGridLayout, QGroupBox, QHBoxLayout,
                                         QLabel, QMainWindow, QPlainTextEdit,
                                         QPushButton, QSlider, QVBoxLayout,
                                         QWidget)
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from arm_lab_model.config import load_config
from arm_lab_model.spec_report import SpecReport

from .state import LiveState
from .widgets import (ACCENT, BAD, BG, DIM, GOOD, GRID, PANEL, TEXT, WARN,
                      BigNumber, MeterBar, ScoreCard, TimePlot)

STYLE = f"""
QMainWindow, QWidget {{ background: rgb({BG.red()},{BG.green()},{BG.blue()});
                        color: rgb({TEXT.red()},{TEXT.green()},{TEXT.blue()}); }}
QGroupBox {{ border: 1px solid rgb({GRID.red()},{GRID.green()},{GRID.blue()});
             border-radius: 3px; margin-top: 10px; padding-top: 8px;
             font-weight: bold; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px;
                    color: rgb({DIM.red()},{DIM.green()},{DIM.blue()}); }}
QPushButton {{ background: rgb(52,58,68); border: 1px solid rgb(72,80,92);
               border-radius: 3px; padding: 5px 10px; }}
QPushButton:hover {{ background: rgb(64,72,86); }}
QPushButton:pressed {{ background: rgb(40,45,54); }}
QPushButton#stop {{ background: rgb(120,44,40); border-color: rgb(170,70,64); }}
QDoubleSpinBox, QComboBox {{ background: rgb(44,49,58);
                             border: 1px solid rgb(72,80,92);
                             border-radius: 3px; padding: 3px; }}
QLabel#mono {{ font-family: monospace; }}
"""


class DashboardNode(Node):
    """ROS side of the dashboard: one subscription, two publishers."""

    def __init__(self):
        super().__init__('arm_dashboard')
        self.declare_parameter('config_file', '')
        self.declare_parameter('ee_mass', -1.0)
        self.declare_parameter('gravity', -1.0)
        self.declare_parameter('payload_mass', 0.0)
        self.declare_parameter('arm_controller', 'arm_controller')
        self.declare_parameter('gripper_controller', 'gripper_controller')

        def opt(name):
            value = float(self.get_parameter(name).value)
            return None if value < 0.0 else value

        config_file = self.get_parameter('config_file').value or None
        self.cfg = load_config(config_file, ee_mass=opt('ee_mass'),
                               gravity=opt('gravity'))
        self.state = LiveState(
            self.cfg, payload_mass=float(self.get_parameter('payload_mass').value))

        arm = self.get_parameter('arm_controller').value
        grip = self.get_parameter('gripper_controller').value
        self.traj_pub = self.create_publisher(
            JointTrajectory, f'/{arm}/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(
            Float64MultiArray, f'/{grip}/commands', 10)
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 20)
        self.last_msg_time: Optional[float] = None

    def _on_joint_state(self, msg: JointState) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = time.time()
        self.state.ingest(list(msg.name), list(msg.position), list(msg.velocity),
                          list(msg.effort), stamp)
        self.last_msg_time = time.time()

    # ------------------------------------------------------------- commands
    def send_pose(self, q: np.ndarray, duration: float) -> None:
        msg = JointTrajectory()
        msg.joint_names = list(self.cfg.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q]
        point.velocities = [0.0] * len(q)
        seconds = max(duration, 0.2)
        point.time_from_start = DurationMsg(
            sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))
        msg.points = [point]
        self.traj_pub.publish(msg)

    def hold(self) -> None:
        """Stop where we are: command the current position over a short ramp."""
        self.send_pose(self.state.q.copy(), 0.3)

    def send_gripper(self, opening: float) -> None:
        if not self.cfg.end_effector.finger_joint_names:
            return
        half = max(0.0, min(opening, self.cfg.end_effector.stroke)) / 2.0
        msg = Float64MultiArray()
        msg.data = [half] * len(self.cfg.end_effector.finger_joint_names)
        self.grip_pub.publish(msg)


class Dashboard(QMainWindow):
    def __init__(self, node: DashboardNode):
        super().__init__()
        self.node = node
        self.cfg = node.cfg
        self.state = node.state
        self.model = node.state.model
        self.t0 = time.monotonic()
        self._sweep = False
        self._sweep_flip = 0.0
        self._sweep_index = 0

        self.setWindowTitle(f'Arm capability dashboard - {self.cfg.name}')
        self.setStyleSheet(STYLE)
        self.resize(1360, 900)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(8)
        body.addWidget(self._build_joint_panel(), 5)
        body.addWidget(self._build_capability_panel(), 4)
        outer.addLayout(body, 1)
        outer.addWidget(self._build_controls())

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(20)          # 50 Hz: spin ROS and repaint

    # -------------------------------------------------------------- layout
    def _build_header(self) -> QWidget:
        box = QFrame()
        box.setStyleSheet(f'background: rgb({PANEL.red()},{PANEL.green()},'
                          f'{PANEL.blue()}); border-radius: 3px;')
        row = QHBoxLayout(box)
        row.setContentsMargins(12, 8, 12, 8)

        title = QLabel(f'<b style="font-size:15px">{self.cfg.name}</b>')
        row.addWidget(title)

        cfg = self.cfg
        summary = (f'{cfg.dof} dof   |   arm {cfg.arm_mass:.2f} kg'
                   f'   |   tool {cfg.end_effector.mass:.2f} kg'
                   f'   |   g {cfg.gravity:.2f} m/s\u00b2'
                   f'   |   reach {self.model.geometric_max_reach * 1000:.0f} mm')
        info = QLabel(summary)
        info.setTextFormat(Qt.PlainText)
        info.setStyleSheet(f'color: rgb({DIM.red()},{DIM.green()},{DIM.blue()})')
        row.addWidget(info)
        row.addStretch(1)

        self.link_label = QLabel('waiting for /joint_states ...')
        self.link_label.setStyleSheet(f'color: rgb({WARN.red()},{WARN.green()},'
                                      f'{WARN.blue()})')
        row.addWidget(self.link_label)

        path = QLabel(cfg.source_path)
        path.setObjectName('mono')
        path.setStyleSheet(f'color: rgb({DIM.red()},{DIM.green()},{DIM.blue()});'
                           ' font-size: 10px;')
        row.addWidget(path)
        return box

    def _build_joint_panel(self) -> QWidget:
        box = QGroupBox('JOINTS   -   speed and torque against their limits')
        grid = QGridLayout(box)
        grid.setContentsMargins(10, 16, 10, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        for col, text in enumerate(
                ('joint', 'position', 'speed  [rad/s]', 'torque  [N·m]')):
            label = QLabel(text)
            label.setStyleSheet(f'color: rgb({DIM.red()},{DIM.green()},{DIM.blue()})')
            grid.addWidget(label, 0, col)
        grid.setColumnStretch(2, 3)
        grid.setColumnStretch(3, 4)

        self.pos_labels: List[QLabel] = []
        self.speed_bars: List[MeterBar] = []
        self.torque_bars: List[MeterBar] = []
        for i, joint in enumerate(self.cfg.joints):
            name = QLabel(joint.name)
            name.setToolTip(f'actuator: {joint.actuator.name}\n'
                            f'gear ratio {joint.actuator.gear_ratio:.0f}, '
                            f'peak {joint.effort_limit:.1f} N·m, '
                            f'continuous {joint.actuator.output_continuous_torque:.1f} N·m')
            grid.addWidget(name, i + 1, 0)

            pos = QLabel('0.0°')
            pos.setObjectName('mono')
            pos.setMinimumWidth(64)
            pos.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(pos, i + 1, 1)
            self.pos_labels.append(pos)

            speed = MeterBar(unit='', limit=joint.usable_speed, decimals=2)
            grid.addWidget(speed, i + 1, 2)
            self.speed_bars.append(speed)

            torque = MeterBar(unit='', limit=joint.effort_limit, decimals=1)
            torque.setToolTip(
                'solid bar: torque from the inverse-dynamics model\n'
                'dashed line: effort reported by the simulator')
            grid.addWidget(torque, i + 1, 3)
            self.torque_bars.append(torque)

        row = len(self.cfg.joints) + 1
        note = QLabel('bar = model torque from measured motion   ·   '
                      'dashed line = simulator effort')
        note.setStyleSheet(f'color: rgb({DIM.red()},{DIM.green()},{DIM.blue()});'
                           ' font-size: 10px;')
        grid.addWidget(note, row, 0, 1, 4)

        self.plot = TimePlot(window_s=20.0)
        self.plot.add_series('TCP speed [m/s]', ACCENT)
        self.plot.add_series('max torque use [x100 %]', WARN, scale=1.0)
        self.plot.add_limit(float(self.cfg.spec_targets.get('tcp_speed', 0.2)),
                            GOOD, 'TCP speed spec')
        self.plot.add_limit(1.0, BAD, 'torque limit')
        grid.addWidget(self.plot, row + 1, 0, 1, 4)
        grid.setRowStretch(row + 1, 1)
        return box

    def _build_capability_panel(self) -> QWidget:
        box = QGroupBox('CAPABILITY AT THE CURRENT POSE')
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 16, 10, 10)
        layout.setSpacing(6)

        t = self.cfg.spec_targets
        grid = QGridLayout()
        grid.setSpacing(6)
        self.payload_tile = BigNumber(
            'PAYLOAD AT TCP', 'kg', 2,
            target=float(t.get('payload_at_full_reach', 2.0)),
            higher_is_better=True)
        self.speed_tile = BigNumber(
            'TCP SPEED', 'm/s', 3, target=float(t.get('tcp_speed', 0.2)),
            higher_is_better=False)
        self.reach_tile = BigNumber('REACH FROM SHOULDER', 'mm', 0)
        self.droop_tile = BigNumber(
            'TCP DROOP UNDER LOAD', 'mm', 1,
            target=float(t.get('raw_positioning_accuracy', 0.01)) * 1000.0,
            higher_is_better=False)
        grid.addWidget(self.payload_tile, 0, 0)
        grid.addWidget(self.speed_tile, 0, 1)
        grid.addWidget(self.reach_tile, 1, 0)
        grid.addWidget(self.droop_tile, 1, 1)
        layout.addLayout(grid)

        self.detail = QLabel('')
        self.detail.setObjectName('mono')
        self.detail.setStyleSheet('font-size: 11px;')
        self.detail.setTextFormat(Qt.RichText)
        self.detail.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.detail)

        self.scorecard = ScoreCard()
        layout.addWidget(self.scorecard)
        layout.addStretch(1)
        return box

    def _build_controls(self) -> QWidget:
        box = QGroupBox('TEST DRIVE')
        outer = QVBoxLayout(box)
        outer.setContentsMargins(10, 16, 10, 10)

        top = QHBoxLayout()
        top.addWidget(QLabel('pose'))
        self.pose_box = QComboBox()
        for name in self.cfg.test_poses:
            self.pose_box.addItem(name)
        top.addWidget(self.pose_box)

        go = QPushButton('Move')
        go.clicked.connect(self._on_move)
        top.addWidget(go)

        top.addSpacing(14)
        top.addWidget(QLabel('TCP speed'))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.01, 2.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setSuffix(' m/s')
        self.speed_spin.setValue(float(self.cfg.control.get('tcp_speed_limit', 0.2)))
        self.speed_spin.setToolTip(
            'Commanded average TCP speed; the move duration is set from the '
            'straight-line distance. The controller interpolates in joint '
            'space, so the peak runs well above this - watch the TCP SPEED '
            'tile. Raise it to see which joint saturates first.')
        top.addWidget(self.speed_spin)

        self.sweep_button = QPushButton('Start speed sweep')
        self.sweep_button.setCheckable(True)
        self.sweep_button.clicked.connect(self._on_sweep)
        self.sweep_button.setToolTip(
            'Drive back and forth between the two poses at the commanded speed '
            'and watch the peak torque and speed build up.')
        top.addWidget(self.sweep_button)

        stop = QPushButton('STOP')
        stop.setObjectName('stop')
        stop.clicked.connect(self.node.hold)
        top.addWidget(stop)
        top.addStretch(1)
        outer.addLayout(top)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel('payload at TCP'))
        self.payload_spin = QDoubleSpinBox()
        self.payload_spin.setRange(0.0, 50.0)
        self.payload_spin.setSingleStep(0.25)
        self.payload_spin.setDecimals(2)
        self.payload_spin.setSuffix(' kg')
        self.payload_spin.setValue(self.state.payload_mass)
        self.payload_spin.valueChanged.connect(self._on_payload)
        self.payload_spin.setToolTip(
            'Load carried at the TCP for the torque and droop figures. This is '
            'analysis only: to make Gazebo carry it, relaunch with '
            'payload_mass:=<kg>.')
        bottom.addWidget(self.payload_spin)

        if self.cfg.end_effector.finger_joint_names:
            bottom.addSpacing(14)
            bottom.addWidget(QLabel('gripper'))
            self.grip_slider = QSlider(Qt.Horizontal)
            self.grip_slider.setRange(0, int(self.cfg.end_effector.stroke * 1000))
            self.grip_slider.setValue(int(self.cfg.end_effector.stroke * 1000))
            self.grip_slider.setFixedWidth(160)
            self.grip_slider.valueChanged.connect(
                lambda mm: self.node.send_gripper(mm / 1000.0))
            bottom.addWidget(self.grip_slider)
            self.grip_label = QLabel('180 mm')
            self.grip_label.setObjectName('mono')
            self.grip_slider.valueChanged.connect(
                lambda mm: self.grip_label.setText(f'{mm} mm'))
            bottom.addWidget(self.grip_label)

        bottom.addSpacing(14)
        reset = QPushButton('Reset peaks')
        reset.clicked.connect(self.state.reset_peaks)
        bottom.addWidget(reset)

        report = QPushButton('Full spec report')
        report.clicked.connect(self._on_report)
        bottom.addWidget(report)
        bottom.addStretch(1)
        outer.addLayout(bottom)
        return box

    # ------------------------------------------------------------ callbacks
    def _on_payload(self, value: float) -> None:
        self.state.payload_mass = float(value)

    def _target_pose(self, name: str) -> np.ndarray:
        return self.model.resolve_pose(self.cfg.test_poses.get(name, name))

    def _duration_for(self, q_target: np.ndarray) -> float:
        """Time such that the TCP travels at roughly the commanded speed."""
        here = self.model.fk(self.state.q)
        there = self.model.fk(q_target)
        distance = float(np.linalg.norm(there - here))
        return max(distance / max(self.speed_spin.value(), 1e-3), 0.5)

    def _on_move(self) -> None:
        q = self._target_pose(self.pose_box.currentText())
        self.node.send_pose(q, self._duration_for(q))

    def _on_sweep(self, checked: bool) -> None:
        self._sweep = checked
        self.sweep_button.setText('Stop speed sweep' if checked
                                  else 'Start speed sweep')
        if checked:
            self.state.reset_peaks()
            self.plot.clear()
            self._sweep_flip = 0.0

    def _sweep_poses(self) -> List[str]:
        names = [n for n in ('home', 'full_reach', 'reach_700', 'overhead')
                 if n in self.cfg.test_poses]
        return names[:2] if len(names) >= 2 else list(self.cfg.test_poses)[:2]

    def _on_report(self) -> None:
        report = SpecReport(self.cfg, payload_override=self.state.payload_mass or None)
        report.run()
        text = report.render(color=False)
        print(text)

        dialog = QDialog(self)
        dialog.setWindowTitle('Spec report')
        dialog.resize(920, 760)
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        font = QFont('monospace')
        font.setPointSize(9)
        font.setStyleHint(QFont.Monospace)
        view.setFont(font)
        layout.addWidget(view)
        dialog.setStyleSheet(STYLE)
        dialog.exec_() if hasattr(dialog, 'exec_') else dialog.exec()

    # ----------------------------------------------------------------- tick
    def _tick(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)

        fresh = (self.node.last_msg_time is not None
                 and time.time() - self.node.last_msg_time < 1.0)
        if fresh:
            self.link_label.setText('/joint_states live')
            self.link_label.setStyleSheet(
                f'color: rgb({GOOD.red()},{GOOD.green()},{GOOD.blue()})')
        else:
            self.link_label.setText('waiting for /joint_states ...')
            self.link_label.setStyleSheet(
                f'color: rgb({WARN.red()},{WARN.green()},{WARN.blue()})')

        if self._sweep:
            now = time.monotonic()
            if now >= self._sweep_flip:
                names = self._sweep_poses()
                if names:
                    q = self._target_pose(names[self._sweep_index % len(names)])
                    duration = self._duration_for(q)
                    self.node.send_pose(q, duration)
                    self._sweep_index += 1
                    self._sweep_flip = now + duration + 0.6

        if not self.state.ready:
            return
        self._refresh(self.state.metrics())

    def _refresh(self, mtr) -> None:
        q = mtr['q']
        qd = mtr['qd']
        tau = mtr['tau_model']
        tau_sim = mtr['tau_sim']
        for i in range(len(self.cfg.joints)):
            self.pos_labels[i].setText(f'{np.degrees(q[i]):7.1f}°')
            self.speed_bars[i].set_value(qd[i])
            self.torque_bars[i].set_value(
                tau[i], tau_sim[i] if abs(tau_sim[i]) > 1e-6 else None)

        speed = float(mtr['tcp_speed'])
        self.speed_tile.set_value(speed, f'peak {mtr["peak_tcp_speed"]:.3f}')
        self.payload_tile.set_value(
            float(mtr['payload_capacity']),
            f'limited by {mtr["limiting_joint"]}')
        self.reach_tile.set_value(
            float(mtr['reach']) * 1000.0,
            f'geometric max {self.model.geometric_max_reach * 1000:.0f} mm')
        bend, tors, joints = mtr['droop_parts']
        self.droop_tile.set_value(
            float(mtr['droop']) * 1000.0,
            f'tubes {bend * 1000:.2f} + gearboxes {joints * 1000:.2f} mm')

        def force(value: float) -> str:
            """Near a singular direction the joints are barely loaded at all,
            so the figure runs away; say so instead of printing it."""
            return '>999' if value > 999.0 else f'{value:.0f}'

        tcp = mtr['tcp']
        util = mtr['utilisation']
        worst = int(np.argmax(util))
        buses = '  '.join(f'{v:.0f} V {a:.1f} A'
                          for v, a in sorted(mtr['power_per_bus'].items()))
        stress = float(np.max(mtr['stress_utilisation'])) * 100.0
        self.detail.setText(
            f'TCP&nbsp; x {tcp[0]:+.3f} &nbsp; y {tcp[1]:+.3f} &nbsp; '
            f'z {tcp[2]:+.3f} m<br>'
            f'worst joint&nbsp; {self.cfg.joint_names[worst]} at '
            f'{util[worst] * 100:.0f} % torque &nbsp;|&nbsp; peak this run '
            f'{mtr["peak_utilisation"] * 100:.0f} %<br>'
            f'reachable TCP speed here&nbsp; {mtr["max_tcp_speed"]:.2f} m/s '
            f'&nbsp;|&nbsp; push down {force(mtr["contact_force_down"])} N '
            f'&nbsp;|&nbsp; push forward {force(mtr["contact_force_fwd"])} N<br>'
            f'tube stress&nbsp; {stress:.1f} % of yield &nbsp;|&nbsp; '
            f'power {mtr["power_w"]:.0f} W &nbsp; ({buses})<br>'
            f'gripper opening&nbsp; {max(mtr["gripper_opening"], 0.0) * 1000:.0f} mm')

        t = time.monotonic() - self.t0
        self.plot.push(t, {
            'TCP speed [m/s]': speed,
            'max torque use [x100 %]': float(np.max(util)),
        })
        self.scorecard.set_rows(self.state.spec_status(mtr))


def main(argv: Optional[List[str]] = None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = DashboardNode()
    app = QApplication(sys.argv[:1])
    window = Dashboard(node)
    window.show()
    try:
        code = app.exec_() if hasattr(app, 'exec_') else app.exec()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
