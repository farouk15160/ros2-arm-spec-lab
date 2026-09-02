"""Publish live capability figures so they can be plotted, logged or bagged.

    ros2 run arm_lab_gui capability_node
    ros2 topic echo /arm_lab/payload_capacity
    ros2 run rqt_plot rqt_plot /arm_lab/tcp_speed/data /arm_lab/torque_utilisation/data

Headless twin of the dashboard: same model, no window.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

from arm_lab_model.config import load_config

from .state import LiveState


class CapabilityNode(Node):
    def __init__(self):
        super().__init__('arm_capability')
        self.declare_parameter('config_file', '')
        self.declare_parameter('ee_mass', -1.0)
        self.declare_parameter('gravity', -1.0)
        self.declare_parameter('payload_mass', 0.0)
        self.declare_parameter('publish_rate', 20.0)

        def opt(name):
            value = float(self.get_parameter(name).value)
            return None if value < 0.0 else value

        self.cfg = load_config(self.get_parameter('config_file').value or None,
                               ee_mass=opt('ee_mass'), gravity=opt('gravity'))
        self.state = LiveState(
            self.cfg, payload_mass=float(self.get_parameter('payload_mass').value))

        self.pubs = {
            name: self.create_publisher(Float64, f'/arm_lab/{name}', 10)
            for name in ('tcp_speed', 'payload_capacity', 'torque_utilisation',
                         'tcp_droop', 'reach', 'power')
        }
        self.torque_pub = self.create_publisher(
            Float64MultiArray, '/arm_lab/joint_torque_model', 10)
        self.diag_pub = self.create_publisher(
            DiagnosticArray, '/diagnostics', 10)

        self.create_subscription(JointState, '/joint_states', self._on_state, 20)
        rate = float(self.get_parameter('publish_rate').value)
        self.create_timer(1.0 / max(rate, 1.0), self._publish)
        self.get_logger().info(
            f'capability publisher up: {self.cfg.dof} dof, '
            f'{self.cfg.arm_mass:.2f} kg, g={self.cfg.gravity:.2f}')

    def _on_state(self, msg: JointState) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.state.ingest(list(msg.name), list(msg.position), list(msg.velocity),
                          list(msg.effort), stamp if stamp > 0 else time.time())

    def _publish(self) -> None:
        if not self.state.ready:
            return
        m = self.state.metrics()
        util = float(np.max(m['utilisation']))
        values = {
            'tcp_speed': float(m['tcp_speed']),
            'payload_capacity': float(m['payload_capacity']),
            'torque_utilisation': util,
            'tcp_droop': float(m['droop']),
            'reach': float(m['reach']),
            'power': float(m['power_w']),
        }
        for name, value in values.items():
            self.pubs[name].publish(Float64(data=value))

        torques = Float64MultiArray()
        torques.data = [float(v) for v in m['tau_model']]
        self.torque_pub.publish(torques)

        t = self.cfg.spec_targets
        status = DiagnosticStatus(name='arm_lab: capability',
                                 hardware_id=self.cfg.name)
        breaches = []
        if values['tcp_speed'] > float(t.get('tcp_speed', 0.2)):
            breaches.append('TCP over speed')
        if util > 1.0:
            breaches.append('joint torque over limit')
        if values['tcp_droop'] > float(t.get('raw_positioning_accuracy', 0.01)):
            breaches.append('droop over accuracy budget')
        status.level = DiagnosticStatus.WARN if breaches else DiagnosticStatus.OK
        status.message = ', '.join(breaches) if breaches else 'within spec'
        status.values = [
            KeyValue(key=k, value=f'{v:.4f}') for k, v in values.items()]
        status.values.append(
            KeyValue(key='limiting_joint', value=str(m['limiting_joint'])))
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status = [status]
        self.diag_pub.publish(array)


def main(argv: Optional[List[str]] = None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = CapabilityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
