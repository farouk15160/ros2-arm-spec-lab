"""Headless speed and torque test: drive the arm and report what it managed.

    ros2 run arm_lab_gui speed_test
    ros2 run arm_lab_gui speed_test --ros-args -p target_speed:=0.4 -p cycles:=6

Commands a back-and-forth move between two poses at a requested TCP speed, then
prints the achieved TCP speed, the peak joint torques and how close each joint
came to its limits. Use it to check whether a speed figure survives contact with
the controller, not just the kinematics.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from arm_lab_model.config import load_config

from .state import LiveState


class SpeedTest(Node):
    def __init__(self):
        super().__init__('arm_speed_test')
        self.declare_parameter('config_file', '')
        self.declare_parameter('payload_mass', 0.0)
        self.declare_parameter('target_speed', 0.0)
        self.declare_parameter('cycles', 4)
        self.declare_parameter('pose_a', 'home')
        self.declare_parameter('pose_b', 'full_reach')
        self.declare_parameter('arm_controller', 'arm_controller')

        self.cfg = load_config(self.get_parameter('config_file').value or None)
        self.state = LiveState(
            self.cfg, payload_mass=float(self.get_parameter('payload_mass').value))
        self.model = self.state.model

        target = float(self.get_parameter('target_speed').value)
        self.target_speed = target if target > 0 else float(
            self.cfg.control.get('tcp_speed_limit', 0.2))
        self.cycles = int(self.get_parameter('cycles').value)

        controller = self.get_parameter('arm_controller').value
        self.pub = self.create_publisher(
            JointTrajectory, f'/{controller}/joint_trajectory', 10)
        self.create_subscription(JointState, '/joint_states', self._on_state, 20)

        self.poses = [self._pose(self.get_parameter('pose_a').value),
                      self._pose(self.get_parameter('pose_b').value)]
        self.peak_torque = np.zeros(self.model.n)
        self.samples: List[float] = []

    def _pose(self, name: str) -> np.ndarray:
        return self.model.resolve_pose(self.cfg.test_poses.get(name, name))

    def _on_state(self, msg: JointState) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.state.ingest(list(msg.name), list(msg.position), list(msg.velocity),
                          list(msg.effort), stamp if stamp > 0 else time.time())

    def _send(self, q: np.ndarray, duration: float) -> None:
        msg = JointTrajectory()
        msg.joint_names = list(self.cfg.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q]
        point.velocities = [0.0] * len(q)
        point.time_from_start = DurationMsg(
            sec=int(duration), nanosec=int((duration % 1.0) * 1e9))
        msg.points = [point]
        self.pub.publish(msg)

    def _spin(self, seconds: float, record: bool) -> None:
        end = time.time() + seconds
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)
            if record and self.state.ready:
                m = self.state.metrics()
                self.samples.append(float(m['tcp_speed']))
                self.peak_torque = np.maximum(
                    self.peak_torque, np.abs(m['tau_model']))

    def run(self) -> int:
        self.get_logger().info('waiting for /joint_states ...')
        deadline = time.time() + 30.0
        while not self.state.ready and time.time() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)
        if not self.state.ready:
            self.get_logger().error('no /joint_states; is the simulation running?')
            return 2

        distance = float(np.linalg.norm(
            self.model.fk(self.poses[0]) - self.model.fk(self.poses[1])))
        duration = max(distance / self.target_speed, 0.5)
        print(f'\ncommanded TCP speed  {self.target_speed:.3f} m/s')
        print(f'path length          {distance:.3f} m  ->  {duration:.2f} s per leg')
        print(f'cycles               {self.cycles}\n')

        self._send(self.poses[0], duration)
        self._spin(duration + 1.0, record=False)
        self.state.reset_peaks()

        for cycle in range(self.cycles):
            target = self.poses[(cycle + 1) % 2]
            self._send(target, duration)
            self._spin(duration + 0.4, record=True)
            print(f'  leg {cycle + 1}/{self.cycles} done, '
                  f'peak TCP speed so far {max(self.samples or [0.0]):.3f} m/s')

        achieved = max(self.samples) if self.samples else 0.0
        mean = float(np.mean([s for s in self.samples if s > 0.02])) \
            if self.samples else 0.0
        print('\n' + '-' * 72)
        print(f'{"peak TCP speed":<28}{achieved:.3f} m/s   '
              f'(commanded {self.target_speed:.3f})')
        print(f'{"mean TCP speed in motion":<28}{mean:.3f} m/s')
        print(f'{"tracking":<28}{achieved / self.target_speed * 100:.0f} % of command')
        print('-' * 72)
        print(f'{"joint":<10}{"peak N.m":>10}{"limit N.m":>12}{"use %":>8}'
              f'{"speed limit":>13}')
        for i, joint in enumerate(self.cfg.joints):
            print(f'{joint.name:<10}{self.peak_torque[i]:>10.1f}'
                  f'{joint.effort_limit:>12.1f}'
                  f'{self.peak_torque[i] / joint.effort_limit * 100:>8.0f}'
                  f'{joint.usable_speed:>13.2f}')
        print('-' * 72)
        over = [j.name for i, j in enumerate(self.cfg.joints)
                if self.peak_torque[i] > j.effort_limit]
        if over:
            print('OVER TORQUE LIMIT: ' + ', '.join(over))
        elif achieved > 1.15 * self.target_speed:
            print(f'Peak TCP speed overshot the command by '
                  f'{(achieved / self.target_speed - 1) * 100:.0f} %.')
            print('Expected: the duration was set from the straight-line TCP '
                  'distance, but the controller interpolates in joint space, so '
                  'the real path is longer and a spline peaks above its own '
                  'average. If the TCP speed limit is a safety requirement it '
                  'has to be enforced by a Cartesian speed monitor, not by '
                  'timing the trajectory.')
        elif achieved < 0.9 * self.target_speed:
            print('Did not reach the commanded speed: the trajectory controller or '
                  'a joint velocity limit is the binding constraint, not torque.')
        else:
            print('Command met with torque to spare.')
        return 0


def main(argv: Optional[List[str]] = None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = SpeedTest()
    try:
        return node.run()
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
