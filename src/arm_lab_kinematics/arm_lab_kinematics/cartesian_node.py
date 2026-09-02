"""Execute straight-line TCP moves with a real Cartesian speed limit.

    ros2 run arm_lab_kinematics cartesian_move
    ros2 topic pub --once /arm_lab/cartesian_target geometry_msgs/msg/PoseStamped \\
        '{pose: {position: {x: 0.62, y: -0.20, z: 0.18}}}'

The trajectory controller interpolates in joint space, so a Cartesian speed
limit cannot be enforced by timing a two-point trajectory. This node plans the
line, solves IK along it, and hands the controller a dense trajectory whose TCP
speed is the commanded one.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from arm_lab_model.config import load_config
from arm_lab_model.kinematics import ArmModel

from .cartesian import plan_line
from .ik import IKSolver
from .workspace import TOOL_DOWN


def _quaternion_to_matrix(q) -> Optional[np.ndarray]:
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return None
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class CartesianMoveNode(Node):
    def __init__(self):
        super().__init__('cartesian_move')
        self.declare_parameter('config_file', '')
        self.declare_parameter('speed', 0.0)
        self.declare_parameter('accel', 0.0)
        self.declare_parameter('arm_controller', 'arm_controller')

        self.cfg = load_config(self.get_parameter('config_file').value or None)
        self.model = ArmModel(self.cfg)
        self.solver = IKSolver(self.model)
        motion = self.cfg.raw.get('motion', {})
        speed = float(self.get_parameter('speed').value)
        accel = float(self.get_parameter('accel').value)
        self.speed = speed if speed > 0 else float(
            motion.get('cartesian_speed', 0.2))
        self.accel = accel if accel > 0 else float(
            motion.get('cartesian_accel', 0.3))

        controller = self.get_parameter('arm_controller').value
        self.pub = self.create_publisher(
            JointTrajectory, f'/{controller}/joint_trajectory', 10)
        self.info = self.create_publisher(String, '/arm_lab/cartesian_plan', 10)
        self.create_subscription(JointState, '/joint_states', self._on_state, 20)
        self.create_subscription(PoseStamped, '/arm_lab/cartesian_target',
                                 self._on_target, 5)

        self.q = np.zeros(self.model.n)
        self.have_state = False
        self._index = {n: i for i, n in enumerate(self.cfg.joint_names)}
        self.get_logger().info(
            f'cartesian move ready: {self.speed:.3f} m/s, '
            f'{self.accel:.3f} m/s^2. Publish a PoseStamped on '
            '/arm_lab/cartesian_target')

    def _on_state(self, msg: JointState) -> None:
        for k, name in enumerate(msg.name):
            idx = self._index.get(name)
            if idx is not None and k < len(msg.position):
                self.q[idx] = float(msg.position[k])
        self.have_state = True

    def _on_target(self, msg: PoseStamped) -> None:
        if not self.have_state:
            self.get_logger().warn('no /joint_states yet; ignoring target')
            return
        target = np.array([msg.pose.position.x, msg.pose.position.y,
                           msg.pose.position.z])
        rotation = _quaternion_to_matrix(msg.pose.orientation)
        if rotation is None:
            rotation = TOOL_DOWN
            self.get_logger().info('target has no orientation; using tool-down')

        t0 = time.time()
        path = plan_line(self.model, self.q, target, rotation,
                         speed=self.speed, accel=self.accel,
                         solver=self.solver)
        if not path.feasible:
            text = 'plan failed: ' + '; '.join(path.notes)
            self.get_logger().error(text)
            self.info.publish(String(data=text))
            return

        traj = JointTrajectory()
        traj.joint_names = list(self.cfg.joint_names)
        for k in range(len(path.times)):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in path.joints[k]]
            point.velocities = [float(v) for v in path.joint_speeds[k]]
            seconds = float(path.times[k])
            point.time_from_start = DurationMsg(
                sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))
            traj.points.append(point)
        self.pub.publish(traj)

        text = (f'planned {len(path.times)} waypoints over '
                f'{path.path_length:.3f} m in {path.duration:.2f} s; '
                f'peak TCP speed {path.peak_tcp_speed:.3f} m/s '
                f'(limit {self.speed:.3f}); planned in '
                f'{(time.time() - t0) * 1000:.0f} ms')
        if path.notes:
            text += ' | ' + '; '.join(path.notes)
        self.get_logger().info(text)
        self.info.publish(String(data=text))


def main(argv: Optional[List[str]] = None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = CartesianMoveNode()
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
