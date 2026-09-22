"""MuJoCo stand-in for Gazebo + ros2_control, with an optional viewer window.

    ros2 launch arm_lab_bringup sim.launch.py simulator:=mujoco
    ros2 launch arm_lab_bringup sim.launch.py simulator:=mujoco mujoco_gui:=false
    ros2 run arm_lab_gui mujoco_sim --ros-args -p config_file:=/path/arm.yaml

It speaks the interface the dashboard, pick_place, cartesian_move and
speed_test already use against Gazebo, so they run unchanged:

    publishes   /joint_states   arm joints and jaws: position, velocity, effort
                /clock          simulation time, for use_sim_time nodes
                /arm_lab/objects  the world's sample boxes, as RViz markers
    subscribes  /<arm_controller>/joint_trajectory
                /<gripper_controller>/commands   force per jaw (+ opens) in force
                                                 grasp mode, half-opening otherwise
    action      /<arm_controller>/follow_joint_trajectory

The world's ground and sample boxes are loaded from the same SDF Gazebo uses,
and the jaws grip them by contact and friction. The joint servo is the
computed-torque controller in arm_lab_model.mujoco_sim, not ros2_control's PID.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from typing import List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from arm_lab_model.config import load_config
from arm_lab_model.mujoco_sim import ArmSimulation, TrajectoryPoint

GOAL_TOLERANCE = 0.02       # rad, worst joint, before a goal counts as reached
OBJECT_PERIOD = 0.1         # s between object marker updates
SETTLE_TIMEOUT = 1.0        # s of simulation after the trajectory ends


def _seconds(duration) -> float:
    return duration.sec + duration.nanosec * 1e-9


def _stamp(t: float) -> TimeMsg:
    sec = int(t)
    return TimeMsg(sec=sec, nanosec=int((t - sec) * 1e9))


def points_from_msg(msg: JointTrajectory) -> List[TrajectoryPoint]:
    n = len(msg.joint_names)
    points = []
    for k, p in enumerate(msg.points):
        if len(p.positions) != n:
            raise ValueError(f'point {k} has {len(p.positions)} positions for {n} joints')
        velocities = np.array(p.velocities, float) if len(p.velocities) == n else None
        points.append(TrajectoryPoint(_seconds(p.time_from_start),
                                      np.array(p.positions, float), velocities))
    return points


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        self.declare_parameter('config_file', '')
        self.declare_parameter('ee_mass', -1.0)
        self.declare_parameter('gravity', -1.0)
        self.declare_parameter('payload_mass', 0.0)
        self.declare_parameter('initial_pose', 'home')
        self.declare_parameter('world', '')
        self.declare_parameter('gui', True)
        self.declare_parameter('real_time_factor', 1.0)
        self.declare_parameter('publish_rate', 0.0)     # 0 = control.update_rate
        self.declare_parameter('bandwidth_hz', 5.0)
        self.declare_parameter('arm_controller', 'arm_controller')
        self.declare_parameter('gripper_controller', 'gripper_controller')

        def param(name):
            return self.get_parameter(name).value

        def opt(name):
            value = float(param(name))
            return None if value < 0.0 else value

        self.cfg = load_config(param('config_file') or None,
                               ee_mass=opt('ee_mass'), gravity=opt('gravity'))
        self.sim = ArmSimulation(self.cfg,
                                 payload=float(param('payload_mass')),
                                 world=param('world') or None,
                                 initial_pose=param('initial_pose'),
                                 bandwidth_hz=float(param('bandwidth_hz')))
        self.lock = threading.Lock()
        self.gui = bool(param('gui'))
        self.rtf = max(float(param('real_time_factor')), 1e-3)
        rate = float(param('publish_rate')) or float(self.cfg.control.get('update_rate', 200))
        self.publish_period = 1.0 / max(rate, 1.0)

        arm = param('arm_controller')
        grip = param('gripper_controller')
        self.state_pub = self.create_publisher(JointState, '/joint_states', 20)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.objects_pub = self.create_publisher(MarkerArray, '/arm_lab/objects', 10)
        self.create_subscription(JointTrajectory, f'/{arm}/joint_trajectory',
                                 self._on_trajectory, 10)
        self.create_subscription(Float64MultiArray, f'/{grip}/commands',
                                 self._on_gripper, 10)
        self.action = ActionServer(
            self, FollowJointTrajectory, f'/{arm}/follow_joint_trajectory',
            execute_callback=self._execute,
            goal_callback=lambda _goal: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup())

        boxes = [b.name for b in self.sim.scene.boxes] if self.sim.scene else []
        self.get_logger().info(
            f'{self.cfg.name}: {self.cfg.dof} joints, payload '
            f'{float(param("payload_mass")):.2f} kg, gravity {self.cfg.gravity:.2f} m/s^2, '
            f'objects {boxes}, viewer {"on" if self.gui else "off"}, '
            f'servo bandwidth {float(param("bandwidth_hz")):.1f} Hz')

    # ------------------------------------------------------------- commands
    def _on_trajectory(self, msg: JointTrajectory) -> None:
        try:
            points = points_from_msg(msg)
            with self.lock:
                self.sim.set_trajectory(list(msg.joint_names), points)
        except ValueError as exc:
            self.get_logger().error(f'trajectory rejected: {exc}')

    def _on_gripper(self, msg: Float64MultiArray) -> None:
        try:
            with self.lock:
                self.sim.set_gripper(list(msg.data))
        except ValueError as exc:
            self.get_logger().error(f'gripper command rejected: {exc}')

    def _execute(self, goal_handle):
        result = FollowJointTrajectory.Result()
        traj = goal_handle.request.trajectory
        try:
            points = points_from_msg(traj)
            with self.lock:
                self.sim.set_trajectory(list(traj.joint_names), points)
                generation = self.sim.generation
                end = self.sim.time + self.sim.trajectory.duration
        except ValueError as exc:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = str(exc)
            return result

        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = list(self.cfg.joint_names)
        n = self.cfg.dof
        while rclpy.ok():
            with self.lock:
                now = self.sim.time
                superseded = self.sim.generation != generation
                q_d, v_d, _ = self.sim.desired()
                _, pos, vel, _ = self.sim.joint_state()
            if superseded:
                goal_handle.abort()
                result.error_string = 'preempted by a newer trajectory'
                return result
            if goal_handle.is_cancel_requested:
                with self.lock:
                    self.sim.hold()
                goal_handle.canceled()
                result.error_string = 'canceled; holding position'
                return result
            error = q_d - np.array(pos[:n])
            feedback.header.stamp = _stamp(now)
            feedback.desired = JointTrajectoryPoint(
                positions=q_d.tolist(), velocities=v_d.tolist())
            feedback.actual = JointTrajectoryPoint(positions=pos[:n], velocities=vel[:n])
            feedback.error = JointTrajectoryPoint(positions=error.tolist())
            goal_handle.publish_feedback(feedback)
            if now >= end and float(np.max(np.abs(error))) <= GOAL_TOLERANCE:
                goal_handle.succeed()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return result
            if now >= end + SETTLE_TIMEOUT:
                goal_handle.abort()
                result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                result.error_string = (f'worst joint error {np.max(np.abs(error)):.3f} rad '
                                       f'{SETTLE_TIMEOUT:.1f} s after the trajectory ended')
                return result
            time.sleep(0.05)
        goal_handle.abort()
        return result

    # ----------------------------------------------------------------- loop
    def _publish(self) -> None:
        with self.lock:
            now = self.sim.time
            names, pos, vel, eff = self.sim.joint_state()
        stamp = _stamp(now)
        self.clock_pub.publish(Clock(clock=stamp))
        msg = JointState(name=names, position=pos, velocity=vel, effort=eff)
        msg.header.stamp = stamp
        self.state_pub.publish(msg)

    def _publish_objects(self) -> None:
        with self.lock:
            now = self.sim.time
            poses = self.sim.object_poses()
        markers = MarkerArray()
        for k, (box, pos, quat) in enumerate(poses):
            m = Marker(ns='objects', id=k, type=Marker.CUBE, action=Marker.ADD)
            m.header.frame_id = 'world'
            m.header.stamp = _stamp(now)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
            (m.pose.orientation.w, m.pose.orientation.x,
             m.pose.orientation.y, m.pose.orientation.z) = (float(v) for v in quat)
            m.scale.x, m.scale.y, m.scale.z = (float(v) for v in box.size)
            m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in box.rgba)
            markers.markers.append(m)
        self.objects_pub.publish(markers)

    def _open_viewer(self):
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.sim.m, self.sim.d,
                                                  show_right_ui=False)
        except Exception as exc:  # no display, no GL: carry on headless
            self.get_logger().error(f'MuJoCo viewer unavailable ({exc}); running headless')
            return None
        viewer.cam.lookat[:] = [0.35, 0.0, 0.30]
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0
        return viewer

    def run(self) -> None:
        """Step physics in real time, publish state, and keep the viewer fed."""
        viewer = self._open_viewer() if self.gui else None
        wall0, sim0 = time.monotonic(), self.sim.time
        next_publish, next_objects, next_frame = self.sim.time, self.sim.time, 0.0
        max_steps = int(0.05 / self.sim.m.opt.timestep)      # 50 ms per pass
        try:
            while rclpy.ok():
                target = sim0 + (time.monotonic() - wall0) * self.rtf
                steps = 0
                while self.sim.time < target and steps < max_steps:
                    with self.lock:
                        self.sim.step()
                    steps += 1
                    if self.sim.time >= next_publish:
                        self._publish()
                        next_publish += self.publish_period
                    if self.sim.time >= next_objects:
                        self._publish_objects()
                        next_objects += OBJECT_PERIOD
                if steps == max_steps:
                    # Falling behind real time: rebase rather than spiral.
                    wall0, sim0 = time.monotonic(), self.sim.time
                if viewer is not None:
                    if not viewer.is_running():
                        self.get_logger().info('viewer closed; simulation continues headless')
                        viewer = None
                    elif time.monotonic() >= next_frame:
                        with self.lock:
                            viewer.sync()
                        next_frame = time.monotonic() + 1.0 / 60.0
                time.sleep(0.0005)
        finally:
            if viewer is not None:
                viewer.close()


def _prefer_x11() -> None:
    """Run the viewer through X11 (XWayland on a Wayland desktop) when it exists.

    On a Wayland session pyglfw loads its Wayland-only GLFW build, where the
    MuJoCo viewer cannot place its window and crashed on exit in testing;
    through X11 it ran and shut down cleanly. pyglfw picks its build when
    `import mujoco` imports it, so this must run first. Export
    PYGLFW_LIBRARY_VARIANT=wayland to override.
    """
    if os.environ.get('DISPLAY') and os.environ.get('XDG_SESSION_TYPE') == 'wayland':
        os.environ.setdefault('PYGLFW_LIBRARY_VARIANT', 'x11')


def _spin(executor) -> None:
    try:
        executor.spin()
    except Exception:
        # Ctrl-C shuts the context down underneath the executor; that is the
        # normal way out, not an error worth a traceback.
        if rclpy.ok():
            raise


def main(argv: Optional[List[str]] = None) -> int:
    _prefer_x11()
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = MujocoSimNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=_spin, args=(executor,), daemon=True)
    thread.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl-C in a terminal reaches every node twice: directly, and again
        # forwarded by `ros2 launch`. The second must not interrupt cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor.shutdown(timeout_sec=1.0)
        # Let the spin thread leave rclpy before the process tears down; exiting
        # with it still inside a wait set aborts with std::terminate.
        thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
