"""Plan position/pose targets, preview, execute and save confirmed successes.

ROS imports are confined to the adapter so target validation and lifecycle
contracts can be tested on machines without ROS.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math


def validate_target(xyz, xyzw=None):
    """Return finite coordinates and a unit quaternion (or unconstrained rotation)."""
    def vector(values, size):
        if len(values) != size or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                     or not math.isfinite(v) for v in values):
            raise ValueError(f'target requires {size} finite numeric coordinates')
        return tuple(float(v) for v in values)
    position = vector(xyz, 3)
    if xyzw is None:
        return position, None
    quaternion = vector(xyzw, 4)
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError('target orientation requires a nonzero quaternion')
    return position, tuple(value / norm for value in quaternion)


@dataclass(frozen=True)
class PlanSession:
    """A preview does not imply execution, and only successful execution is saved."""
    state: str = 'idle'
    trajectory: object = None
    start: object = None
    target: object = None

    @property
    def can_execute(self):
        return self.state == 'preview' and self.trajectory is not None

    @property
    def can_save(self):
        return self.state == 'succeeded' and self.trajectory is not None

    def planned(self, trajectory, start, target):
        return PlanSession('preview', trajectory, start, target)

    def executing(self):
        if not self.can_execute:
            raise ValueError('no unexecuted successful plan available')
        return replace(self, state='executing')

    def finished(self, success):
        if self.state != 'executing':
            raise ValueError('no execution in progress')
        return replace(self, state='succeeded' if success else 'failed')


def build_goal(group_name, group, options, frame, xyz, xyzw=None):
    """Build the real MoveGroup action request; position-only means no orientation constraint."""
    from geometry_msgs.msg import Pose
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
    from shape_msgs.msg import SolidPrimitive

    xyz, xyzw = validate_target(xyz, xyzw)
    goal = MoveGroup.Goal()
    request = goal.request
    request.group_name = group_name
    request.planner_id = options.get('planner_id', 'RRTConnectkConfigDefault')
    request.pipeline_id = 'ompl'
    request.num_planning_attempts = 1
    request.allowed_planning_time = float(options.get('planning_time', 5.0))
    request.max_velocity_scaling_factor = float(options.get('velocity_scaling', 0.1))
    request.max_acceleration_scaling_factor = float(options.get('acceleration_scaling', 0.1))
    request.start_state.is_diff = True
    position = PositionConstraint()
    position.header.frame_id = frame
    position.link_name = group['tip_link']
    position.weight = 1.0
    sphere = SolidPrimitive(type=SolidPrimitive.SPHERE,
                            dimensions=[float(options.get('position_tolerance', 0.005))])
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.w = 1.0
    position.constraint_region.primitives = [sphere]
    position.constraint_region.primitive_poses = [pose]
    constraints = Constraints(position_constraints=[position])
    if xyzw is not None:
        orientation = OrientationConstraint()
        orientation.header.frame_id = frame
        orientation.link_name = group['tip_link']
        orientation.orientation.x, orientation.orientation.y, orientation.orientation.z, orientation.orientation.w = xyzw
        tolerance = float(options.get('orientation_tolerance', 0.01))
        orientation.absolute_x_axis_tolerance = tolerance
        orientation.absolute_y_axis_tolerance = tolerance
        orientation.absolute_z_axis_tolerance = tolerance
        orientation.weight = 1.0
        constraints.orientation_constraints = [orientation]
    request.goal_constraints = [constraints]
    goal.planning_options.plan_only = True
    goal.planning_options.planning_scene_diff.is_diff = True
    goal.planning_options.planning_scene_diff.robot_state.is_diff = True
    return goal


def trajectory_message_digest(trajectory):
    """Canonical digest used to associate a bridge benchmark with one exact plan."""
    import hashlib
    content = {'joint_names': list(trajectory.joint_names), 'points': [
        {'time_from_start': float(point.time_from_start.sec + point.time_from_start.nanosec * 1e-9),
         'positions': list(point.positions), 'velocities': list(point.velocities),
         'accelerations': list(point.accelerations)} for point in trajectory.points]}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def create_node():
    """Construct the ROS adapter; no process is started by importing this module."""
    import rclpy
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PointStamped, PoseStamped
    from moveit_msgs.action import ExecuteTrajectory, MoveGroup
    from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from tf2_ros import Buffer, TransformListener
    import tf2_geometry_msgs  # noqa: F401 -- registers Point/PoseStamped transforms
    from arm_lab_model.project_config import load_project
    from arm_lab_model.physical_robot import resolve_robot, robot_fingerprint
    from .pipeline_moveit import build_moveit_config

    class TargetNode(Node):
        def __init__(self):
            super().__init__('pipeline_target')
            self.declare_parameter('project_file', '')
            self.declare_parameter('group', '')
            self.declare_parameter('scenario_id', 'ros_execution')
            self.project = load_project(self.get_parameter('project_file').value)
            self.robot = resolve_robot(self.project)
            self.options = self.project.sections['moveit']
            build_moveit_config(self.robot, self.options)
            self.group_name = self.get_parameter('group').value or self.options.get(
                'default_group', next(iter(self.options['groups'])))
            if self.group_name not in self.options['groups']:
                raise ValueError('unknown planning group: ' + self.group_name)
            self.group = self.options['groups'][self.group_name]
            self.session = PlanSession()
            self.benchmark_result = None
            self.scene_ready = not self.project.sections.get('environment', {}).get('enabled', False)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.plan_client = ActionClient(self, MoveGroup, '/move_action')
            self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.display = self.create_publisher(DisplayTrajectory, '/display_planned_path', latched)
            self.status = self.create_publisher(String, '/arm_lab/planning_status', latched)
            self.create_subscription(PointStamped, '/clicked_point', self.on_point, 10)
            self.create_subscription(PoseStamped, '/arm_lab/target', self.on_pose, 10)
            self.create_subscription(String, '/arm_lab/benchmark_result', self.on_benchmark, 10)
            self.create_subscription(String, '/arm_lab/scene_status', self.on_scene, latched)
            self.create_service(Trigger, '/arm_lab/execute_plan', self.execute)
            self.create_service(Trigger, '/arm_lab/save_trajectory', self.save)
            self.report('ready', 'Publish a target, preview in RViz, then call /arm_lab/execute_plan')

        def report(self, state, message, **details):
            text = json.dumps({'state': state, 'message': message, **details}, allow_nan=False)
            self.status.publish(String(data=text))
            self.get_logger().info(text)

        def on_scene(self, message):
            try:
                report = json.loads(message.data)
                self.scene_ready = report.get('ready') is True
            except (ValueError, TypeError, AttributeError) as exc:
                self.scene_ready = False
                self.report('scene_error', str(exc))

        def on_benchmark(self, message):
            try:
                report = json.loads(message.data)
                if (self.session.trajectory is not None and report.get('trajectory_sha256') ==
                        trajectory_message_digest(self.session.trajectory.joint_trajectory)):
                    self.benchmark_result = report.get('result')
            except (ValueError, TypeError, AttributeError) as exc:
                self.report('benchmark_error', str(exc))

        def on_point(self, message):
            pose = PoseStamped(header=message.header)
            pose.pose.position.x = message.point.x
            pose.pose.position.y = message.point.y
            pose.pose.position.z = message.point.z
            pose.pose.orientation.w = 1.0
            self.plan(pose, position_only=True)

        def on_pose(self, message):
            self.plan(message, position_only=False)

        def plan(self, pose, position_only):
            if self.session.state in ('planning', 'executing'):
                self.report('busy', 'Wait for the current planning/execution result')
                return
            self.session = PlanSession()  # A rejected new target cannot execute an old preview.
            try:
                if not self.scene_ready:
                    raise ValueError('Configured collision environment is not yet applied; wait for /arm_lab/scene_status ready')
                if not pose.header.frame_id:
                    raise ValueError('target header.frame_id is required')
                p, q = pose.pose.position, pose.pose.orientation
                validate_target([p.x, p.y, p.z], None if position_only else [q.x, q.y, q.z, q.w])
                frame = self.group['base_link']
                if pose.header.frame_id != frame:
                    pose = self.tf_buffer.transform(pose, frame)
                p, q = pose.pose.position, pose.pose.orientation
                xyz, xyzw = validate_target([p.x, p.y, p.z],
                                           None if position_only else [q.x, q.y, q.z, q.w])
                goal = build_goal(self.group_name, self.group, self.options, frame, xyz, xyzw)
                if not self.plan_client.server_is_ready():
                    raise ValueError('MoveGroup action server is not ready')
                target = {'frame_id': frame, 'position': list(xyz),
                          'orientation_xyzw': list(xyzw) if xyzw is not None else None}
                self.session = PlanSession(state='planning', target=target)
                self.plan_client.send_goal_async(goal).add_done_callback(self.plan_accepted)
                self.report('planning', 'Planning target', target=target)
            except Exception as exc:  # ROS TF/action exceptions are reported at the boundary.
                self.session = PlanSession()
                self.report('planning_failed', str(exc))

        def plan_accepted(self, future):
            try:
                handle = future.result()
                if not handle.accepted:
                    raise ValueError('MoveGroup rejected target')
                handle.get_result_async().add_done_callback(self.plan_finished)
            except Exception as exc:
                self.session = PlanSession()
                self.report('planning_failed', str(exc))

        def plan_finished(self, future):
            try:
                response = future.result()
                result = response.result
                if response.status != GoalStatus.STATUS_SUCCEEDED or result.error_code.val != MoveItErrorCodes.SUCCESS:
                    raise ValueError(f'MoveGroup planning error {result.error_code.val}')
                if not result.planned_trajectory.joint_trajectory.points:
                    raise ValueError('MoveGroup returned an empty trajectory')
                self.session = self.session.planned(result.planned_trajectory, result.trajectory_start,
                                                    self.session.target)
                self.display.publish(DisplayTrajectory(model_id=self.robot['name'],
                                     trajectory_start=result.trajectory_start,
                                     trajectory=[result.planned_trajectory]))
                self.report('preview', 'Plan ready; call /arm_lab/execute_plan to execute')
            except Exception as exc:
                self.session = PlanSession()
                self.report('planning_failed', str(exc))

        def execute(self, request, response):
            if not self.scene_ready or not self.session.can_execute or not self.execute_client.server_is_ready():
                response.success = False
                response.message = 'Scene not ready, no unexecuted preview, or ExecuteTrajectory server unavailable'
                return response
            goal = ExecuteTrajectory.Goal(trajectory=self.session.trajectory)
            self.benchmark_result = None
            self.session = self.session.executing()
            try:
                self.execute_client.send_goal_async(goal).add_done_callback(self.execute_accepted)
            except Exception as exc:
                self.session = self.session.finished(False)
                response.success, response.message = False, str(exc)
                self.report('execution_failed', str(exc))
                return response
            response.success = True
            response.message = 'Execution requested; inspect /arm_lab/planning_status for final result'
            self.report('executing', response.message)
            return response

        def execute_accepted(self, future):
            try:
                handle = future.result()
                if not handle.accepted:
                    raise ValueError('ExecuteTrajectory rejected plan')
                handle.get_result_async().add_done_callback(self.execute_finished)
            except Exception as exc:
                self.session = self.session.finished(False)
                self.report('execution_failed', str(exc))

        def execute_finished(self, future):
            try:
                response = future.result()
                code = response.result.error_code.val
                success = response.status == GoalStatus.STATUS_SUCCEEDED and code == MoveItErrorCodes.SUCCESS
                self.session = self.session.finished(success)
                self.report(self.session.state, f'Execution MoveIt result {code}', error_code=code)
            except Exception as exc:
                self.session = self.session.finished(False)
                self.report('execution_failed', str(exc))

        def save(self, request, response):
            if not self.session.can_save:
                response.success = False
                response.message = 'Only confirmed successful executions can be saved'
                return response
            try:
                path = save_session(self.session, self.robot, self.options, self.project,
                                    robot_fingerprint(self.robot), self.benchmark_result,
                                    scenario_id=self.get_parameter('scenario_id').value)
                response.success, response.message = True, str(path)
                self.report('saved', str(path))
            except Exception as exc:
                response.success, response.message = False, str(exc)
                self.report('save_failed', str(exc))
            return response

    return TargetNode()


def save_session(session, robot, options, project, fingerprint, benchmark=None, *, scenario_id='ros_execution'):
    """Adapter to the shared, validated trajectory storage contract."""
    from datetime import datetime, timezone
    from pathlib import Path
    import numpy as np
    from arm_lab_model.physical_robot import plain
    from arm_lab_model.trajectory_store import save_trajectory

    if not session.can_save:
        raise ValueError('only successful executions can be saved')
    if robot.get('base', {}).get('type') == 'floating':
        raise ValueError('Saving floating-base MoveIt executions requires captured base pose and velocity; currently unsupported')
    trajectory = session.trajectory.joint_trajectory
    names = list(trajectory.joint_names)
    points = tuple(trajectory.points)
    if len(points) < 2:
        raise ValueError('trajectory needs at least two points to save')
    times = np.asarray([point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                        for point in points])
    positions = np.asarray([list(point.positions) for point in points])
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('saved trajectory must start at t=0 with strictly increasing timing')
    velocity_complete = all(len(point.velocities) == len(names) for point in points)
    acceleration_complete = all(len(point.accelerations) == len(names) for point in points)
    velocities = np.asarray([list(point.velocities) for point in points]) if velocity_complete else np.gradient(positions, times, axis=0)
    accelerations = np.asarray([list(point.accelerations) for point in points]) if acceleration_complete else np.gradient(velocities, times, axis=0)
    target = {'frame': session.target['frame_id'], 'position': session.target['position']}
    if session.target.get('orientation_xyzw') is not None:
        target = {**target, 'orientation': session.target['orientation_xyzw']}
    record = {
        'schema_version': 1, 'robot': robot['name'], 'model_sha256': fingerprint,
        'scenario_id': scenario_id, 'timestamp': datetime.now(timezone.utc).isoformat(),
        'joint_names': names, 'start_state': positions[0].tolist(), 'target': target,
        'trajectory': {'points': [
            {'time_from_start': float(t), 'positions': p.tolist(), 'velocities': v.tolist(),
             'accelerations': a.tolist()} for t, p, v, a in zip(times, positions, velocities, accelerations)]},
        'planner': {'name': options.get('planner_id', 'RRTConnectkConfigDefault'), 'parameters': {
            **plain(options), 'velocity_source': 'planner' if velocity_complete else 'numerical_gradient',
            'acceleration_source': 'planner' if acceleration_complete else 'numerical_gradient'}},
        'collision': {'status': 'clear', 'details': 'MoveIt planning-scene collision checked; execution was successful'},
        'execution': {'status': 'succeeded', 'details': 'ExecuteTrajectory action and MoveIt error code both SUCCESS'},
        'benchmark': benchmark,
    }
    trajectories = project.sections.get('trajectories', {})
    root = options.get('saved_trajectories_dir', trajectories.get('directory', 'saved_trajectories'))
    component = 'moveit' if 'saved_trajectories_dir' in options else 'trajectories'
    declared_in = getattr(project, 'files', {}).get(component, project.source_path)
    directory = Path(root)
    if not directory.is_absolute():
        directory = Path(declared_in).parent / directory
    return save_trajectory(record, directory)


def quaternion_from_rpy(rpy):
    """Fixed-axis roll, pitch, yaw in radians -> xyzw quaternion."""
    (roll, pitch, yaw), _ = validate_target(rpy)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def publish_target(args):
    """Publish either xyz or xyz/rpy without forcing a position target orientation."""
    import argparse
    import time
    import rclpy
    from geometry_msgs.msg import PointStamped, PoseStamped
    parser = argparse.ArgumentParser(description='Plan an xyz or xyz/rpy target; execution is separate')
    parser.add_argument('--xyz', nargs=3, type=float, required=True)
    parser.add_argument('--rpy', nargs=3, type=float)
    parser.add_argument('--frame', required=True)
    options = parser.parse_args(args)
    xyz, _ = validate_target(options.xyz)
    if not options.frame.strip():
        raise ValueError('--frame cannot be empty')
    rclpy.init(args=[])
    node = rclpy.create_node('pipeline_target_command')
    try:
        if options.rpy is None:
            message = PointStamped()
            message.point.x, message.point.y, message.point.z = xyz
            publisher = node.create_publisher(PointStamped, '/clicked_point', 10)
        else:
            message = PoseStamped()
            message.pose.position.x, message.pose.position.y, message.pose.position.z = xyz
            q = message.pose.orientation
            q.x, q.y, q.z, q.w = quaternion_from_rpy(options.rpy)
            publisher = node.create_publisher(PoseStamped, '/arm_lab/target', 10)
        message.header.frame_id = options.frame
        deadline = time.monotonic() + 5.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if publisher.get_subscription_count() == 0:
            raise RuntimeError('No target subscriber found; launch pipeline first')
        publisher.publish(message)  # zero stamp requests the latest available TF
        rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def cli_main():
    import sys
    return publish_target(sys.argv[1:])


def main(args=None):
    import sys
    import rclpy
    args = sys.argv[1:] if args is None else args
    if args and args[0] == 'target':
        return publish_target(args[1:])
    rclpy.init(args=args)
    node = None
    try:
        node = create_node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
