"""ROS trajectory-action bridge for the unified MuJoCo robot and calibrated cameras."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from builtin_interfaces.msg import Time
from control_msgs.action import FollowJointTrajectory
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from arm_lab_model.physical_robot import plain, resolve_robot
from arm_lab_model.pipeline_runtime import RobotSimulation
from arm_lab_model.project_config import load_project
from arm_lab_model.scene_config import resolve_environment, resolve_sensors
from .trajectory_tolerance import tolerance_policy, tracking_violation, validate_terminal_state


def stamp(value):
    nanoseconds = round(value * 1e9)
    return Time(sec=nanoseconds // 10**9, nanosec=nanoseconds % 10**9)


def points_from_message(trajectory):
    return [{'time_from_start': p.time_from_start.sec + p.time_from_start.nanosec*1e-9,
             'positions': list(p.positions), 'velocities': list(p.velocities),
             'accelerations': list(p.accelerations)} for p in trajectory.points]


def trajectory_digest(message):
    payload = {'joint_names': list(message.joint_names), 'points': points_from_message(message)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def configured_controllers(moveit, names):
    """Use the generic joint action when planning is absent or explicitly disabled."""
    if not moveit.get('enabled', False):
        return {'arm_controller': tuple(names)}
    from arm_lab_kinematics.pipeline_moveit import controller_joints
    return controller_joints(moveit)


class ExecutionViolation(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class PipelineSimulationNode(Node):
    def __init__(self):
        super().__init__('pipeline_sim')
        for name, default in (('project_file', ''), ('benchmark', False), ('benchmark_robot', ''),
                              ('benchmark_reference', ''), ('output_dir', 'benchmark_reports'),
                              ('headless_render', True), ('scenario_id', 'ros_execution'),
                              ('goal_position_tolerance', 0.02), ('goal_velocity_tolerance', 0.01),
                              ('goal_time_tolerance', 1.0)):
            self.declare_parameter(name, default)
        if self.param('headless_render'):
            os.environ.setdefault('MUJOCO_GL', 'egl')
        self.project = load_project(self.param('project_file'))
        self.robot = resolve_robot(self.project)
        self.cameras = resolve_sensors(self.project)
        self.sim = RobotSimulation(self.robot, plain(self.project.sections.get('simulation', {})),
                                   resolve_environment(self.project), self.cameras)
        self.benchmark_reference_data = None
        if self.param('benchmark') or self.project.sections.get('benchmark', {}).get('enabled'):
            from arm_lab_model.pipeline_cli import prepare_benchmark
            self.benchmark_reference_data = prepare_benchmark(
                self.project, reference=self.param('benchmark_reference') or None,
                robot=self.param('benchmark_robot') or None, expected={
                    'scenario_id': self.param('scenario_id'), 'frame': 'world', 'units': 'SI',
                    'joint_types': [joint['type'] for joint in self.sim.joints],
                    'end_effector': self.sim.endpoint()})
        initial = self.project.sections.get('simulation', {}).get('initial_positions')
        if initial is not None:
            self.sim.set_state(initial)
        self.lock = threading.RLock()
        self.busy = False
        self.rows = []
        self.saturation = 0
        self.contacts = 0
        self.renderer = None
        self.execution_policy = None
        self.execution_indices = ()
        self.execution_end = 0.0
        self.execution_fault = None
        self.state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.base_tf = TransformBroadcaster(self)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.benchmark_pub = self.create_publisher(String, '/arm_lab/benchmark_result', 10)
        self.camera_publishers = {camera['name']: self._camera_pubs(camera) for camera in self.cameras}
        self.camera_due = {camera['name']: 0.0 for camera in self.cameras}
        options = self.project.sections.get('moveit', {})
        self.accelerations = plain(options.get('acceleration_limits', {}))
        controllers = configured_controllers(options, self.sim.names)
        self.actions = [ActionServer(self, FollowJointTrajectory, f'/{name}/follow_joint_trajectory',
                                    execute_callback=self.execute,
                                    goal_callback=lambda goal, allowed=tuple(joints): self.accept_goal(goal, allowed),
                                    cancel_callback=lambda _: CancelResponse.ACCEPT,
                                    callback_group=ReentrantCallbackGroup()) for name, joints in controllers.items()]

    def param(self, name):
        return self.get_parameter(name).value

    def _camera_pubs(self, camera):
        base = '/sensors/' + camera['name']
        return {key: self.create_publisher(msg, base + suffix, 5) for key, msg, suffix in (
            ('rgb', Image, '/rgb/image_raw'), ('depth', Image, '/depth/image_raw'),
            ('info', CameraInfo, '/camera_info'))}

    def request_policy(self, request):
        if request.component_path_tolerance or request.component_goal_tolerance:
            raise ValueError('multi-DOF component tolerances are unsupported')
        def entries(values):
            return [{'name': value.name, 'position': value.position, 'velocity': value.velocity,
                     'acceleration': value.acceleration} for value in values]
        for point in request.trajectory.points:
            if point.effort:
                raise ValueError('effort trajectory commands are unsupported')
        validate_terminal_state({'velocities': list(request.trajectory.points[-1].velocities)})
        duration = request.goal_time_tolerance
        if not 0 <= duration.nanosec < 10**9:
            raise ValueError('goal_time_tolerance nanoseconds out of range')
        return tolerance_policy(request.trajectory.joint_names,
            path=entries(request.path_tolerance), goal=entries(request.goal_tolerance),
            goal_time=duration.sec + duration.nanosec * 1e-9,
            path_position=float(self.sim.settings.get('max_error', .05)),
            goal_position=self.param('goal_position_tolerance'),
            goal_velocity=self.param('goal_velocity_tolerance'),
            default_goal_time=self.param('goal_time_tolerance'))

    def accept_goal(self, goal, allowed=None):
        with self.lock:
            if self.busy or not goal.trajectory.points or goal.multi_dof_trajectory.points:
                return GoalResponse.REJECT
            if not set(goal.trajectory.joint_names) <= set(allowed if allowed is not None else self.sim.names):
                return GoalResponse.REJECT
            try:
                self.request_policy(goal)
            except ValueError as exc:
                self.get_logger().error('Rejected trajectory: ' + str(exc))
                return GoalResponse.REJECT
            self.busy = True
        return GoalResponse.ACCEPT

    def execute(self, goal):
        result = FollowJointTrajectory.Result()
        message = goal.request.trajectory
        points = [{key: value for key, value in point.items() if value != []}
                  for point in points_from_message(message)]
        try:
            policy = self.request_policy(goal.request)
            with self.lock:
                if message.header.stamp.sec or message.header.stamp.nanosec:
                    requested = message.header.stamp.sec + message.header.stamp.nanosec*1e-9
                    if abs(requested-self.sim.time) > 0.05:
                        raise ValueError('only immediate trajectories are supported; use zero header stamp')
                path = self.sim.command(message.joint_names, points, self.accelerations)
                start, end = self.sim.time, self.sim.time + path.duration
                self.rows, self.saturation, self.contacts = [self.sim.sample()], 0, 0
                indices = [self.sim.names.index(n) for n in message.joint_names]
                self.execution_policy, self.execution_indices = policy, tuple(indices)
                self.execution_end, self.execution_fault = end, None
            while rclpy.ok():
                if goal.is_cancel_requested:
                    with self.lock:
                        self.sim.hold()
                    goal.canceled()
                    result.error_string = 'Canceled; holding current joint positions.'
                    return result
                with self.lock:
                    now = self.sim.time
                    q, v, a = self.sim.desired()
                    actual = self.sim.sample()
                    error = q - np.asarray(actual['joint_position'])
                    fault = self.execution_fault
                feedback = FollowJointTrajectory.Feedback()
                feedback.header.stamp = stamp(now)
                feedback.joint_names = list(message.joint_names)
                desired = (q[indices].tolist(), v[indices].tolist(), a[indices].tolist())
                measured = tuple([actual[key][i] for i in indices]
                                 for key in ('joint_position', 'joint_velocity', 'joint_acceleration'))
                feedback.desired = JointTrajectoryPoint(positions=desired[0], velocities=desired[1], accelerations=desired[2])
                feedback.actual = JointTrajectoryPoint(positions=measured[0], velocities=measured[1], accelerations=measured[2])
                feedback.error = JointTrajectoryPoint(positions=error[indices].tolist(),
                    velocities=(np.asarray(desired[1])-measured[1]).tolist(),
                    accelerations=(np.asarray(desired[2])-measured[2]).tolist())
                goal.publish_feedback(feedback)
                if fault is not None:
                    raise ExecutionViolation(*fault)
                violation = tracking_violation(policy, desired, measured, final=True)
                if now >= end and violation is None:
                    with self.lock:
                        if self.execution_fault is not None:
                            raise ExecutionViolation(*self.execution_fault)
                        rows = list(self.rows)
                        self.rows = []
                        self.execution_policy = None
                    self._finish_report(message, start, rows)
                    goal.succeed()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    return result
                if now > end + policy.goal_time:
                    raise ExecutionViolation(FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                                             'Goal settling deadline exceeded: ' + str(violation))
                time.sleep(0.02)
            raise RuntimeError('ROS shutdown interrupted execution')
        except (ValueError, RuntimeError, OSError) as exc:
            with self.lock:
                self.sim.hold()
            goal.abort()
            result.error_code = (exc.code if isinstance(exc, ExecutionViolation) else
                                 FollowJointTrajectory.Result.INVALID_GOAL if isinstance(exc, ValueError) else
                                 FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED)
            result.error_string = str(exc)
            return result
        finally:
            with self.lock:
                self.busy = False
                self.rows = []
                self.execution_policy = None
                self.execution_fault = None

    def _finish_report(self, message, start, rows=None):
        if rows is None:
            with self.lock:
                rows = list(self.rows)
        samples = {key: [row[key] for row in rows] for key in rows[0]}
        samples['time'] = [t-start for t in samples['time']]
        observation = {'schema_version': 1, 'robot': self.robot['name'],
                       'model_sha256': self.sim.model_hash, 'scenario_id': self.param('scenario_id'),
                       'joint_names': list(self.sim.names), 'frame': 'world', 'units': 'SI',
                       'end_effector': self.sim.endpoint(),
                       'joint_types': [j['type'] for j in self.sim.joints],
                       'evidence': {'kind': 'synthetic', 'source': 'MuJoCo ROS joint-action execution'},
                       'samples': samples}
        digest = trajectory_digest(message)
        folder = Path(self.param('output_dir')) / (digest[:16] + '-' + str(time.time_ns()))
        folder.mkdir(parents=True, exist_ok=False)
        (folder / 'observation.json').write_text(json.dumps(observation, allow_nan=False))
        if self.param('benchmark') or self.project.sections.get('benchmark', {}).get('enabled'):
            from arm_lab_model.pipeline_cli import benchmark_execution
            result = benchmark_execution(self.project, observation, folder,
                                         reference=self.param('benchmark_reference') or None,
                                         robot=self.param('benchmark_robot') or None,
                                         reference_data=self.benchmark_reference_data)
            self.benchmark_pub.publish(String(data=json.dumps({'trajectory_sha256': digest, 'result': result})))

    def publish(self):
        with self.lock:
            sample = self.sim.sample()
        when = stamp(sample['time'])
        self.clock_pub.publish(Clock(clock=when))
        state = JointState(name=list(self.sim.names), position=sample['joint_position'],
                           velocity=sample['joint_velocity'], effort=sample['joint_torque'])
        state.header.stamp = when
        self.state_pub.publish(state)
        if self.robot['base']['type'] == 'floating':
            with self.lock:
                body = self.sim.m.body(self.robot['base']['link']).id
                pos, quat = self.sim.d.xpos[body].copy(), self.sim.d.xquat[body].copy()
            transform = TransformStamped()
            transform.header.stamp, transform.header.frame_id = when, 'world'
            transform.child_frame_id = self.robot['base']['link']
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, pos)
            q = transform.transform.rotation
            q.w, q.x, q.y, q.z = map(float, quat)
            self.base_tf.sendTransform(transform)

    def publish_cameras(self):
        from arm_lab_model.sensor_runtime import CameraRenderer
        if self.cameras and self.renderer is None:
            self.renderer = CameraRenderer(self.sim.m, self.cameras,
                                           self.project.sections.get('simulation', {}).get('seed', 0))
        for camera in self.cameras:
            name = camera['name']
            if self.sim.time < self.camera_due[name]:
                continue
            with self.lock:
                frame = self.renderer.render(self.sim.d, name)
            pubs = self.camera_publishers[name]
            for key, encoding, dtype in (('rgb', 'rgb8', np.uint8), ('depth', '32FC1', np.float32)):
                if key not in frame:
                    continue
                pixels = np.ascontiguousarray(frame[key], dtype=dtype)
                msg = Image(height=pixels.shape[0], width=pixels.shape[1], encoding=encoding,
                            is_bigendian=0, step=pixels.strides[0], data=pixels.tobytes())
                msg.header.stamp, msg.header.frame_id = stamp(frame['stamp']), frame['frame_id']
                pubs[key].publish(msg)
            info = CameraInfo(width=camera['resolution'][0], height=camera['resolution'][1],
                              distortion_model='plumb_bob', d=[0.0]*5)
            k = frame['intrinsics']
            info.k = [float(k['fx']),0.,float(k['cx']),0.,float(k['fy']),float(k['cy']),0.,0.,1.]
            info.r = [1.,0.,0.,0.,1.,0.,0.,0.,1.]
            info.p = [float(k['fx']),0.,float(k['cx']),0.,0.,float(k['fy']),float(k['cy']),0.,0.,0.,1.,0.]
            info.header.stamp, info.header.frame_id = stamp(frame['stamp']), frame['frame_id']
            pubs['info'].publish(info)
            self.camera_due[name] = self.sim.time + 1/camera['frame_rate_hz']

    def check_execution_sample(self):
        """Called under the simulation lock at every physics step, not feedback rate."""
        if self.execution_policy is None or self.execution_fault is not None:
            return
        violation = None
        if self.saturation:
            violation = 'Actuator effort saturation during trajectory execution'
        elif self.contacts:
            violation = 'Contact detected during trajectory execution'
        elif self.sim.time < self.execution_end:
            indices = list(self.execution_indices)
            desired = tuple(value[indices].tolist() for value in self.sim.desired())
            sample = self.rows[-1]
            actual = tuple([sample[key][i] for i in indices]
                           for key in ('joint_position', 'joint_velocity', 'joint_acceleration'))
            violation = tracking_violation(self.execution_policy, desired, actual, final=False)
        if violation is not None:
            self.execution_fault = (FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED, violation)
            self.sim.hold()

    def run(self):
        next_publish = 0.
        wall = time.monotonic()
        sim_start = self.sim.time
        try:
            while rclpy.ok():
                target = sim_start + time.monotonic()-wall
                for _ in range(50):
                    with self.lock:
                        if self.sim.time >= target:
                            break
                        self.sim.step()
                        if self.busy and self.rows:
                            self.rows.append(self.sim.sample())
                            self.saturation += int(self.sim.saturated)
                            self.contacts = max(self.contacts, self.sim.d.ncon)
                            self.check_execution_sample()
                if self.sim.time >= next_publish:
                    self.publish()
                    self.publish_cameras()
                    next_publish = self.sim.time + .01
                time.sleep(.001)
        finally:
            if self.renderer:
                self.renderer.close()


def main(argv=None):
    rclpy.init(args=argv)
    node = PipelineSimulationNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(timeout_sec=2)
        thread.join(timeout=2)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
