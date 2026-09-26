"""Calibrated camera -> point cloud and opt-in perception plugin ROS adapter."""
import json
import time
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformListener, TransformException

from arm_lab_model.project_config import load_project
from arm_lab_model.perception import (depth_to_pointcloud, filter_pointcloud,
                                      load_algorithm, validate_perception_dependencies)
from arm_lab_model.scene_config import mutable


def image_array(message):
    """Respect ROS Image encoding, byte order and padded row stride."""
    encodings = {'rgb8': (np.dtype('u1'), 3), '32FC1': (np.dtype('f4'), 1), '16UC1': (np.dtype('u2'), 1)}
    if message.encoding not in encodings:
        raise ValueError('unsupported image encoding ' + message.encoding)
    dtype, channels = encodings[message.encoding]
    dtype = dtype.newbyteorder('>' if message.is_bigendian else '<')
    expected = message.width * channels * dtype.itemsize
    if message.width <= 0 or message.height <= 0 or message.step < expected or len(message.data) < message.height * message.step:
        raise ValueError('invalid Image dimensions, stride or buffer length')
    array = np.ndarray((message.height, message.width, channels), dtype=dtype,
                       buffer=bytes(message.data), strides=(message.step, channels*dtype.itemsize, dtype.itemsize)).copy()
    return array[:, :, 0] if channels == 1 else array


def transform_matrix(transform):
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = np.linalg.norm([x, y, z, w])
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('invalid TF quaternion')
    x, y, z, w = np.array([x, y, z, w]) / norm
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
    return result


class PipelinePerception(Node):
    def __init__(self):
        super().__init__('pipeline_perception')
        project = load_project(self.declare_parameter('project_file', '').value)
        validate_perception_dependencies(project)
        self.config = mutable(project.sections.get('perception', {}))
        if not self.config.get('enabled'):
            raise ValueError('perception is disabled in this project')
        self.intrinsics = {}
        self.pending_clouds = deque(maxlen=10)
        self.create_timer(.02, self.flush_clouds)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cloud_publisher = self.create_publisher(PointCloud2, '/perception/points', qos_profile_sensor_data)
        self.octomap_publisher = self.create_publisher(PointCloud2, '/perception/octomap_points', qos_profile_sensor_data)
        self.results = self.create_publisher(String, '/perception/results', 10)
        self.algorithms = {name: load_algorithm(spec['plugin']) for name, spec in self.config.items()
                           if isinstance(spec, dict) and spec.get('enabled') and 'plugin' in spec}
        names = {spec['camera'] for spec in self.config.values()
                 if isinstance(spec, dict) and spec.get('enabled') and 'camera' in spec}
        for name in names:
            self.create_subscription(CameraInfo, f'/sensors/{name}/camera_info',
                                     lambda msg, n=name: self.on_info(n, msg), qos_profile_sensor_data)
            for stream in ('rgb', 'depth'):
                self.create_subscription(Image, f'/sensors/{name}/{stream}/image_raw',
                                         lambda msg, n=name, s=stream: self.on_image(n, s, msg), qos_profile_sensor_data)

    def on_info(self, name, message):
        self.intrinsics[name] = {'fx': float(message.k[0]), 'fy': float(message.k[4]),
                                 'cx': float(message.k[2]), 'cy': float(message.k[5]),
                                 'width': message.width, 'height': message.height,
                                 'frame_id': message.header.frame_id}

    def on_image(self, name, stream, message):
        try:
            image = image_array(message)
            intrinsics = self.intrinsics.get(name)
            if intrinsics is None:
                return
            if (intrinsics['width'], intrinsics['height'], intrinsics['frame_id']) != (message.width, message.height, message.header.frame_id):
                raise ValueError('Image and CameraInfo frame/resolution mismatch')
            cloud_config = self.config.get('pointcloud', {})
            if stream == 'depth' and cloud_config.get('enabled') and cloud_config['camera'] == name:
                self.pending_clouds.append((time.monotonic(), message, image, intrinsics, cloud_config))
            bundle = {'stamp': message.header.stamp.sec + message.header.stamp.nanosec*1e-9,
                      'frame_id': message.header.frame_id, stream: image, 'intrinsics': intrinsics}
            for module, algorithm in self.algorithms.items():
                spec = self.config[module]
                required_stream = 'rgb' if module in ('object_recognition', 'image_policy') else 'depth'
                if spec['camera'] == name and stream == required_stream:
                    result = algorithm(bundle, spec.get('parameters', {}))
                    self.results.publish(String(data=json.dumps({'module': module, 'result': result}, allow_nan=False)))
        except (ValueError, TypeError, RuntimeError, TransformException) as exc:
            self.get_logger().error(f'perception frame rejected: {exc}', throttle_duration_sec=2.)

    def flush_clouds(self):
        pending, self.pending_clouds = self.pending_clouds, deque(maxlen=10)
        for queued, message, image, intrinsics, spec in pending:
            stamp = rclpy.time.Time.from_msg(message.header.stamp)
            if spec['target_frame'] != message.header.frame_id and not self.tf_buffer.can_transform(
                    spec['target_frame'], message.header.frame_id, stamp):
                if time.monotonic() - queued < 1.:
                    self.pending_clouds.append((queued, message, image, intrinsics, spec))
                else:
                    self.get_logger().warning('Depth frame dropped: matching timestamp TF unavailable for 1 second', throttle_duration_sec=2.)
                continue
            try:
                self.publish_cloud(message, image, intrinsics, spec)
            except (ValueError, TypeError, RuntimeError, TransformException) as exc:
                self.get_logger().error(f'point cloud rejected: {exc}', throttle_duration_sec=2.)

    def publish_cloud(self, message, image, intrinsics, spec):
        cloud = depth_to_pointcloud(image, intrinsics, depth_scale=.001 if message.encoding == '16UC1' else 1.,
                                    min_depth=spec['min_depth'], max_depth=spec['max_depth'])
        if self.config.get('octomap', {}).get('enabled'):
            sensor_cloud = filter_pointcloud(cloud, voxel_size=spec['voxel_size'])
            self.octomap_publisher.publish(create_cloud_xyz32(message.header, sensor_cloud.tolist()))
        matrix = None
        if spec['target_frame'] != message.header.frame_id:
            transform = self.tf_buffer.lookup_transform(spec['target_frame'], message.header.frame_id,
                                                         rclpy.time.Time.from_msg(message.header.stamp))
            matrix = transform_matrix(transform.transform)
        cloud = filter_pointcloud(cloud, transform=matrix, voxel_size=spec['voxel_size'])
        header = Header(stamp=message.header.stamp, frame_id=spec['target_frame'])
        self.cloud_publisher.publish(create_cloud_xyz32(header, cloud.tolist()))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PipelinePerception()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
