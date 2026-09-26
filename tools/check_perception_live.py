#!/usr/bin/env python3
"""Verify an already running UR5e perception example over actual ROS topics/services."""
import argparse
import json
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from arm_lab_gui.pipeline_perception import image_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=30.)
    args = parser.parse_args()
    rclpy.init()
    node = Node('check_perception_live')
    received = {}
    camera = '/sensors/wrist_camera'
    subscriptions = []
    for key, msg_type, topic in (
            ('rgb', Image, camera+'/rgb/image_raw'), ('depth', Image, camera+'/depth/image_raw'),
            ('calibration', CameraInfo, camera+'/camera_info'),
            ('world_cloud', PointCloud2, '/perception/points'),
            ('optical_cloud', PointCloud2, '/perception/octomap_points')):
        subscriptions.append(node.create_subscription(msg_type, topic, lambda msg, k=key: received.update({k: msg}), qos_profile_sensor_data))
    client = node.create_client(GetPlanningScene, '/get_planning_scene')
    future, scene = None, None
    deadline = time.monotonic()+args.timeout
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if len(received) == 5 and client.service_is_ready() and future is None:
                request = GetPlanningScene.Request()
                request.components.components = PlanningSceneComponents.OCTOMAP | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
                future = client.call_async(request)
            if future is not None and future.done():
                scene = future.result().scene
                if scene.world.octomap.octomap.data:
                    break
                future = None
        missing = sorted({'rgb','depth','calibration','world_cloud','optical_cloud'}-received.keys())
        if missing:
            raise RuntimeError('missing live messages: '+', '.join(missing))
        depth = image_array(received['depth'])
        valid = int(np.isfinite(depth).sum())
        assert valid > 0, 'rendered depth contains no visible geometry'
        assert received['world_cloud'].header.frame_id == 'world'
        assert received['optical_cloud'].header.frame_id == 'wrist_camera_optical_frame'
        assert received['world_cloud'].width > 0
        assert received['optical_cloud'].width > 0
        assert received['calibration'].k[0] == 280.
        assert image_array(received['rgb']).shape == (240,320,3)
        assert scene is not None and len(scene.world.octomap.octomap.data) > 0, 'MoveIt OctoMap has not received occupied/free cells'
        assert {'workbench', 'fixture'} <= {obj.id for obj in scene.world.collision_objects}
        print(json.dumps({'passed': True, 'valid_depth_pixels': valid,
                          'world_cloud_points': received['world_cloud'].width,
                          'optical_cloud_points': received['optical_cloud'].width,
                          'octomap_bytes': len(scene.world.octomap.octomap.data),
                          'collision_objects': [obj.id for obj in scene.world.collision_objects]}, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
