"""Publish a saved workspace map as RViz markers.

    ros2 run arm_lab_kinematics workspace --save /tmp/ws.npz
    ros2 run arm_lab_kinematics workspace_markers --ros-args -p map_file:=/tmp/ws.npz

The map is built offline because the orientation sweep takes a minute; the node
just revolves the stored cross-section about the base axis and draws it.
"""

from __future__ import annotations

import math
import sys
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class WorkspaceMarkerNode(Node):
    def __init__(self):
        super().__init__('workspace_markers')
        self.declare_parameter('map_file', '')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('revolutions', 24)
        self.declare_parameter('publish_period', 2.0)

        path = self.get_parameter('map_file').value
        if not path:
            self.get_logger().error(
                'set map_file to an .npz written by '
                '`ros2 run arm_lab_kinematics workspace --save <file>`')
            raise SystemExit(2)
        data = np.load(path)
        self.r_edges = data['r_edges']
        self.z_edges = data['z_edges']
        self.layers = [
            ('reachable', data['reachable'], ColorRGBA(r=0.35, g=0.55, b=0.85, a=0.10)),
            ('tool_down', data['tool_down'], ColorRGBA(r=0.95, g=0.70, b=0.20, a=0.22)),
            ('dexterous', data['dexterity'] >= 0.999,
             ColorRGBA(r=0.25, g=0.80, b=0.40, a=0.45)),
        ]
        self.sweep = float(data['sweep_angle'])
        self.pub = self.create_publisher(MarkerArray, '/arm_lab/workspace', 1)
        period = float(self.get_parameter('publish_period').value)
        self.create_timer(period, self._publish)
        self.get_logger().info(f'workspace markers from {path}')

    def _publish(self) -> None:
        frame = self.get_parameter('frame_id').value
        turns = int(self.get_parameter('revolutions').value)
        dr = float(self.r_edges[1] - self.r_edges[0])
        dz = float(self.z_edges[1] - self.z_edges[0])
        r_c = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        z_c = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        angles = np.linspace(-self.sweep / 2.0, self.sweep / 2.0, turns)

        array = MarkerArray()
        for index, (name, mask, colour) in enumerate(self.layers):
            marker = Marker()
            marker.header.frame_id = frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = name
            marker.id = index
            marker.type = Marker.CUBE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = dr
            marker.scale.y = dr
            marker.scale.z = dz
            marker.color = colour
            for i, j in np.argwhere(mask):
                for angle in angles:
                    marker.points.append(Point(
                        x=float(r_c[i] * math.cos(angle)),
                        y=float(r_c[i] * math.sin(angle)),
                        z=float(z_c[j])))
            array.markers.append(marker)
        self.pub.publish(array)


def main(argv: Optional[List[str]] = None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    try:
        node = WorkspaceMarkerNode()
    except SystemExit as exc:
        rclpy.shutdown()
        return int(exc.code or 1)
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
