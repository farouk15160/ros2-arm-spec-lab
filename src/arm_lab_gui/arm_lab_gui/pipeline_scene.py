"""Apply the configured world to MoveIt and publish matching RViz geometry."""
from pathlib import Path
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Point, Pose
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from arm_lab_model.project_config import load_project
from arm_lab_model.scene_config import resolve_environment, collision_objects


def _pose(record):
    result = Pose()
    result.position.x, result.position.y, result.position.z = map(float, record['position'])
    result.orientation.x, result.orientation.y, result.orientation.z, result.orientation.w = map(float, record['orientation_xyzw'])
    return result


def collision_message(record):
    from moveit_msgs.msg import CollisionObject
    result = CollisionObject()
    result.id = record['id']
    result.header.frame_id = record['frame_id']
    result.operation = CollisionObject.ADD
    if 'primitive' in record:
        primitive = SolidPrimitive()
        primitive.type = {'box': SolidPrimitive.BOX, 'sphere': SolidPrimitive.SPHERE,
                          'cylinder': SolidPrimitive.CYLINDER}[record['primitive']['type']]
        primitive.dimensions = [float(v) for v in record['primitive']['dimensions']]
        result.primitives = [primitive]
        result.primitive_poses = [_pose(record['pose'])]
    else:
        mesh = Mesh()
        mesh.vertices = [Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in record['mesh']['vertices']]
        mesh.triangles = [MeshTriangle(vertex_indices=t) for t in record['mesh']['triangles']]
        result.meshes = [mesh]
        result.mesh_poses = [_pose(record['pose'])]
    return result


def environment_marker(obj, record, index):
    marker = Marker()
    marker.header.frame_id = obj['frame']
    marker.ns, marker.id, marker.action = 'pipeline_environment', index, Marker.ADD
    marker.pose = _pose(record['pose'])
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = .55, .6, .7, 1.
    geo = obj['geometry']
    if geo['type'] == 'mesh':
        from arm_lab_model.mesh_physics import mesh_scale
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = Path(geo['path']).as_uri()
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, mesh_scale(geo))
    else:
        marker.type = {'box': Marker.CUBE, 'sphere': Marker.SPHERE, 'cylinder': Marker.CYLINDER}[geo['type']]
        dimensions = geo['size'] if geo['type'] == 'box' else [2*geo['radius'], 2*geo['radius'],
                     2*geo['radius'] if geo['type'] == 'sphere' else geo['length']]
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, dimensions)
    return marker


class PipelineScene(Node):
    def __init__(self):
        super().__init__('pipeline_scene')
        project = load_project(self.declare_parameter('project_file', '').value)
        apply_moveit = self.declare_parameter('apply_moveit',
            project.sections.get('moveit', {}).get('enabled', False)).value
        objects = resolve_environment(project)
        records = collision_objects(objects)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status = self.create_publisher(String, '/arm_lab/scene_status', qos)
        self.status.publish(String(data=json.dumps({'ready': False, 'planning_scene_applied': False})))
        self.markers = self.create_publisher(MarkerArray, '/pipeline/environment_markers', qos)
        self.markers.publish(MarkerArray(markers=[environment_marker(o, r, i) for i, (o, r) in enumerate(zip(objects, records))]))
        self.client, self.timer, self.pending = None, None, None
        if not apply_moveit:
            self.status.publish(String(data=json.dumps({'ready': True, 'planning_scene_applied': False})))
            self.get_logger().info('Configured environment visualization ready; MoveIt is disabled.')
            return
        from moveit_msgs.msg import PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        self.scene = PlanningScene(is_diff=True)
        self.scene.world.collision_objects = [collision_message(record) for record in records]
        self.client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.timer = self.create_timer(1., self.apply)

    def apply(self):
        if self.client is None:
            return
        from moveit_msgs.srv import ApplyPlanningScene
        if self.pending is not None:
            if not self.pending.done():
                return
            try:
                if self.pending.result().success:
                    self.status.publish(String(data=json.dumps({'ready': True, 'planning_scene_applied': True})))
                    self.get_logger().info('Configured environment applied to MoveIt planning scene.')
                    self.timer.cancel()
                    return
                self.get_logger().error('MoveIt rejected environment planning scene; retrying.')
            except Exception as exc:
                self.get_logger().error(f'ApplyPlanningScene failed: {exc}')
            self.pending = None
        if self.client.service_is_ready():
            self.pending = self.client.call_async(ApplyPlanningScene.Request(scene=self.scene))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PipelineScene()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
