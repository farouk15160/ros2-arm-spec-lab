"""Optional ROS adapters; source ROS Humble before running these tests."""
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip('rclpy')
from arm_lab_gui.pipeline_perception import image_array, transform_matrix
from arm_lab_gui.pipeline_scene import collision_message, environment_marker
from arm_lab_model.scene_config import collision_objects


def test_depth_big_endian_padded_row_decodes_without_alignment_assumptions():
    message = SimpleNamespace(encoding='16UC1',width=2,height=2,step=6,is_bigendian=1,
                              data=bytes([0,1,0,2,0,0,0,3,0,4,0,0]))
    np.testing.assert_array_equal(image_array(message), [[1,2],[3,4]])


def test_image_rejects_truncated_buffer():
    message = SimpleNamespace(encoding='32FC1',width=2,height=2,step=8,is_bigendian=0,data=bytes(8))
    with pytest.raises(ValueError, match='buffer'):
        image_array(message)


def test_collision_geometry_and_rviz_marker_share_world_pose():
    obj = {'name':'box','geometry':{'type':'box','size':[1,2,3]},'frame':'world',
           'pose':{'xyz':[1,2,3],'rpy':[0,0,0]},'static':True,'mass':None}
    record = collision_objects([obj])[0]
    collision = collision_message(record)
    marker = environment_marker(obj, record, 0)
    assert list(collision.primitives[0].dimensions) == [1.,2.,3.]
    assert marker.pose == collision.primitive_poses[0]
    assert marker.scale.z == 3.


def test_tf_quaternion_rotation_correctly_moves_optical_points():
    transform = SimpleNamespace(rotation=SimpleNamespace(x=0,y=0,z=2**-.5,w=2**-.5),
                                translation=SimpleNamespace(x=1,y=2,z=3))
    np.testing.assert_allclose(transform_matrix(transform) @ [1,0,0,1], [1,3,3,1], atol=1e-12)


def test_ros_camera_info_numpy_scalars_feed_strict_projection():
    from sensor_msgs.msg import CameraInfo
    from arm_lab_gui.pipeline_perception import PipelinePerception
    from arm_lab_model.perception import depth_to_pointcloud
    state = SimpleNamespace(intrinsics={})
    info = CameraInfo(width=3, height=3)
    info.k = [2.,0.,1.,0.,2.,1.,0.,0.,1.]
    PipelinePerception.on_info(state, 'camera', info)
    points = depth_to_pointcloud([[2.,2.,2.]]*3, state.intrinsics['camera'])
    np.testing.assert_allclose(points[0], [-1,-1,2])
