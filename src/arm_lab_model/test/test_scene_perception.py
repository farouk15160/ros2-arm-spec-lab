"""Public scene and perception contracts, with analytical camera fixtures."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from arm_lab_model.scene_config import validate_environment, resolve_environment, collision_objects


def test_environment_resolution_preserves_pose_and_collision_dimensions(tmp_path):
    section = {'schema_version': 1, 'enabled': True, 'objects': [
        {'name': 'table', 'geometry': {'type': 'box', 'size': [2, 1, .1]},
         'pose': {'xyz': [1, 0, .4], 'rpy': [0, 0, 1.5707963267948966]},
         'frame': 'world', 'static': True, 'mass': None}]}
    validate_environment(section)
    project = SimpleNamespace(sections={'environment': section}, files={'environment': tmp_path/'world.yaml'})
    objects = resolve_environment(project)
    collision = collision_objects(objects)[0]
    assert collision['id'] == 'table'
    assert collision['primitive']['dimensions'] == [2, 1, .1]
    assert collision['pose']['position'] == [1, 0, .4]
    assert collision['pose']['orientation_xyzw'] == pytest.approx([0, 0, .7071067812, .7071067812])
    assert section['objects'][0]['pose']['xyz'] == [1, 0, .4]


def camera_spec():
    return {'type': 'rgbd', 'model': 'synthetic_pinhole', 'parent_link': 'wrist',
            'transform': {'xyz': [0, 0, .1], 'rpy': [0, 0, 0]},
            'resolution': [3, 3], 'frame_rate_hz': 30,
            'intrinsics': {'fx': 2., 'fy': 2., 'cx': 1., 'cy': 1.},
            'clip': [.1, 5.], 'noise_stddev': 0.}


def test_sensor_mount_checks_parent_and_calibrated_projection():
    from arm_lab_model.scene_config import validate_sensors, resolve_sensors
    section = {'schema_version': 1, 'enabled': True, 'cameras': {'camera': camera_spec()}}
    validate_sensors(section)
    project = SimpleNamespace(sections={'sensors': section, 'robot': {'format': 'tree', 'links': [{'name': 'wrist'}]}})
    assert resolve_sensors(project)[0]['optical_frame'] == 'camera_optical_frame'
    invalid = SimpleNamespace(sections={'sensors': section, 'robot': {'format': 'tree', 'links': [{'name': 'base'}]}})
    with pytest.raises(ValueError, match='parent_link'):
        resolve_sensors(invalid)


def test_depth_known_plane_and_invalid_pixels():
    from arm_lab_model.perception import depth_to_pointcloud, filter_pointcloud
    depth = np.array([[2., 2., 2.], [2., 0., 2.], [2., np.nan, np.inf]])
    cloud = depth_to_pointcloud(depth, camera_spec()['intrinsics'], min_depth=.1, max_depth=3.)
    assert cloud.shape == (6, 3)
    np.testing.assert_allclose(cloud[0], [-1, -1, 2])
    np.testing.assert_allclose(cloud[-1], [-1, 1, 2])
    filtered = filter_pointcloud(cloud, bounds=[[-.1, -2, 1], [1.1, 2, 3]])
    assert filtered.shape == (3, 3)


def test_voxel_centroids_transform_and_input_immutable():
    from arm_lab_model.perception import filter_pointcloud
    points = np.array([[.01, 0, 1], [.03, 0, 1], [.2, 0, 1]])
    transform = np.eye(4)
    transform[0, 3] = 1.
    result = filter_pointcloud(points, voxel_size=.1, transform=transform)
    np.testing.assert_allclose(result, [[1.02, 0, 1], [1.2, 0, 1]])
    np.testing.assert_allclose(points[0], [.01, 0, 1])


def test_perception_dependency_and_plugin_errors_are_explicit():
    from arm_lab_model.perception import validate_perception, load_algorithm, moveit_octomap_config
    with pytest.raises(ValueError, match='plugin'):
        validate_perception({'schema_version': 1, 'enabled': True, 'slam': {'enabled': True}})
    with pytest.raises(RuntimeError, match='unavailable'):
        load_algorithm('nonexistent_arm_lab_plugin:run')
    config = {'schema_version': 1, 'enabled': True,
              'pointcloud': {'enabled': True, 'camera': 'wrist', 'min_depth': .1, 'max_depth': 5., 'voxel_size': .01, 'target_frame': 'world'},
              'octomap': {'enabled': True, 'resolution': .05, 'max_range': 4.}}
    validate_perception(config)
    result = moveit_octomap_config(config)
    assert result['sensors'] == ['depth_cloud']
    assert result['depth_cloud']['max_update_rate'] == 5.0
    assert result['depth_cloud']['point_cloud_topic'] == '/perception/octomap_points'


def test_color_detector_recognizes_separate_components_without_semantic_claims():
    from arm_lab_model.perception import color_components
    image = np.zeros((5, 6, 3), dtype=np.uint8)
    image[0:2, 0:2] = [255, 0, 0]
    image[3:5, 4:6] = [255, 0, 0]
    result = color_components(image, {'lower_rgb': [200, 0, 0], 'upper_rgb': [255, 10, 10], 'min_pixels': 3})
    assert [r['bbox_xyxy'] for r in result] == [[0, 0, 1, 1], [4, 3, 5, 4]]
    assert all(r['label'] == 'color_region' for r in result)


@pytest.mark.parametrize('field,value', [('resolution', [0, 2]), ('frame_rate_hz', 0), ('clip', [2, 1]),
                                        ('noise_stddev', -1), ('fov_y_deg', 10), ('parent_link', 'bad/name'),
                                        ('intrinsics', {'fx': 0, 'fy': 1, 'cx': 1, 'cy': 1})])
def test_invalid_camera_is_rejected(field, value):
    from arm_lab_model.scene_config import validate_sensors
    with pytest.raises(ValueError):
        validate_sensors({'schema_version': 1, 'enabled': True, 'cameras': {'camera': {**camera_spec(), field: value}}})


@pytest.mark.parametrize('geometry', [{'type': 'box', 'size': [0, 1, 1]}, {'type': 'sphere', 'radius': -1},
                                    {'type': 'mesh', 'path': 'a.stl', 'scale': [1, 1, 1]},
                                    {'type': 'mesh', 'path': 'a.obj', 'scale': [1, 1, 1], 'units': 'm'},
                                    {'type': 'mesh', 'path': 'a.stl', 'scale': [1, 0, 1], 'units': 'm'}, None])
def test_geometry_requires_units_positive_sizes_and_stl(geometry):
    from arm_lab_model.scene_config import validate_geometry
    with pytest.raises(ValueError):
        validate_geometry(geometry)


def test_dynamic_environment_requires_mass_and_supported_frame():
    obj = {'name': 'box', 'geometry': {'type': 'sphere', 'radius': .1}, 'frame': 'world',
           'pose': {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]}, 'static': False, 'mass': None}
    with pytest.raises(ValueError, match='mass'):
        validate_environment({'schema_version': 1, 'enabled': True, 'objects': [obj]})
    validate_environment({'schema_version': 1, 'enabled': True, 'objects': [{**obj, 'mass': 1}]})
    with pytest.raises(ValueError, match='frame'):
        validate_environment({'schema_version': 1, 'enabled': True, 'objects': [{**obj, 'mass': 1, 'frame': 'unknown'}]})


def test_scene_disabled_and_missing_mesh_paths(tmp_path):
    project = SimpleNamespace(sections={}, files={})
    assert resolve_environment(project) == ()
    obj = {'name': 'mesh', 'geometry': {'type': 'mesh', 'path': 'missing.stl', 'scale': [1, 1, 1], 'units': 'mm'},
           'frame': 'world', 'pose': {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]}, 'static': True, 'mass': None}
    project = SimpleNamespace(sections={'environment': {'schema_version': 1, 'enabled': True, 'objects': [obj]}}, files={'environment': tmp_path/'environment.yaml'})
    with pytest.raises(ValueError, match='missing STL'):
        resolve_environment(project)


def test_mesh_collision_scale_millimetres(tmp_path):
    # Closed tetrahedron in millimetres, independent from any mass-property calculation.
    triangles = [[[0,0,0],[0,1000,0],[1000,0,0]], [[0,0,0],[1000,0,0],[0,0,1000]],
                 [[0,0,0],[0,0,1000],[0,1000,0]], [[1000,0,0],[0,1000,0],[0,0,1000]]]
    path = tmp_path/'tetra.stl'
    path.write_text('solid tetra\n' + '\n'.join('facet normal 0 0 0\nouter loop\n' + '\n'.join('vertex '+' '.join(map(str,p)) for p in t) + '\nendloop\nendfacet' for t in triangles) + '\nendsolid tetra\n')
    obj = {'name': 'mesh', 'geometry': {'type': 'mesh', 'path': 'tetra.stl', 'scale': [1,2,1], 'units': 'mm'},
           'frame': 'world', 'pose': {'xyz': [0,0,0], 'rpy': [0,0,0]}, 'static': True, 'mass': None}
    project = SimpleNamespace(sections={'environment': {'schema_version': 1, 'enabled': True, 'objects': [obj]}}, files={'environment': tmp_path/'environment.yaml'})
    mesh = collision_objects(resolve_environment(project))[0]['mesh']
    assert mesh['vertices'][1] == [0., 2., 0.]
    assert mesh['triangles'][-1] == [9,10,11]


def test_perception_dependencies_reject_rgb_for_cloud():
    from arm_lab_model.perception import validate_perception_dependencies
    sensors = {'schema_version': 1, 'enabled': True, 'cameras': {'camera': {**camera_spec(), 'type': 'rgb'}}}
    config = {'schema_version': 1, 'enabled': True, 'pointcloud': {'enabled': True, 'camera': 'camera',
              'min_depth': .1, 'max_depth': 5., 'voxel_size': .01, 'target_frame': 'world'}}
    project = SimpleNamespace(sections={'sensors': sensors, 'perception': config, 'robot': {'format': 'tree', 'links': [{'name': 'wrist'}]}})
    with pytest.raises(ValueError, match='depth or rgbd'):
        validate_perception_dependencies(project)


@pytest.mark.parametrize('kwargs', [{'voxel_size': 0}, {'bounds': [[1,0,0],[0,1,1]]},
                                    {'transform': np.diag([2,1,1,1])}, {'transform': np.zeros((3,3))}])
def test_pointcloud_rejects_invalid_filter_settings(kwargs):
    from arm_lab_model.perception import filter_pointcloud
    with pytest.raises(ValueError):
        filter_pointcloud([[0,0,1]], **kwargs)


def test_depth_millimetres_and_empty_result():
    from arm_lab_model.perception import depth_to_pointcloud
    cloud = depth_to_pointcloud(np.array([[2000]], dtype=np.uint16), {'fx': 1, 'fy': 1, 'cx': 0, 'cy': 0}, depth_scale=.001)
    np.testing.assert_allclose(cloud, [[0,0,2]])
    assert depth_to_pointcloud([[0]], {'fx': 1, 'fy': 1, 'cx': 0, 'cy': 0}).shape == (0,3)


def test_mujoco_camera_export_keeps_asymmetric_calibration():
    import xml.etree.ElementTree as ET
    from arm_lab_model.robot_export import build_robot_mjcf
    robot = {'name': 'camera_fixture', 'base': {'link': 'base', 'type': 'fixed'},
             'links': [{'name': 'base', 'inertial': {'mass': 1, 'com': [0,0,0], 'inertia': [.1,.1,.1,0,0,0]}}], 'joints': [], 'end_effectors': {}}
    camera = {**camera_spec(), 'name': 'camera', 'parent_link': 'base', 'optical_frame': 'camera_optical_frame',
              'resolution': [100,80], 'intrinsics': {'fx':100, 'fy':80, 'cx':60, 'cy':45}}
    camera_xml = ET.fromstring(build_robot_mjcf(robot, sensors=[camera])).find('.//camera')
    assert camera_xml.get('focalpixel') == '100 80'
    assert camera_xml.get('principalpixel') == '-10.5 -5.5'


def test_real_mujoco_renderer_known_sphere_projection(tmp_path):
    """Headless renderer test in separate process to select EGL before importing MuJoCo."""
    import os
    import subprocess
    import sys
    pytest.importorskip('mujoco')
    script = '''
import mujoco, numpy as np
from arm_lab_model.robot_export import build_robot_mjcf
from arm_lab_model.sensor_runtime import CameraRenderer
robot = {'name':'fixture','base':{'link':'base','type':'fixed'},'links':[{'name':'base','inertial':{'mass':1,'com':[0,0,0],'inertia':[.1,.1,.1,0,0,0]}}],'joints':[],'end_effectors':{}}
camera = {'name':'camera','type':'rgbd','parent_link':'base','transform':{'xyz':[0,0,0],'rpy':[0,0,0]},'resolution':[100,80],'intrinsics':{'fx':100,'fy':80,'cx':60,'cy':45},'optical_frame':'camera_optical_frame','clip':[.1,5.],'noise_stddev':0}
obj = {'name':'target','geometry':{'type':'sphere','radius':.04},'pose':{'xyz':[2,-.2,-.2],'rpy':[0,0,0]},'static':True,'mass':None}
model = mujoco.MjModel.from_xml_string(build_robot_mjcf(robot,sensors=[camera],environment=[obj]))
data = mujoco.MjData(model)
mujoco.mj_forward(model,data)
renderer = CameraRenderer(model,[camera])
try:
 frame = renderer.render(data,'camera')
 y,x = np.where(np.isfinite(frame['depth']))
 assert abs(x.mean()-70)<.6 and abs(y.mean()-53)<.6, (x.mean(),y.mean())
 assert 1.95 < np.nanmin(frame['depth']) < 1.98
 assert frame['rgb'].shape == (80,100,3)
 assert frame['frame_id'] == 'camera_optical_frame'
finally:
 renderer.close()
'''
    env = {**os.environ, 'MUJOCO_GL': 'egl', 'PYTHONPATH': str(Path(__file__).parents[1])}
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, timeout=30)
    if result.returncode and ('EGL' in result.stderr or 'OpenGL' in result.stderr):
        pytest.skip('EGL unavailable: ' + result.stderr[-300:])
    assert result.returncode == 0, result.stderr


def test_dynamic_body_cannot_be_exported_as_stale_moveit_collision():
    obj = {'name': 'moving', 'geometry': {'type': 'sphere', 'radius': .1},
           'pose': {'xyz': [0,0,0], 'rpy': [0,0,0]}, 'frame': 'world', 'static': False, 'mass': 1.}
    validate_environment({'schema_version': 1, 'enabled': True, 'objects': [obj]})
    with pytest.raises(ValueError, match='live pose'):
        collision_objects([obj])


def test_hd_camera_sets_offscreen_buffer_to_largest_resolution():
    import xml.etree.ElementTree as ET
    from arm_lab_model.robot_export import build_robot_mjcf
    robot = {'name': 'camera_fixture', 'base': {'link': 'base', 'type': 'fixed'},
             'links': [{'name': 'base', 'inertial': {'mass': 1, 'com': [0,0,0], 'inertia': [.1,.1,.1,0,0,0]}}], 'joints': [], 'end_effectors': {}}
    camera = {**camera_spec(), 'name': 'camera', 'parent_link': 'base', 'optical_frame': 'camera_optical_frame',
              'resolution': [1280,720], 'intrinsics': {'fx':1000, 'fy':1000, 'cx':639.5, 'cy':359.5}}
    visual = ET.fromstring(build_robot_mjcf(robot, sensors=[camera])).find('visual/global')
    assert visual is not None
    assert int(visual.get('offwidth')) >= 1280
    assert int(visual.get('offheight')) >= 720


def test_camera_clipping_is_metric_even_in_large_factory_scene():
    import os
    import subprocess
    import sys
    pytest.importorskip('mujoco')
    script = '''
import mujoco, numpy as np
from arm_lab_model.sensor_runtime import CameraRenderer
# MuJoCo global default near plane would be one metre for extent=100.
model = mujoco.MjModel.from_xml_string("""<mujoco><statistic extent="100"/><worldbody>
<camera name="camera" pos="0 0 0" fovy="60"/>
<geom type="sphere" size=".05" pos="0 0 -.5"/>
</worldbody></mujoco>""")
data = mujoco.MjData(model)
mujoco.mj_forward(model,data)
camera = {'name':'camera','type':'rgbd','resolution':[160,120],
'intrinsics':{'fx':104.,'fy':104.,'cx':79.5,'cy':59.5},
'optical_frame':'camera_optical_frame','clip':[.01,10.],'noise_stddev':0}
renderer = CameraRenderer(model,[camera])
try:
 frame = renderer.render(data,'camera')
 assert np.isfinite(frame['depth']).sum() > 100, 'valid nearby target clipped by scene extent'
 assert abs(float(np.nanmin(frame['depth']))-.45) < .005
 assert abs(float(model.vis.map.znear*model.stat.extent)-.01) < 1e-6
 assert abs(float(model.vis.map.zfar*model.stat.extent)-10.) < 1e-5
finally:
 renderer.close()
'''
    env = {**os.environ, 'MUJOCO_GL': 'egl', 'PYTHONPATH': str(Path(__file__).parents[1])}
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, timeout=30)
    if result.returncode and ('EGL' in result.stderr or 'OpenGL' in result.stderr):
        pytest.skip('EGL unavailable: ' + result.stderr[-300:])
    assert result.returncode == 0, result.stderr
