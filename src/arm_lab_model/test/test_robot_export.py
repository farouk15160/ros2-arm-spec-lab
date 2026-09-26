import xml.etree.ElementTree as ET
import numpy as np
import pytest
from arm_lab_model.physical_robot import resolve_robot, physical_report
from arm_lab_model.robot_export import build_robot_urdf, build_robot_mjcf
from test_physical_robot import pendulum, project


def test_urdf_and_mjcf_share_full_com_and_inertial_tensor():
    robot = resolve_robot(project(pendulum()))
    urdf = ET.fromstring(build_robot_urdf(robot))
    inertial = urdf.find("link[@name='arm']/inertial")
    assert float(inertial.find('mass').get('value')) == pytest.approx(2)
    assert inertial.find('origin').get('xyz') == '1 0 0'
    assert urdf.find("joint[@name='world_to_base']") is not None
    mjcf = ET.fromstring(build_robot_mjcf(robot))
    inertial = mjcf.find(".//body[@name='arm']/inertial")
    assert float(inertial.get('mass')) == pytest.approx(2)
    assert inertial.get('pos') == '1 0 0'


def test_mujoco_gravity_matches_independent_analytic_pendulum():
    mujoco = pytest.importorskip('mujoco')
    robot = resolve_robot(project(pendulum()))
    model = mujoco.MjModel.from_xml_string(build_robot_mjcf(robot))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.qfrc_bias[0] == pytest.approx(-19.62)
    assert model.nu == 1
    assert data.subtree_com[0] == pytest.approx([1,0,0])


def test_floating_root_is_unanchored_and_mujoco_has_six_extra_velocity_dof():
    mujoco = pytest.importorskip('mujoco')
    original = pendulum()
    original['base']['type'] = 'floating'
    original['links'][0]['inertial'] = {'mass':1,'com':[0,0,0],
                                       'inertia':[.1,.1,.1,0,0,0],'source':'analytic'}
    robot = resolve_robot(project(original))
    urdf = ET.fromstring(build_robot_urdf(robot))
    assert urdf.find("link[@name='world']") is None
    model = mujoco.MjModel.from_xml_string(build_robot_mjcf(robot))
    assert model.nv == 7
    assert model.nq == 8
    assert model.nu == 1


def test_prismatic_effort_and_static_environment_are_exported():
    mujoco = pytest.importorskip('mujoco')
    original = pendulum()
    original['joints'][0] = {**original['joints'][0],'type':'prismatic','axis':[0,0,1]}
    robot = resolve_robot(project(original))
    env = ({'name':'table','geometry':{'type':'box','size':[1,1,.1]},
            'pose':{'xyz':[2,0,0],'rpy':[0,0,0]},'static':True},)
    model = mujoco.MjModel.from_xml_string(build_robot_mjcf(robot,environment=env))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    assert data.qfrc_bias[0] == pytest.approx(19.62)
    assert model.body('table').id > 0
    assert physical_report(robot)['joint_static_effort']['hinge'] == pytest.approx(19.62)


def test_unknown_effort_fails_urdf_but_allows_inverse_model():
    original = pendulum()
    original['joints'][0]['limits']['effort'] = None
    robot = resolve_robot(project(original))
    with pytest.raises(ValueError, match='missing limits'):
        build_robot_urdf(robot)
    assert 'motor' in build_robot_mjcf(robot)


def test_legacy_adapter_preserves_analytical_arm_static_torque():
    from pathlib import Path
    from arm_lab_model.project_config import load_project
    from arm_lab_model.config import load_config
    from arm_lab_model.kinematics import ArmModel
    mujoco = pytest.importorskip('mujoco')
    path = Path(__file__).parents[1]/'config/pipeline/project.yaml'
    config = load_project(path)
    robot = resolve_robot(config)
    cfg = load_config(str(config.legacy_source))
    expected = ArmModel(cfg).gravity_torque(np.zeros(cfg.dof))
    actual = physical_report(robot)['joint_static_effort']
    assert np.array([actual[name] for name in cfg.joint_names]) == pytest.approx(expected,abs=1e-6)
    model = mujoco.MjModel.from_xml_string(build_robot_mjcf(robot))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    for index, name in enumerate(cfg.joint_names):
        address = model.jnt_dofadr[model.joint(name).id]
        assert data.qfrc_bias[address] == pytest.approx(expected[index], abs=1e-6)


def test_stl_mesh_export_preserves_scale_and_full_inertia(tmp_path):
    from test_mesh_physics import cube_stl
    mujoco = pytest.importorskip('mujoco')
    original = pendulum()
    path = cube_stl(tmp_path/'cube.stl')
    original['links'][1]['geometry'] = {'type':'mesh','path':str(path),'units':'m','scale':[2,4,6],
        'origin':{'xyz':[1,2,3],'rpy':[0,0,.7]}}
    robot = resolve_robot(project(original))
    urdf = ET.fromstring(build_robot_urdf(robot))
    mesh = urdf.find("link[@name='arm']/visual/geometry/mesh")
    assert mesh.get('scale') == '2 4 6'
    model = mujoco.MjModel.from_xml_string(build_robot_mjcf(robot))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    report = physical_report(robot)
    assert data.subtree_com[0] == pytest.approx(report['center_of_mass'])
    assert data.qfrc_bias[0] == pytest.approx(report['joint_static_effort']['hinge'])


def test_optional_mock_and_gazebo_control_exports():
    robot = resolve_robot(project(pendulum()))
    xml = ET.fromstring(build_robot_urdf(robot,ros2_control=True))
    assert xml.find('ros2_control/hardware/plugin').text == 'mock_components/GenericSystem'
    assert xml.find('ros2_control/joint/command_interface').get('name') == 'position'
    xml = ET.fromstring(build_robot_urdf(robot,ros2_control=True,
                       hardware_plugin='gz_ros2_control/GazeboSimSystem',controllers_file='/tmp/control.yaml'))
    assert xml.find('gazebo/plugin/parameters').text == '/tmp/control.yaml'
