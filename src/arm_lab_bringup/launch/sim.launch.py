"""Full simulation: Gazebo or MuJoCo + RViz + the capability dashboard.

    ros2 launch arm_lab_bringup sim.launch.py
    ros2 launch arm_lab_bringup sim.launch.py ee_mass:=1.4 payload_mass:=2.0
    ros2 launch arm_lab_bringup sim.launch.py config_file:=/path/to/variant.yaml
    ros2 launch arm_lab_bringup sim.launch.py gravity:=3.72 gz_gui:=false
    ros2 launch arm_lab_bringup sim.launch.py simulator:=mujoco
    ros2 launch arm_lab_bringup sim.launch.py simulator:=mujoco mujoco_gui:=false

`simulator:=mujoco` replaces Gazebo and ros2_control with the mujoco_sim node,
which serves the same topics and trajectory action; `mujoco_gui` opens its
viewer window. It needs `pip install mujoco` but no gz_ros2_control.
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler)
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    bringup_share = get_package_share_directory('arm_lab_bringup')
    config_file = arg('config_file') or os.path.join(
        get_package_share_directory('arm_lab_model'), 'config', 'arm_config.yaml')

    ee_mass = common.as_float(arg('ee_mass'))
    payload_mass = common.as_float(arg('payload_mass'), 0.0) or 0.0
    gravity = common.as_float(arg('gravity'))
    command_interface = arg('command_interface') or None

    cfg, paths = common.generate(
        config_file,
        ee_mass=ee_mass,
        payload_mass=payload_mass,
        gravity=gravity,
        command_interface=command_interface,
        initial_pose=arg('initial_pose'),
        use_sim_time=True,
    )

    world = arg('world') or os.path.join(bringup_share, 'worlds', 'arm_test_world.sdf')
    if gravity is not None:
        world = common.patch_world_gravity(world, gravity)

    print(f'[arm_lab] config      : {cfg.source_path}', file=sys.stderr)
    print(f'[arm_lab] generated   : {paths["dir"]}', file=sys.stderr)
    print(f'[arm_lab] dof {cfg.dof}, arm mass {cfg.arm_mass:.2f} kg, '
          f'gravity {cfg.gravity:.2f} m/s^2, '
          f'command interface {command_interface or cfg.control.get("command_interface")}',
          file=sys.stderr)

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', os.path.join(bringup_share, 'rviz', 'arm.rviz')],
        parameters=[{'use_sim_time': True}],
        output='log',
    )

    live_parameters = [{
        'config_file': cfg.source_path,
        'ee_mass': float(cfg.end_effector.mass),
        'payload_mass': float(payload_mass),
        'gravity': float(cfg.gravity),
        'use_sim_time': True,
    }]
    dashboard = Node(
        package='arm_lab_gui', executable='dashboard', name='arm_dashboard',
        condition=IfCondition(LaunchConfiguration('dashboard')),
        output='screen',
        parameters=live_parameters,
    )
    capability = Node(
        package='arm_lab_gui', executable='capability_node', name='arm_capability',
        condition=IfCondition(LaunchConfiguration('capability')),
        output='screen',
        parameters=live_parameters,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': paths['urdf_xml'], 'use_sim_time': True}],
    )

    simulator = (arg('simulator') or 'gazebo').strip().lower()
    if simulator not in ('gazebo', 'mujoco'):
        raise RuntimeError(f"simulator must be 'gazebo' or 'mujoco', not {simulator!r}")
    print(f'[arm_lab] simulator   : {simulator}', file=sys.stderr)

    if simulator == 'mujoco':
        # MuJoCo is the clock source, so it runs on wall time itself.
        mujoco_sim = Node(
            package='arm_lab_gui', executable='mujoco_sim', name='mujoco_sim',
            output='screen',
            parameters=[{
                'config_file': cfg.source_path,
                'ee_mass': float(cfg.end_effector.mass),
                'payload_mass': float(payload_mass),
                'gravity': float(cfg.gravity),
                'initial_pose': arg('initial_pose'),
                'world': world,
                'gui': common.as_bool(arg('mujoco_gui')),
            }],
        )
        # Like Gazebo's on_exit_shutdown: no simulator, no point in the rest.
        stop_with_sim = RegisterEventHandler(OnProcessExit(
            target_action=mujoco_sim, on_exit=[EmitEvent(event=Shutdown())]))
        return [mujoco_sim, stop_with_sim, robot_state_publisher, rviz, dashboard,
                capability]

    headless = not common.as_bool(arg('gz_gui'))
    gz_args = f'-r -v 3 {world}' + (' -s --headless-rendering' if headless else '')

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items(),
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description',
                   '-name', cfg.name,
                   '-x', '0', '-y', '0', '-z', '0.0'],
    )

    def spawner(name, extra=()):
        return Node(
            package='controller_manager',
            executable='spawner',
            output='screen',
            arguments=[name, '--controller-manager', '/controller_manager',
                       '--controller-manager-timeout', '60', *extra],
        )

    jsb = spawner('joint_state_broadcaster')
    arm_ctrl = spawner('arm_controller')
    controllers = [
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm_ctrl])),
    ]
    last = arm_ctrl
    if cfg.end_effector.finger_joint_names:
        grip = spawner('gripper_controller')
        controllers.append(
            RegisterEventHandler(OnProcessExit(target_action=last, on_exit=[grip])))
        last = grip
    if (command_interface or cfg.control.get('command_interface')) == 'velocity':
        # Loaded but left stopped; the dashboard's speed sweep switches to it.
        spare = spawner('arm_velocity_controller', extra=['--inactive'])
        controllers.append(
            RegisterEventHandler(OnProcessExit(target_action=last, on_exit=[spare])))

    return [gz, robot_state_publisher, clock_bridge, spawn, *controllers,
            rviz, dashboard, capability]


def generate_launch_description():
    args = [
        DeclareLaunchArgument('config_file', default_value='',
                              description='arm_config.yaml to build the robot from'),
        DeclareLaunchArgument('ee_mass', default_value='',
                              description='end-effector mass override, kg'),
        DeclareLaunchArgument('payload_mass', default_value='0.0',
                              description='rigid test mass welded at the TCP, kg'),
        DeclareLaunchArgument('gravity', default_value='',
                              description='gravity override, m/s^2 (3.72 Mars)'),
        DeclareLaunchArgument('command_interface', default_value='',
                              description='position | velocity | effort'),
        DeclareLaunchArgument('initial_pose', default_value='home',
                              description='named pose from test_poses'),
        DeclareLaunchArgument('world', default_value='',
                              description='world SDF; blank uses the test bench'),
        DeclareLaunchArgument('simulator', default_value='gazebo',
                              description='gazebo | mujoco'),
        DeclareLaunchArgument('gz_gui', default_value='true',
                              description='show the Gazebo window'),
        DeclareLaunchArgument('mujoco_gui', default_value='true',
                              description='show the MuJoCo viewer (simulator:=mujoco)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('dashboard', default_value='true'),
        DeclareLaunchArgument('capability', default_value='true',
                              description='publish capability topics for plotting'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
