"""Look at the arm without starting Gazebo: RViz + joint sliders.

    ros2 launch arm_lab_bringup view.launch.py
    ros2 launch arm_lab_bringup view.launch.py config_file:=/path/to/variant.yaml

Fastest way to check that a geometry change came out the way you meant.
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
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

    cfg, paths = common.generate(
        config_file,
        ee_mass=common.as_float(arg('ee_mass')),
        payload_mass=common.as_float(arg('payload_mass'), 0.0) or 0.0,
        gravity=common.as_float(arg('gravity')),
        use_sim_time=False,
    )
    print(f'[arm_lab] {cfg.dof} dof, {cfg.arm_mass:.2f} kg, urdf at {paths["urdf"]}',
          file=sys.stderr)

    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen',
             parameters=[{'robot_description': paths['urdf_xml'],
                          'use_sim_time': False}]),
        Node(package='joint_state_publisher_gui',
             executable='joint_state_publisher_gui', output='screen'),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', os.path.join(bringup_share, 'rviz', 'arm.rviz')]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=''),
        DeclareLaunchArgument('ee_mass', default_value=''),
        DeclareLaunchArgument('payload_mass', default_value='0.0'),
        DeclareLaunchArgument('gravity', default_value=''),
        OpaqueFunction(function=launch_setup),
    ])
