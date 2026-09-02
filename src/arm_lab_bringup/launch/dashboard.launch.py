"""Attach the dashboard (and capability publisher) to an already running sim.

    ros2 launch arm_lab_bringup dashboard.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    config_file = arg('config_file') or os.path.join(
        get_package_share_directory('arm_lab_model'), 'config', 'arm_config.yaml')
    params = {
        'config_file': config_file,
        'payload_mass': float(arg('payload_mass') or 0.0),
        'use_sim_time': arg('use_sim_time').lower() in ('1', 'true', 'yes'),
    }
    if arg('ee_mass'):
        params['ee_mass'] = float(arg('ee_mass'))
    if arg('gravity'):
        params['gravity'] = float(arg('gravity'))

    return [
        Node(package='arm_lab_gui', executable='dashboard', name='arm_dashboard',
             output='screen', parameters=[params]),
        Node(package='arm_lab_gui', executable='capability_node',
             name='arm_capability', output='screen', parameters=[params],
             condition=IfCondition(LaunchConfiguration('capability'))),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=''),
        DeclareLaunchArgument('ee_mass', default_value=''),
        DeclareLaunchArgument('payload_mass', default_value='0.0'),
        DeclareLaunchArgument('gravity', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('capability', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
