"""Unified project -> robot description -> MuJoCo -> MoveIt -> RViz pipeline."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_setup(context):
    from arm_lab_model.project_config import load_project
    from arm_lab_model.physical_robot import resolve_robot
    from arm_lab_model.robot_export import build_robot_urdf
    from arm_lab_model.scene_config import resolve_sensors

    def arg(name):
        return LaunchConfiguration(name).perform(context)

    def boolean(name):
        value = arg(name).lower()
        if value not in ('true', 'false'):
            raise ValueError(name + ': expected true or false')
        return value == 'true'

    path = arg('project_file')
    if not path:
        path = str(Path(get_package_share_directory('arm_lab_model')) / 'config' / 'pipeline' / 'project_ur5e.yaml')
    project = load_project(path)
    robot = resolve_robot(project)
    if project.sections.get('simulation', {}).get('backend', 'mujoco') != 'mujoco':
        raise ValueError('pipeline.launch.py currently supports the MuJoCo backend')
    sensors = resolve_sensors(project)
    xml = build_robot_urdf(robot, sensors=sensors)
    planning_enabled = project.sections.get('moveit', {}).get('enabled', False)
    description = {'robot_description': ParameterValue(xml, value_type=str), 'use_sim_time': True}
    parameters = description
    if planning_enabled:
        from arm_lab_model.perception import moveit_octomap_config
        from arm_lab_kinematics.pipeline_moveit import build_moveit_config
        planning = build_moveit_config(robot, project.sections['moveit'])
        octomap = moveit_octomap_config(project.sections.get('perception', {'schema_version': 1, 'enabled': False}))
        if octomap.get('sensors') == []:
            octomap = {}
        parameters = {**planning, **octomap, **description}
    nodes = [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen', parameters=[description]),
        Node(package='arm_lab_gui', executable='pipeline_sim', output='screen', parameters=[{
            'project_file': str(project.source_path), 'benchmark': boolean('benchmark'),
            'benchmark_robot': arg('benchmark_robot'), 'benchmark_reference': arg('benchmark_reference'),
            'output_dir': arg('output_dir'), 'headless_render': True, 'scenario_id': arg('scenario_id')}]),
        Node(package='arm_lab_gui', executable='pipeline_scene', output='screen',
             parameters=[{'project_file': str(project.source_path), 'use_sim_time': True,
                          'apply_moveit': planning_enabled}]),
    ]
    if planning_enabled:
        nodes += [
            Node(package='moveit_ros_move_group', executable='move_group',
                 output='screen', parameters=[parameters]),
            Node(package='arm_lab_kinematics', executable='pipeline_target', output='screen',
                 parameters=[{'project_file': str(project.source_path), 'use_sim_time': True,
                              'group': arg('group'), 'scenario_id': arg('scenario_id')}]),
        ]
    if project.sections.get('perception', {}).get('enabled', False):
        nodes += [Node(package='arm_lab_gui', executable='pipeline_perception', output='screen',
                       parameters=[{'project_file': str(project.source_path), 'use_sim_time': True}])]
    if boolean('rviz'):
        config_name = 'pipeline.rviz' if planning_enabled else 'robot.rviz'
        config = Path(get_package_share_directory('arm_lab_bringup')) / 'rviz' / config_name
        nodes += [Node(package='rviz2', executable='rviz2', output='screen',
                       arguments=['-d', str(config)], parameters=[parameters])]
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('project_file', default_value=''),
        DeclareLaunchArgument('benchmark', default_value='false'),
        DeclareLaunchArgument('benchmark_robot', default_value=''),
        DeclareLaunchArgument('benchmark_reference', default_value=''),
        DeclareLaunchArgument('output_dir', default_value='benchmark_reports'),
        DeclareLaunchArgument('group', default_value=''),
        DeclareLaunchArgument('scenario_id', default_value='ros_execution'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
