"""Launch graph boundaries: generic simulation works without MoveIt configured."""
from pathlib import Path
import importlib.util

import pytest
import yaml


@pytest.fixture
def launch_module(tmp_path, monkeypatch):
    pytest.importorskip('launch_ros')
    import launch.logging
    if not any(launch.logging.launch_config.file_handlers):
        launch.logging.launch_config.log_dir = str(tmp_path)
    source = Path(__file__).resolve().parents[2]
    path = source / 'arm_lab_bringup/launch/pipeline.launch.py'
    spec = importlib.util.spec_from_file_location('tested_pipeline_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'get_package_share_directory', lambda name: str(source / name))
    return module


def project_variant(tmp_path, mode):
    root = Path(__file__).resolve().parents[2] / 'arm_lab_model/config/pipeline'
    project = yaml.safe_load((root / 'project_ur5e.yaml').read_text())
    project = {key: str(root / value) if key != 'schema_version' else value for key, value in project.items()}
    if mode == 'absent':
        project.pop('moveit')
    elif mode == 'disabled':
        config = tmp_path / 'moveit.yaml'
        config.write_text('schema_version: 1\nenabled: false\n')
        project['moveit'] = str(config)
    path = tmp_path / 'project.yaml'
    path.write_text(yaml.safe_dump(project))
    return path


def launch_nodes(module, project, rviz=True):
    from launch import LaunchContext
    context = LaunchContext()
    context.launch_configurations.update({
        'project_file': str(project), 'benchmark': 'false', 'benchmark_robot': '',
        'benchmark_reference': '', 'output_dir': '/tmp/launch-test-benchmarks',
        'group': '', 'scenario_id': 'launch_test', 'rviz': str(rviz).lower()})
    return module.launch_setup(context)


@pytest.mark.parametrize('mode', ['absent', 'disabled'])
def test_simulation_only_project_has_no_moveit_nodes_and_uses_generic_rviz(launch_module, tmp_path, mode):
    nodes = launch_nodes(launch_module, project_variant(tmp_path, mode))
    assert {node.node_executable for node in nodes} == {
        'robot_state_publisher', 'pipeline_sim', 'pipeline_scene', 'rviz2'}
    rviz = next(node for node in nodes if node.node_executable == 'rviz2')
    assert any(getattr(value, 'text', '').endswith('/rviz/robot.rviz') for argument in rviz.cmd for value in argument)


def test_ur5e_planning_graph_remains_enabled(launch_module, tmp_path):
    nodes = launch_nodes(launch_module, project_variant(tmp_path, 'enabled'), rviz=False)
    assert {node.node_executable for node in nodes} == {
        'robot_state_publisher', 'pipeline_sim', 'pipeline_scene', 'move_group', 'pipeline_target'}
