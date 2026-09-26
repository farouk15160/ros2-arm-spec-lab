"""Scene/controller middleware boundary checks without starting ROS networking."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('options', [{}, {'schema_version': 1, 'enabled': False}])
def test_simulation_only_controllers_use_fallback_for_every_joint(options):
    pytest.importorskip('rclpy')
    from arm_lab_gui.pipeline_sim_node import configured_controllers
    assert configured_controllers(options, ('hip', 'knee')) == {'arm_controller': ('hip', 'knee')}


def test_visualization_only_scene_publishes_geometry_without_moveit_service(monkeypatch):
    pytest.importorskip('rclpy')
    import json
    import arm_lab_gui.pipeline_scene as module
    published, clients, timers = {}, [], []
    project = SimpleNamespace(sections={'moveit': {'enabled': False}})
    monkeypatch.setattr(module, 'load_project', lambda _: project)
    monkeypatch.setattr(module, 'resolve_environment', lambda _: ())
    monkeypatch.setattr(module.Node, '__init__', lambda *_: None)
    monkeypatch.setattr(module.PipelineScene, 'declare_parameter',
                        lambda self, name, default: SimpleNamespace(value=default))
    monkeypatch.setattr(module.PipelineScene, 'create_publisher',
                        lambda self, cls, topic, qos: SimpleNamespace(publish=lambda msg: published.setdefault(topic, []).append(msg)))
    monkeypatch.setattr(module.PipelineScene, 'create_client', lambda self, *args: clients.append(args))
    monkeypatch.setattr(module.PipelineScene, 'create_timer', lambda self, *args: timers.append(args))
    monkeypatch.setattr(module.PipelineScene, 'get_logger',
                        lambda self: SimpleNamespace(info=lambda _: None))
    node = module.PipelineScene()
    assert not clients and not timers
    assert '/pipeline/environment_markers' in published
    report = json.loads(published['/arm_lab/scene_status'][-1].data)
    assert report['ready'] is True and report['planning_scene_applied'] is False
    node.apply()  # Explicit no-op in this mode, not a dangling service request.
