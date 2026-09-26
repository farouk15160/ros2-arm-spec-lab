"""Planning-scene readiness follows actual service outcomes, not publication."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_scene_waits_retries_and_only_becomes_ready_after_success(monkeypatch):
    pytest.importorskip('rclpy')
    from rclpy.task import Future
    from moveit_msgs.srv import ApplyPlanningScene
    import arm_lab_gui.pipeline_scene as module

    project = Path(__file__).parents[2] / 'arm_lab_model/config/pipeline/project_ur5e_perception.yaml'
    published, requests, futures, canceled, errors = {}, [], [], [], []
    available = [False]

    def call(request):
        future = Future()
        requests.append(request)
        futures.append(future)
        return future

    # Only the ROS middleware boundary is replaced. Project loading, geometry,
    # messages and future completion are real.
    monkeypatch.setattr(module.Node, '__init__', lambda *_: None)
    monkeypatch.setattr(module.PipelineScene, 'declare_parameter', lambda self, name, default:
                        SimpleNamespace(value=str(project) if name == 'project_file' else default))
    monkeypatch.setattr(module.PipelineScene, 'create_publisher', lambda self, cls, topic, qos:
                        SimpleNamespace(publish=lambda msg: published.setdefault(topic, []).append(msg)))
    monkeypatch.setattr(module.PipelineScene, 'create_client', lambda self, cls, name:
                        SimpleNamespace(service_is_ready=lambda: available[0], call_async=call))
    monkeypatch.setattr(module.PipelineScene, 'create_timer', lambda self, period, callback:
                        SimpleNamespace(cancel=lambda: canceled.append(True)))
    monkeypatch.setattr(module.PipelineScene, 'get_logger', lambda self:
                        SimpleNamespace(info=lambda _: None, error=errors.append))
    node = module.PipelineScene()

    def status():
        return json.loads(published['/arm_lab/scene_status'][-1].data)

    node.apply()
    assert not requests and status()['ready'] is False
    available[0] = True
    node.apply()
    assert {obj.id for obj in requests[0].scene.world.collision_objects} == {'workbench', 'fixture'}
    node.apply()
    assert len(requests) == 1 and status()['ready'] is False

    futures[0].set_result(ApplyPlanningScene.Response(success=False))
    node.apply()
    assert len(requests) == 2 and status()['ready'] is False
    futures[1].set_exception(RuntimeError('service disconnected'))
    node.apply()
    assert len(requests) == 3 and status()['ready'] is False
    assert len(errors) == 2

    futures[2].set_result(ApplyPlanningScene.Response(success=True))
    node.apply()
    assert status() == {'ready': True, 'planning_scene_applied': True}
    assert canceled == [True]
