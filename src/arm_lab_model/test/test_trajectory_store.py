"""Public storage and replay contract; synthetic single-joint fixture."""
from copy import deepcopy

import pytest

from arm_lab_model.trajectory_store import load_trajectory, save_trajectory


def record():
    return {'schema_version': 1, 'robot': 'test_arm', 'model_sha256': 'a' * 64,
            'scenario_id': 'move', 'joint_names': ['joint1'], 'start_state': [0.0],
            'target': {'frame': 'base', 'position': [1., 0., 0.],
                       'orientation': [0., 0., 0., 1.]},
            'trajectory': {'points': [
                {'time_from_start': 0., 'positions': [0.], 'velocities': [0.],
                 'accelerations': [0.]},
                {'time_from_start': 1., 'positions': [0.5], 'velocities': [0.],
                 'accelerations': [0.]}]},
            'planner': {'name': 'test', 'parameters': {}},
            'collision': {'status': 'clear'}, 'execution': {'status': 'succeeded'},
            'timestamp': '2026-09-25T12:00:00+00:00'}


def test_success_roundtrip_and_no_overwrite(tmp_path):
    source = record()
    before = deepcopy(source)
    first, second = [save_trajectory(source, tmp_path) for _ in range(2)]
    assert first != second
    assert first.parent.name == 'test_arm'
    assert load_trajectory(first) == source == before


@pytest.mark.parametrize('field,value', [
    ('execution', {'status': 'failed'}), ('collision', {'status': 'unknown'}),
    ('robot', '../escape'), ('model_sha256', 'bad'), ('start_state', [float('nan')]),
    ('timestamp', 'yesterday'), ('joint_names', ['joint1', 'joint1'])])
def test_reject_unsafe_or_unusable_record(tmp_path, field, value):
    with pytest.raises(ValueError):
        save_trajectory({**record(), field: value}, tmp_path)


def test_replay_identity_and_limits(tmp_path):
    path = save_trajectory(record(), tmp_path)
    robot = {'name': 'test_arm', 'model_sha256': 'a' * 64, 'joint_names': ['joint1'],
             'limits': {'joint1': {'lower': -1., 'upper': 1., 'velocity': 1.,
                                    'acceleration': 2.}}}
    assert load_trajectory(path, robot)['robot'] == 'test_arm'
    with pytest.raises(ValueError, match='robot'):
        load_trajectory(path, {**robot, 'name': 'other'})
    with pytest.raises(ValueError, match='limit'):
        load_trajectory(path, {**robot, 'limits': {'joint1': {'lower': -0.1, 'upper': 0.1}}})


def test_joint_target_is_reusable(tmp_path):
    value = {**record(), 'target': {'joint_positions': [0.5]}}
    assert load_trajectory(save_trajectory(value, tmp_path)) == value


def test_floating_base_start_state_roundtrip(tmp_path):
    value = {**record(), 'base_start_state': {'qpos': [0., 0., 0.5, 1., 0., 0., 0.],
                                             'qvel': [0.] * 6}}
    assert load_trajectory(save_trajectory(value, tmp_path)) == value


@pytest.mark.parametrize('state', [
    {'qpos': [0.] * 7, 'qvel': [0.] * 6},
    {'qpos': [0., 0., 0., 1., 0., 0., 0.], 'qvel': [0.] * 5},
    {'qpos': [0., 0., 0., 1., 0., 0., float('nan')], 'qvel': [0.] * 6},
    {'qpos': [0., 0., 0., 1., 0., 0., 0.]},
])
def test_invalid_floating_base_start_state_rejected(tmp_path, state):
    with pytest.raises(ValueError):
        save_trajectory({**record(), 'base_start_state': state}, tmp_path)


@pytest.mark.parametrize('change', ['time', 'shape', 'start'])
def test_reject_inconsistent_trajectory(tmp_path, change):
    value = record()
    if change == 'time':
        value['trajectory']['points'][1]['time_from_start'] = 0.
    elif change == 'shape':
        value['trajectory']['points'][1]['positions'] = []
    else:
        value['start_state'] = [1.]
    with pytest.raises(ValueError):
        save_trajectory(value, tmp_path)
