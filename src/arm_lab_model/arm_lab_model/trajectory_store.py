"""Versioned successful trajectories, atomically published and replay-validated."""
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

import numpy as np
import yaml

from .config_contract import choice, fields, identifier, number, read_yaml, text, vector, version


def joint_names(value):
    if not isinstance(value, list) or not value:
        raise ValueError('joint_names: expected nonempty list')
    names = [identifier(item, 'joint_names') for item in value]
    if len(set(names)) != len(names):
        raise ValueError('joint_names: duplicate names')
    return names


def quaternion(value, path):
    values = vector(value, path, 4)
    if not np.isclose(np.linalg.norm(values), 1., atol=1e-6, rtol=0):
        raise ValueError(f'{path}: expected unit quaternion xyzw')
    return values


def validate_trajectory(record):
    """Reject incomplete, failed, nonfinite or structurally inconsistent records."""
    fields(record, ('schema_version', 'robot', 'model_sha256', 'scenario_id',
                    'joint_names', 'start_state', 'target', 'trajectory', 'planner',
                    'collision', 'execution', 'timestamp'), ('benchmark', 'base_start_state'), path='trajectory')
    version(record['schema_version'], 'trajectory')
    identifier(record['robot'], 'robot')
    text(record['scenario_id'], 'scenario_id')
    if not isinstance(record['model_sha256'], str) or not re.fullmatch('[a-f0-9]{64}', record['model_sha256']):
        raise ValueError('model_sha256: expected lowercase SHA256 hex digest')
    try:
        timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
        if timestamp.utcoffset() is None:
            raise ValueError('timezone is required')
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('timestamp: expected ISO8601 with timezone') from exc
    names = joint_names(record['joint_names'])
    start = vector(record['start_state'], 'start_state', len(names))
    if 'base_start_state' in record:
        _validate_base_state(record['base_start_state'])
    _validate_target(record['target'], len(names))
    _validate_points(record['trajectory'], start, len(names))
    fields(record['planner'], ('name', 'parameters'), path='planner')
    text(record['planner']['name'], 'planner.name')
    if not isinstance(record['planner']['parameters'], dict):
        raise ValueError('planner.parameters: expected mapping')
    for key, expected in (('collision', 'clear'), ('execution', 'succeeded')):
        fields(record[key], ('status',), ('details',), path=key)
        choice(record[key]['status'], (expected,), key + '.status')
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('trajectory metadata must be finite JSON-compatible values') from exc
    return deepcopy(record)


def _validate_base_state(state):
    fields(state, ('qpos', 'qvel'), path='base_start_state')
    vector(state['qpos'], 'base_start_state.qpos', 7)
    vector(state['qvel'], 'base_start_state.qvel', 6)
    # MuJoCo free-joint coordinates: xyz followed by quaternion wxyz.
    # Norm validation is independent of quaternion coefficient ordering.
    if not np.isclose(np.linalg.norm(state['qpos'][3:]), 1., atol=1e-6, rtol=0):
        raise ValueError('base_start_state.qpos: expected normalized quaternion wxyz')


def _validate_target(target, count):
    if isinstance(target, dict) and 'joint_positions' in target:
        fields(target, ('joint_positions',), path='target')
        vector(target['joint_positions'], 'target.joint_positions', count)
        return
    fields(target, ('frame', 'position'), ('orientation',), path='target')
    text(target['frame'], 'target.frame')
    vector(target['position'], 'target.position')
    if 'orientation' in target:
        quaternion(target['orientation'], 'target.orientation')


def _validate_points(trajectory, start, count):
    fields(trajectory, ('points',), path='trajectory')
    points = trajectory['points']
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError('trajectory.points: need at least two points')
    previous = -1.
    for i, point in enumerate(points):
        fields(point, ('time_from_start', 'positions', 'velocities', 'accelerations'), path='point')
        time = number(point['time_from_start'], 'time_from_start')
        if time < 0 or time <= previous:
            raise ValueError('time_from_start must be nonnegative and strictly increasing')
        if i == 0 and time != 0:
            raise ValueError('first point must have time_from_start=0')
        for key in ('positions', 'velocities', 'accelerations'):
            vector(point[key], 'point.' + key, count)
        previous = time
    if not np.allclose(points[0]['positions'], start, atol=1e-9, rtol=0):
        raise ValueError('start_state differs from first point')


def save_trajectory(record, directory) -> Path:
    """Save under directory/robot; never overwrite an existing trajectory."""
    checked = validate_trajectory(record)
    folder = Path(directory).resolve() / checked['robot']
    folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink():
        raise ValueError('robot trajectory directory must not be a symbolic link')
    path = folder / ('trajectory_' + uuid4().hex + '.yaml')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=folder, delete=False, suffix='.tmp') as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(checked, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication, fails instead of overwriting.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_trajectory(path, robot=None) -> dict:
    """Load a record; optionally enforce replay robot identity and limits.

    robot = {name, model_sha256, joint_names, limits: {joint: {lower, upper,
    velocity?, acceleration?}}}. Continuous joints omit lower and upper.
    """
    record = validate_trajectory(read_yaml(path))
    if robot is not None:
        _validate_replay(record, robot)
    return record


def _validate_replay(record, robot):
    fields(robot, ('name', 'model_sha256', 'joint_names', 'limits'), path='replay robot')
    if (robot['name'] != record['robot'] or robot['model_sha256'] != record['model_sha256']
            or robot['joint_names'] != record['joint_names']):
        raise ValueError('replay robot identity, model digest or joint order differs')
    if not isinstance(robot['limits'], dict) or set(robot['limits']) != set(record['joint_names']):
        raise ValueError('replay limits must cover every joint')
    for index, name in enumerate(record['joint_names']):
        limits = robot['limits'][name]
        fields(limits, (), ('lower', 'upper', 'velocity', 'acceleration'), path='limits.' + name)
        if ('lower' in limits) != ('upper' in limits):
            raise ValueError('position limits require both lower and upper')
        for key, value in limits.items():
            number(value, 'limit.' + key, positive=key in ('velocity', 'acceleration'))
        if 'lower' in limits and limits['lower'] > limits['upper']:
            raise ValueError('lower limit exceeds upper limit')
        for point in record['trajectory']['points']:
            if 'lower' in limits and not limits['lower'] <= point['positions'][index] <= limits['upper']:
                raise ValueError(f'{name}: position limit exceeded')
            for singular, plural in (('velocity', 'velocities'), ('acceleration', 'accelerations')):
                if singular in limits and abs(point[plural][index]) > limits[singular]:
                    raise ValueError(f'{name}: {singular} limit exceeded')
