"""Explicit SI observation records and external CSV/YAML ingestion."""
from copy import deepcopy
import csv
from pathlib import Path

import numpy as np

from .config_contract import choice, fields, number, read_yaml, text, version
from .trajectory_store import joint_names

CHANNELS = {'joint_position': ('rad', None), 'joint_velocity': ('rad/s', None),
            'joint_acceleration': ('rad/s²', None), 'joint_torque': ('N m', None),
            'tcp_position': ('m', 3), 'tcp_orientation': ('rad', 4),
            'tcp_velocity': ('m/s', 3)}


def validate_observation(record):
    fields(record, ('schema_version', 'robot', 'scenario_id', 'joint_names',
                    'frame', 'units', 'evidence', 'samples'),
           ('model_sha256', 'continuous_joints', 'conditions', 'joint_types', 'end_effector'), path='observation')
    version(record['schema_version'], 'observation')
    for key in ('robot', 'scenario_id', 'frame'):
        text(record[key], key)
    if 'end_effector' in record:
        text(record['end_effector'], 'end_effector')
    choice(record['units'], ('SI',), 'units')
    names = joint_names(record['joint_names'])
    types = record.get('joint_types', ['revolute'] * len(names))
    if not isinstance(types, list) or len(types) != len(names):
        raise ValueError('joint_types must have one entry per joint')
    for kind in types:
        choice(kind, ('revolute', 'continuous', 'prismatic'), 'joint_types')
    fields(record['evidence'], ('kind', 'source'), path='evidence')
    choice(record['evidence']['kind'], ('measured', 'manufacturer', 'synthetic', 'analytical'), 'evidence.kind')
    text(record['evidence']['source'], 'evidence.source')
    continuous = record.get('continuous_joints', [])
    if (not isinstance(continuous, list) or any(not isinstance(name, str) for name in continuous)
            or len(set(continuous)) != len(continuous) or not set(continuous) <= set(names)):
        raise ValueError('continuous_joints must be a unique subset of joint_names')
    if any(types[names.index(name)] == 'prismatic' for name in continuous):
        raise ValueError('continuous_joints cannot contain prismatic joints')
    samples = fields(record['samples'], ('time',), tuple(CHANNELS), path='samples')
    if len(samples) < 2:
        raise ValueError('samples: at least one observation channel required')
    times = _finite_array(samples['time'], 'time')
    if times.ndim != 1 or len(times) < 2 or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError('time: at least two nonnegative strictly increasing samples required')
    for channel in set(samples) - {'time'}:
        values = _finite_array(samples[channel], channel)
        size = CHANNELS[channel][1] or len(names)
        if values.shape != (len(times), size):
            raise ValueError(f'{channel}: expected {len(times)} × {size} values')
        if channel == 'tcp_orientation' and not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-6, rtol=0):
            raise ValueError('tcp_orientation must contain unit quaternions xyzw')
    return deepcopy(record)


def _finite_array(values, path):
    def check(value):
        if isinstance(value, list):
            for item in value:
                check(item)
        else:
            number(value, path)
    if not isinstance(values, list):
        raise ValueError(f'{path}: expected list')
    check(values)
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{path}: expected rectangular numeric array') from exc


def load_observation(path, metadata=None, columns=None):
    """Load YAML or CSV using caller-declared columns and provenance.

    CSV columns example: {time: 'seconds', joint_position: ['q1', ...]}.
    No unit conversion, time reset or frame transform is silently applied.
    """
    path = Path(path)
    if path.suffix.lower() != '.csv':
        if metadata is not None or columns is not None:
            raise ValueError('YAML observations carry their own metadata and samples')
        return validate_observation(read_yaml(path))
    if not isinstance(metadata, dict) or not isinstance(columns, dict) or 'time' not in columns:
        raise ValueError('CSV requires explicit metadata and columns including time')
    fields(columns, ('time',), tuple(CHANNELS), path='columns')
    text(columns['time'], 'columns.time')
    for key, value in columns.items():
        if key != 'time' and (not isinstance(value, list) or not value or any(not isinstance(v, str) for v in value)):
            raise ValueError('channel columns must be nonempty lists of column names')
    try:
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('CSV header names must be unique')
            rows = list(reader)
        samples = {key: ([float(row[value]) for row in rows] if key == 'time' else
                         [[float(row[column]) for column in value] for row in rows])
                   for key, value in columns.items()}
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise ValueError(f'{path}: invalid CSV observations: {exc}') from exc
    return validate_observation({**metadata, 'samples': samples})
