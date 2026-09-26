"""Strict, immutable benchmark evidence catalogue with checked local provenance.

This loader does not turn specification limits or analytical cases into hardware
measurements. No remote resource is fetched during loading.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from types import MappingProxyType

from .config_contract import choice, fields, identifier, number, read_yaml, text, vector, version


KINDS = ('hardware_measurement', 'manufacturer_reference', 'analytical_reference',
         'synthetic_fixture')


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class BenchmarkReference:
    """Catalogue declaration; datasets retain their own evidence classifications."""

    path: Path
    robot_path: Path
    joint_names: tuple[str, ...]
    data: Mapping


def _hash_file(root, entry, path):
    fields(entry, ('path', 'sha256'), path=path)
    file = (root / text(entry['path'], path + '.path')).resolve()
    digest = text(entry['sha256'], path + '.sha256')
    if not re.fullmatch('[a-f0-9]{64}', digest):
        raise ValueError(path + ': expected lowercase SHA256 digest')
    try:
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f'{path}: cannot read provenance file {file}: {exc}') from exc
    if actual != digest:
        raise ValueError(f'{path}: SHA256 mismatch for {file}')


def _sources(data, root):
    sources = data['sources']
    if not isinstance(sources, dict) or not sources:
        raise ValueError('sources: expected nonempty mapping')
    for name, source in sources.items():
        identifier(name, 'sources')
        fields(source, ('kind', 'url', 'revision', 'license', 'retrieved', 'files'),
               path='sources.' + name)
        choice(source['kind'], KINDS, 'sources.kind')
        for key in ('url', 'license', 'retrieved'):
            text(source[key], 'sources.' + key)
        if source['revision'] is not None:
            text(source['revision'], 'sources.revision')
        if not isinstance(source['files'], list):
            raise ValueError('sources.files: expected list')
        for entry in source['files']:
            _hash_file(root, entry, 'sources.' + name)
    return sources


def _limits(data, names, sources):
    limits = data['joint_limits']
    if not isinstance(limits, dict) or set(limits) != set(names):
        raise ValueError('joint_limits: names must match joint_names exactly')
    for name, entry in limits.items():
        fields(entry, ('position_rad', 'velocity_rad_s', 'acceleration_rad_s2',
                       'continuous_torque_nm', 'model_effort_nm',
                       'planning_position_rad', 'source', 'notes'), path=name)
        for key in ('position_rad', 'planning_position_rad'):
            low, high = vector(entry[key], name + '.' + key, size=2)
            if low >= high:
                raise ValueError(name + ': position lower must be less than upper')
        for key in ('velocity_rad_s', 'acceleration_rad_s2', 'continuous_torque_nm',
                    'model_effort_nm'):
            if entry[key] is not None:
                number(entry[key], name + '.' + key, positive=True)
        if entry['source'] not in sources:
            raise ValueError(name + ': unknown source')
        text(entry['notes'], name + '.notes')


def _datasets(data, root, names, sources):
    if not isinstance(data['datasets'], dict):
        raise ValueError('datasets: expected mapping')
    for name, entry in data['datasets'].items():
        identifier(name, 'datasets')
        fields(entry, ('kind', 'source', 'file', 'sha256', 'format', 'scenario',
                       'joint_names', 'channels', 'conditions', 'description'),
               path='datasets.' + name)
        choice(entry['kind'], KINDS, name + '.kind')
        if entry['source'] not in sources:
            raise ValueError(name + ': unknown source')
        if tuple(entry['joint_names']) != names:
            raise ValueError(name + ': dataset joint order must match joint_names')
        for key in ('format', 'scenario', 'description'):
            text(entry[key], name + '.' + key)
        _hash_file(root, {'path': entry['file'], 'sha256': entry['sha256']}, name)
        if not isinstance(entry['channels'], dict) or not entry['channels']:
            raise ValueError(name + ': expected nonempty channels mapping')
        for channel, info in entry['channels'].items():
            identifier(channel, name + '.channels')
            fields(info, ('unit', 'frame', 'uncertainty'), path=name + '.' + channel)
            text(info['unit'], name + '.unit')
            if info['frame'] is not None:
                text(info['frame'], name + '.frame')
            if info['uncertainty'] is not None:
                number(info['uncertainty'], name + '.uncertainty')
                if info['uncertainty'] < 0:
                    raise ValueError(name + ': uncertainty must be nonnegative')
        if not isinstance(entry['conditions'], dict):
            raise ValueError(name + ': expected conditions mapping')
        if entry['kind'] == 'hardware_measurement':
            if sources[entry['source']]['kind'] != 'hardware_measurement':
                raise ValueError(name + ': hardware measurements need a hardware source')
            required = ('calibration', 'controller', 'payload', 'gravity',
                        'timestamp_convention', 'acquisition')
            fields(entry['conditions'], required, ('notes',), path=name + '.conditions')
            for key in required:
                if entry['conditions'][key] is None:
                    raise ValueError(name + ': hardware acquisition conditions missing ' + key)


def _physical(data, sources):
    payload = fields(data['payload'], ('maximum_mass_kg', 'source', 'com_envelope',
                                     'inertia_envelope', 'notes'), path='payload')
    number(payload['maximum_mass_kg'], 'payload.maximum_mass_kg', positive=True)
    text(payload['notes'], 'payload.notes')
    kin = fields(data['kinematics'], ('convention', 'source', 'base_frame',
                                     'tcp_frame', 'rows', 'calibration'), path='kinematics')
    choice(kin['convention'], ('standard_dh',), 'kinematics.convention')
    for key in ('base_frame', 'tcp_frame'):
        identifier(kin[key], 'kinematics.' + key)
    if not isinstance(kin['rows'], list) or len(kin['rows']) != len(data['joint_names']):
        raise ValueError('kinematics.rows: expected one DH row per joint')
    for row in kin['rows']:
        fields(row, ('a_m', 'd_m', 'alpha_rad', 'theta_offset_rad'), path='kinematics.row')
        for key, value in row.items():
            number(value, 'kinematics.' + key)
    props = fields(data['physical_properties'], ('source', 'robot_description',
                   'mass_kg_reference', 'mass_kg_model', 'inertia_order', 'inertia_units',
                   'com_units', 'notes'), path='physical_properties')
    for key in ('mass_kg_reference', 'mass_kg_model'):
        number(props[key], 'physical_properties.' + key, positive=True)
    if props['robot_description'] != data['robot_description']:
        raise ValueError('physical_properties: robot_description must use authoritative robot')
    if props['inertia_order'] != ['ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz']:
        raise ValueError('physical_properties: incorrect inertia_order')
    choice(props['inertia_units'], ('kg*m^2',), 'inertia_units')
    choice(props['com_units'], ('m',), 'com_units')
    for entry in (payload, kin, props):
        if entry['source'] not in sources:
            raise ValueError('physical parameters: unknown source')


def load_benchmark_reference(path: str | Path) -> BenchmarkReference:
    """Validate catalogue and all local SHA256 evidence, without starting a run."""
    path = Path(path).resolve()
    data = read_yaml(path)
    fields(data, ('schema_version', 'robot', 'robot_description', 'joint_names',
                  'joint_limits', 'payload', 'kinematics', 'physical_properties',
                  'sources', 'datasets', 'limitations'), path='benchmark')
    version(data['schema_version'], 'benchmark')
    robot = fields(data['robot'], ('name', 'model', 'manufacturer', 'description'), path='robot')
    identifier(robot['name'], 'robot.name')
    for key in ('model', 'manufacturer', 'description'):
        text(robot[key], 'robot.' + key)
    names = data['joint_names']
    if not isinstance(names, list) or not names:
        raise ValueError('joint_names: expected nonempty list')
    for name in names:
        identifier(name, 'joint_names')
    if len(set(names)) != len(names):
        raise ValueError('joint_names: duplicate joint')
    robot_path = (path.parent / text(data['robot_description'], 'robot_description')).resolve()
    description = read_yaml(robot_path)
    if not isinstance(description, dict) or description.get('name') != robot['name']:
        raise ValueError('robot_description: robot identity mismatch')
    sources = _sources(data, path.parent)
    _limits(data, names, sources)
    _physical(data, sources)
    _datasets(data, path.parent, tuple(names), sources)
    if not isinstance(data['limitations'], list):
        raise ValueError('limitations: expected list')
    for limitation in data['limitations']:
        text(limitation, 'limitations')
    return BenchmarkReference(path, robot_path, tuple(names), _freeze(data))


def _case_error(robot, case, fixture):
    import math
    import numpy as np
    from .physical_robot import forward_tree
    fields(case, ('name', 'joint_position', 'tcp_position', 'tcp_orientation'), path='case')
    text(case['name'], 'case.name')
    q = vector(case['joint_position'], 'case.joint_position', len(fixture['joint_names']))
    position = np.asarray(vector(case['tcp_position'], 'case.tcp_position'))
    x, y, z, w = vector(case['tcp_orientation'], 'case.tcp_orientation', 4)
    if abs(x*x + y*y + z*z + w*w - 1) > 1e-9:
        raise ValueError('case.tcp_orientation: expected unit XYZW quaternion')
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    frames, _ = forward_tree(robot, dict(zip(fixture['joint_names'], q)))
    if fixture['frame'] not in frames or fixture['tcp_frame'] not in frames:
        raise ValueError('static FK reference: frame or TCP frame absent from robot')
    base_r, base_p = frames[fixture['frame']]
    tcp_r, tcp_p = frames[fixture['tcp_frame']]
    relative = rotation.T @ base_r.T @ tcp_r
    skew = np.array([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0],
                     relative[1, 0]-relative[0, 1]]) / 2
    angle = math.atan2(float(np.linalg.norm(skew)), float((np.trace(relative)-1)/2))
    return {'name': case['name'],
            'position_error_m': float(np.linalg.norm(base_r.T @ (tcp_p-base_p) - position)),
            'orientation_error_rad': angle}


def check_reference_model(robot, benchmark) -> dict:
    """Compare a resolved tree to independent static analytical FK cases.

    ``robot`` is the result of physical_robot.resolve_robot(project). ``benchmark``
    is a catalogue path or BenchmarkReference. This does not claim hardware testing.
    """
    reference = (benchmark if isinstance(benchmark, BenchmarkReference)
                 else load_benchmark_reference(benchmark))
    if robot['name'] != reference.data['robot']['name']:
        raise ValueError('reference model: robot identity mismatch')
    results = []
    for dataset in reference.data['datasets'].values():
        if dataset['format'] != 'static_fk_cases_v1':
            continue
        if dataset['kind'] != 'analytical_reference':
            raise ValueError('static FK cases require analytical_reference evidence')
        fixture = read_yaml(reference.path.parent / dataset['file'])
        fields(fixture, ('schema_version', 'robot', 'frame', 'tcp_frame', 'joint_names',
                         'evidence', 'cases'), path='static_fk')
        version(fixture['schema_version'], 'static_fk')
        if fixture['robot'] != robot['name'] or tuple(fixture['joint_names']) != reference.joint_names:
            raise ValueError('static FK cases: robot identity or joint order mismatch')
        if not isinstance(fixture['cases'], list) or not fixture['cases']:
            raise ValueError('static FK cases: expected nonempty cases list')
        results.extend(_case_error(robot, case, fixture) for case in fixture['cases'])
    mass = sum(link['inertial']['mass'] for link in robot['links'])
    expected_mass = reference.data['physical_properties']['mass_kg_reference']
    position = max((case['position_error_m'] for case in results), default=None)
    orientation = max((case['orientation_error_rad'] for case in results), default=None)
    tolerance = 1e-8  # Nominal model numerical consistency, not hardware tolerance.
    return {'schema_version': 1, 'robot': robot['name'],
            'evidence_kind': 'analytical_reference', 'hardware_validation': False,
            'passed': bool(results) and position <= tolerance and orientation <= tolerance,
            'scope': 'Nominal static kinematics only; mass discrepancy reported separately',
            'case_count': len(results), 'cases': results,
            'max_position_error_m': position, 'max_orientation_error_rad': orientation,
            'numerical_tolerance': {'position_m': tolerance, 'orientation_rad': tolerance},
            'mass': {'model_kg': mass, 'manufacturer_kg': expected_mass,
                     'difference_kg': mass-expected_mass},
            'limitations': list(reference.data['limitations'])}
