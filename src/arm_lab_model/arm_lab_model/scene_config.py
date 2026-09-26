"""Validated world objects and cameras shared by physics and planning exporters."""
from collections.abc import Mapping
import math
from pathlib import Path

from .config_contract import choice, fields, identifier, number, text, vector, version


def mutable(value):
    """Return fresh JSON-like data from an immutable project configuration."""
    if isinstance(value, Mapping):
        return {key: mutable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [mutable(item) for item in value]
    return value


def _header(data, section, key):
    fields(data, ('schema_version', 'enabled'), (key,), path=section)
    version(data['schema_version'], section)
    if type(data['enabled']) is not bool:
        raise ValueError(section + '.enabled: expected boolean')
    if data['enabled'] and key not in data:
        raise ValueError(section + '.' + key + ': required when enabled')


def _pose(pose, path):
    fields(pose, ('xyz', 'rpy'), path=path)
    vector(pose['xyz'], path + '.xyz')
    vector(pose['rpy'], path + '.rpy')


def validate_geometry(geometry, path='geometry'):
    """SI primitive dimensions or STL path plus explicit unit scaling."""
    if not isinstance(geometry, dict):
        raise ValueError(path + ': expected mapping')
    choice(geometry.get('type'), ('box', 'sphere', 'cylinder', 'mesh'), path + '.type')
    kind = geometry['type']
    keys = {'box': ('size',), 'sphere': ('radius',), 'cylinder': ('radius', 'length'),
            'mesh': ('path', 'scale', 'units')}[kind]
    fields(geometry, ('type',) + keys, path=path)
    for key in keys:
        if key == 'units':
            choice(geometry[key], ('m', 'mm'), path + '.units')
        elif key == 'path':
            if Path(text(geometry[key], path + '.path')).suffix.lower() != '.stl':
                raise ValueError(path + '.path: only STL meshes are supported')
        elif key in ('size', 'scale'):
            if any(x <= 0 for x in vector(geometry[key], path + '.' + key)):
                raise ValueError(path + '.' + key + ': expected positive values')
        else:
            number(geometry[key], path + '.' + key, positive=True)


def validate_environment(data):
    """Reject unsupported frames and under-specified dynamic bodies."""
    _header(data, 'environment', 'objects')
    objects = data.get('objects', [])
    if not isinstance(objects, list):
        raise ValueError('environment.objects: expected list')
    names = set()
    for obj in objects:
        fields(obj, ('name', 'geometry', 'pose', 'frame', 'static', 'mass'), path='environment.object')
        name = identifier(obj['name'], 'environment.object.name')
        if name in names:
            raise ValueError('environment.objects: duplicate name ' + name)
        names = names | {name}
        validate_geometry(obj['geometry'], name + '.geometry')
        _pose(obj['pose'], name + '.pose')
        choice(obj['frame'], ('world',), name + '.frame')
        if type(obj['static']) is not bool:
            raise ValueError(name + '.static: expected boolean')
        if not obj['static']:
            number(obj['mass'], name + '.mass', positive=True)
        elif obj['mass'] is not None:
            number(obj['mass'], name + '.mass', positive=True)


def resolve_environment(project):
    data = mutable(project.sections.get('environment', {'schema_version': 1, 'enabled': False}))
    validate_environment(data)
    if not data['enabled']:
        return ()
    result = []
    for obj in data['objects']:
        geometry = obj['geometry']
        if geometry['type'] == 'mesh':
            path = (project.files['environment'].parent / geometry['path']).resolve()
            if not path.is_file():
                raise ValueError(f'{obj["name"]}: missing STL file {path}')
            geometry = {**geometry, 'path': str(path)}
        result.append({**obj, 'geometry': geometry})
    return tuple(result)


def quaternion_xyzw(rpy):
    roll, pitch, yaw = (float(x) / 2 for x in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]


def collision_objects(objects):
    """Pure collision records; meshes contain scaled vertices and indexed triangles."""
    result = []
    for obj in objects:
        if not obj['static']:
            raise ValueError(f'{obj["name"]}: dynamic MoveIt collision objects require live pose tracking; only static environments are supported')
        geo = obj['geometry']
        record = {'id': obj['name'], 'frame_id': obj['frame'], 'operation': 'ADD',
                  'pose': {'position': list(obj['pose']['xyz']),
                           'orientation_xyzw': quaternion_xyzw(obj['pose']['rpy'])}}
        if geo['type'] == 'mesh':
            from .mesh_physics import mesh_scale, read_stl
            triangles = read_stl(geo['path']) * mesh_scale(geo)
            record = {**record, 'mesh': {'vertices': triangles.reshape(-1, 3).tolist(),
                                       'triangles': [[3*i, 3*i+1, 3*i+2] for i in range(len(triangles))]}}
        else:
            dimensions = {'box': geo.get('size'), 'sphere': [geo.get('radius')],
                          'cylinder': [geo.get('length'), geo.get('radius')]}[geo['type']]
            record = {**record, 'primitive': {'type': geo['type'], 'dimensions': dimensions}}
        result.append(record)
    return tuple(result)


def validate_sensors(data):
    _header(data, 'sensors', 'cameras')
    cameras = data.get('cameras', {})
    if not isinstance(cameras, dict):
        raise ValueError('sensors.cameras: expected mapping')
    for name, camera in cameras.items():
        identifier(name, 'sensors.camera.name')
        fields(camera, ('type', 'model', 'parent_link', 'transform', 'resolution',
                        'frame_rate_hz', 'intrinsics', 'clip', 'noise_stddev'),
               ('fov_y_deg',), path=name)
        choice(camera['type'], ('rgb', 'depth', 'rgbd'), name + '.type')
        text(camera['model'], name + '.model')
        identifier(camera['parent_link'], name + '.parent_link')
        _pose(camera['transform'], name + '.transform')
        resolution = camera['resolution']
        if not isinstance(resolution, list) or len(resolution) != 2 or any(type(v) is not int or v <= 0 for v in resolution):
            raise ValueError(name + '.resolution: expected two positive integers')
        number(camera['frame_rate_hz'], name + '.frame_rate_hz', positive=True)
        intr = fields(camera['intrinsics'], ('fx', 'fy', 'cx', 'cy'), path=name + '.intrinsics')
        for key, value in intr.items():
            number(value, name + '.intrinsics.' + key, positive=key in ('fx', 'fy'))
        if not (0 <= intr['cx'] < resolution[0] and 0 <= intr['cy'] < resolution[1]):
            raise ValueError(name + '.intrinsics: principal point must lie inside image')
        near, far = vector(camera['clip'], name + '.clip', size=2)
        if not 0 < near < far:
            raise ValueError(name + '.clip: expected 0 < near < far')
        if number(camera['noise_stddev'], name + '.noise_stddev') < 0:
            raise ValueError(name + '.noise_stddev: expected nonnegative metres')
        if 'fov_y_deg' in camera:
            expected = math.degrees(2 * math.atan(resolution[1] / (2 * intr['fy'])))
            if abs(number(camera['fov_y_deg'], name + '.fov_y_deg') - expected) > .1:
                raise ValueError(name + '.fov_y_deg: inconsistent with resolution and fy')


def resolve_sensors(project):
    data = mutable(project.sections.get('sensors', {'schema_version': 1, 'enabled': False}))
    validate_sensors(data)
    if not data['enabled']:
        return ()
    robot = project.sections['robot']
    if robot['format'] != 'tree':
        raise ValueError('sensors: requires the global tree robot description')
    links = {link['name'] for link in robot['links']}
    result = tuple({**camera, 'name': name, 'optical_frame': name + '_optical_frame'}
                   for name, camera in data['cameras'].items())
    for camera in result:
        if camera['parent_link'] not in links:
            raise ValueError(camera['name'] + '.parent_link: unknown robot link')
        if camera['name'] + '_link' in links or camera['optical_frame'] in links:
            raise ValueError(camera['name'] + ': generated camera frame collides with robot link')
    return result
