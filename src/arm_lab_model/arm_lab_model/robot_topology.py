"""Validate a general body tree without interpreting it as a serial arm."""
from .config_contract import choice, fields, identifier, number, text, vector, version


def _named(items, path):
    if not isinstance(items, list) or not items:
        raise ValueError(f'{path}: expected nonempty list')
    names = tuple(identifier(item.get('name'), path + '.name')
                  if isinstance(item, dict) else identifier(None, path) for item in items)
    if len(set(names)) != len(names):
        raise ValueError(f'{path}: duplicate names')
    return names


def _link(entry):
    path = 'links.' + entry['name']
    fields(entry, ('name', 'inertial'), ('geometry', 'material'), path=path)
    if 'geometry' in entry:
        from .mesh_physics import geometry_properties, mesh_scale, transform
        geometry = entry['geometry']
        if not isinstance(geometry, dict):
            raise ValueError(path + '.geometry: expected mapping')
        choice(geometry.get('type'), ('box', 'sphere', 'cylinder', 'mesh'), path + '.geometry.type')
        transform(geometry.get('origin'))
        if geometry['type'] == 'mesh':
            text(geometry.get('path'), path + '.geometry.path')
            mesh_scale(geometry)
        else:
            geometry_properties(geometry, 1.0)
    if 'material' in entry:
        material = entry['material']
        if isinstance(material, str):
            identifier(material, path + '.material')
        else:
            fields(material, ('name', 'density'), ('source',), path=path + '.material')
            text(material['name'], path + '.material.name')
            number(material['density'], path + '.material.density', positive=True)
    if entry['inertial'] is None:
        return (path + '.inertial',)
    if not isinstance(entry['inertial'], dict):
        raise ValueError(path + '.inertial: expected mapping')
    if entry['inertial'].get('mode') == 'auto':
        fields(entry['inertial'], ('mode',), path=path + '.inertial')
        return tuple(path + '.' + key for key in ('geometry', 'material') if key not in entry)
    data = fields(entry['inertial'], ('mass', 'com', 'inertia', 'source'), ('mode',), path=path + '.inertial')
    if 'mode' in data:
        choice(data['mode'], ('manual',), path + '.inertial.mode')
    text(data['source'], path + '.inertial.source')
    number(data['mass'], path + '.inertial.mass')
    vector(data['com'], path + '.inertial.com')
    vector(data['inertia'], path + '.inertial.inertia', size=6)
    from .physical_robot import validate_inertial
    validate_inertial(data, path)
    return ()


def _limits(entry, path):
    keys = ('velocity', 'acceleration', 'effort')
    bounded = entry['type'] in ('revolute', 'prismatic')
    required = ('lower', 'upper') + keys if bounded else keys
    limits = fields(entry['limits'], required, path=path + '.limits')
    for key, value in limits.items():
        if value is not None:
            number(value, path + '.limits.' + key, positive=key in keys)
    if bounded and limits['lower'] is not None and limits['upper'] is not None:
        if limits['lower'] >= limits['upper']:
            raise ValueError(f'{path}.limits: lower must be less than upper')
    return tuple(path + '.limits.' + key for key, value in limits.items() if value is None)


def _joint(entry, links):
    path = 'joints.' + entry['name']
    choice(entry.get('type'), ('fixed', 'revolute', 'continuous', 'prismatic'), path + '.type')
    moving = entry['type'] != 'fixed'
    keys = ('name', 'type', 'parent', 'child', 'origin') + (('axis', 'limits') if moving else ())
    fields(entry, keys, path=path)
    for key in ('parent', 'child'):
        identifier(entry[key], path + '.' + key)
        if entry[key] not in links:
            raise ValueError(f'{path}.{key}: unknown link {entry[key]}')
    origin = fields(entry['origin'], ('xyz', 'rpy'), path=path + '.origin')
    vector(origin['xyz'], path + '.origin.xyz')
    vector(origin['rpy'], path + '.origin.rpy')
    if moving:
        axis = vector(entry['axis'], path + '.axis')
        if abs(sum(x * x for x in axis) - 1) > 1e-6:
            raise ValueError(f'{path}.axis: expected unit vector')
        return _limits(entry, path)
    return ()


def _connected(joints, links, root):
    children = tuple(entry['child'] for entry in joints)
    if len(set(children)) != len(children):
        raise ValueError('robot.joints: a link cannot have multiple parents')
    if set(links) - set(children) != {root}:
        raise ValueError('robot.base.link: expected the single root of a connected tree')
    parents = {entry['child']: entry['parent'] for entry in joints}
    for link in links:
        seen = frozenset()
        node = link
        while node != root:
            if node in seen:
                raise ValueError('robot.joints: cycle or disconnected tree')
            seen = seen | {node}
            node = parents[node]


def validate_robot(robot):
    """Return joint names, base velocity DOF and explicit missing physical inputs."""
    fields(robot, ('schema_version', 'format', 'name', 'family', 'source',
                   'base', 'links', 'joints', 'end_effectors'), path='robot')
    version(robot['schema_version'], 'robot')
    choice(robot['format'], ('tree',), 'robot.format')
    identifier(robot['name'], 'robot.name')
    text(robot['family'], 'robot.family')
    text(robot['source'], 'robot.source')
    base = fields(robot['base'], ('link', 'type'), ('origin',), path='robot.base')
    if 'origin' in base:
        origin = fields(base['origin'], ('xyz', 'rpy'), path='robot.base.origin')
        vector(origin['xyz'], 'robot.base.origin.xyz')
        vector(origin['rpy'], 'robot.base.origin.rpy')
    choice(base['type'], ('fixed', 'floating'), 'robot.base.type')
    identifier(base['link'], 'robot.base.link')
    links = _named(robot['links'], 'robot.links')
    _named(robot['joints'], 'robot.joints')
    missing = tuple(item for link in robot['links'] for item in _link(link))
    missing += tuple(item for joint in robot['joints'] for item in _joint(joint, links))
    _connected(robot['joints'], links, base['link'])
    ends = robot['end_effectors']
    if not isinstance(ends, dict):
        raise ValueError('robot.end_effectors: expected name-to-link mapping')
    for name, link in ends.items():
        identifier(name, 'robot.end_effectors')
        identifier(link, 'robot.end_effectors.' + name)
        if link not in links:
            raise ValueError(f'robot.end_effectors.{name}: unknown link {link}')
    names = tuple(j['name'] for j in robot['joints'] if j['type'] != 'fixed')
    return names, 6 if base['type'] == 'floating' else 0, missing
