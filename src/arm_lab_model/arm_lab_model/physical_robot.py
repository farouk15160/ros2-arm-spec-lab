"""Resolve a single physical robot tree, with explicit provenance and unknowns."""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import MeasuredInertial
from .mesh_physics import (finite_vector, geometry_properties, mesh_scale, positive,
                           tensor6, tensor_matrix, transform)


def plain(value):
    """Copy immutable configuration recursively; never change caller data."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def validate_inertial(data, name):
    if data is None:
        raise ValueError(f'{name}.inertial: missing mass, COM and inertia')
    if data.get('mass') == 0:
        com = finite_vector(data.get('com'), name=name+'.com')
        inertia = finite_vector(data.get('inertia'), 6, name+'.inertia')
        if np.any(com) or np.any(inertia):
            raise ValueError(f'{name}: positive mass required unless massless frame has zero COM and tensor')
        return
    MeasuredInertial.from_dict({key: data.get(key) for key in ('mass', 'com', 'inertia')}, name)


def _geometry(geometry, directory):
    result = plain(geometry)
    if result.get('type') != 'mesh':
        geometry_properties(result, 1.0)  # Validate dimensions/origin even with manual mass.
    if result.get('type') == 'mesh':
        path = (directory / result['path']).resolve()
        if not path.is_file():
            raise ValueError(f'geometry.path: missing mesh {path}')
        result = {**result, 'path': str(path), 'scale': mesh_scale(result).tolist(), 'units':'m',
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def _resolve_link(link, directory, materials):
    geometry = _geometry(link['geometry'], directory) if link.get('geometry') else None
    entry = link.get('inertial')
    if entry is None:
        raise ValueError(f"links.{link['name']}.inertial: missing physical parameters")
    material = link.get('material')
    if isinstance(material, str):
        if material not in materials:
            raise ValueError(f"links.{link['name']}.material: unknown material {material}")
        material = {'name': material, **materials[material]}
    properties = None
    if entry.get('mode', 'manual') == 'auto':
        if geometry is None or not material or 'density' not in material:
            raise ValueError(f"links.{link['name']}: auto inertia requires geometry and material density")
        properties = geometry_properties(geometry, material['density'])
        inertial = {'mass':properties['mass'], 'com':properties['com'],
                    'inertia':tensor6(properties['inertia']),
                    'source':'Homogeneous solid geometry + configured material density'}
    elif entry.get('mode', 'manual') == 'manual':
        inertial = {key: entry.get(key) for key in ('mass','com','inertia','source')}
        # Complete measured CAD/assembly values take precedence over density estimates.
        if not inertial['source']:
            raise ValueError(f"links.{link['name']}.inertial.source: required for manual values")
    else:
        raise ValueError(f"links.{link['name']}.inertial.mode: expected auto or manual")
    validate_inertial(inertial, link['name'])
    return {**link, 'inertial':inertial, **({'geometry':geometry} if geometry else {}),
            'physical': {'volume':properties['volume'] if properties else None,
                         'surface_area':properties['surface_area'] if properties else None,
                         'material':material, 'mode':entry.get('mode','manual')}}


def resolve_robot(project):
    """Return fully specified physical tree; no synthetic replacement for unknown data.

    Mesh paths are relative to the robot YAML and normalized to absolute SI assets.
    Zero inertials are allowed for fixed coordinate frames only.
    """
    if project.legacy_source:
        from .robot_legacy import legacy_robot
        robot = legacy_robot(project.legacy_source)
    else:
        robot = plain(project.sections['robot'])
    directory = project.files['robot'].parent
    materials = plain(project.sections.get('materials', {}).get('materials', {}))
    links = [_resolve_link(link, directory, materials) for link in robot['links']]
    parents = {joint['child']:joint for joint in robot['joints']}
    for link in links:
        if link['inertial']['mass'] == 0:
            joint = parents.get(link['name'])
            if link.get('geometry') or (joint and joint['type'] != 'fixed') or (
                    not joint and robot['base']['type'] == 'floating'):
                raise ValueError(f"{link['name']}: massless inertial only allowed for fixed coordinate frames")
    return {**robot, 'links':links}


def robot_fingerprint(robot):
    """Stable content identity, including referenced mesh bytes and full inertials."""
    data = plain(robot)
    for link in data['links']:
        for geom in [link.get('geometry')] + link.get('visuals', []) + link.get('collisions', []):
            if geom and geom['type'] == 'mesh':
                geom['sha256'] = hashlib.sha256(Path(geom['path']).read_bytes()).hexdigest()
                geom.pop('path', None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def forward_tree(robot, q=None):
    """World transforms and joint axes; floating base defaults to configured pose.

    q maps joint name to radians/metres. A floating base pose can be supplied as
    q['__base__']={xyz:[...],rpy:[...]}; it is not an actuated joint.
    """
    moving = [j for j in robot['joints'] if j['type'] != 'fixed']
    if q is None:
        q = {}
    elif not isinstance(q, Mapping):
        values = finite_vector(q, len(moving), 'q')
        q = dict(zip((j['name'] for j in moving), values))
    unknown = set(q) - {j['name'] for j in moving} - {'__base__'}
    if unknown:
        raise ValueError(f'q: unknown joints {sorted(unknown)}')
    transforms = {robot['base']['link']: transform(q.get('__base__', robot['base'].get('origin')))}
    joints = {}
    pending = list(robot['joints'])
    while pending:
        ready = [j for j in pending if j['parent'] in transforms]
        if not ready:
            raise ValueError('robot joints are disconnected or cyclic')
        for joint in ready:
            parent_r, parent_p = transforms[joint['parent']]
            local_r, local_p = transform(joint['origin'])
            rot, pos = parent_r @ local_r, parent_p + parent_r @ local_p
            kind = joint['type']
            if kind != 'fixed':
                axis = finite_vector(joint['axis'], name='joint.axis')
                value = float(q.get(joint['name'], 0))
                if not np.isfinite(value):
                    raise ValueError('q: expected finite joint values')
                joints[joint['name']] = (pos.copy(), rot @ axis)
                if kind == 'prismatic':
                    pos = pos + rot @ axis * value
                else:
                    x, y, z = axis
                    cross = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                    rot = rot @ (np.eye(3) + np.sin(value)*cross + (1-np.cos(value))*(cross@cross))
            transforms[joint['child']] = (rot, pos)
        pending = [j for j in pending if j not in ready]
    return transforms, joints


def physical_report(robot, q=None, gravity=(0, 0, -9.81), payload=None):
    """Mass properties and actuator effort required to hold q against gravity.

    Floating-base effort assumes externally supported base; unactuated base support
    wrench is reported. Contact equilibrium is not solved. Payload is a point mass
    for static loads only: {link, mass, com}; no invented dynamic payload inertia.
    """
    gravity = finite_vector(gravity, name='gravity')
    transforms, axes = forward_tree(robot, q)
    links = []
    for link in robot['links']:
        inertial = link.get('inertial')
        validate_inertial(inertial, link['name'])
        rot, pos = transforms[link['name']]
        com = pos + rot @ np.array(inertial['com'])
        links.append({'name':link['name'], **link.get('physical', {}), **inertial,
                      'world_com':com.tolist(), 'weight':float(inertial['mass']*np.linalg.norm(gravity))})
    bodies = [(link['name'], link['mass'], np.array(link['world_com'])) for link in links]
    if payload is not None:
        if payload.get('link') not in transforms:
            raise ValueError('payload.link: expected robot link')
        mass = positive(payload.get('mass'), 'payload.mass')
        rot, pos = transforms[payload['link']]
        bodies = bodies + [(payload['link'], mass, pos + rot @ finite_vector(payload.get('com'), name='payload.com'))]
    total = sum(mass for _, mass, _ in bodies)
    if total <= 0:
        raise ValueError('robot: requires positive total mass')
    center = sum(mass*com for _, mass, com in bodies)/total
    parents = {j['child']:j['parent'] for j in robot['joints']}
    def downstream(link, ancestor):
        while link != ancestor and link in parents:
            link = parents[link]
        return link == ancestor
    efforts = {}
    for joint in robot['joints']:
        if joint['type'] == 'fixed':
            continue
        point, axis = axes[joint['name']]
        loads = [(mass, com) for name, mass, com in bodies if downstream(name, joint['child'])]
        effort = sum(np.dot(axis, mass*gravity if joint['type'] == 'prismatic'
                            else np.cross(com-point, mass*gravity)) for mass, com in loads)
        efforts[joint['name']] = float(-effort)
    return {'robot':robot['name'], 'links':links, 'total_mass':float(total),
            'weight':float(total*np.linalg.norm(gravity)), 'center_of_mass':center.tolist(),
            'joint_static_effort':efforts, 'payload':plain(payload),
            'base_support_force':(-total*gravity).tolist(),
            'base_support_moment':(-np.cross(center-transforms[robot['base']['link']][1], total*gravity)).tolist(),
            'assumptions':['Rigid links; no friction or structural compliance.',
                           'Static base supported externally; no contact equilibrium solution.'],
            'dynamic_torque':'Use MuJoCo inverse dynamics with full q, velocity and acceleration.'}
