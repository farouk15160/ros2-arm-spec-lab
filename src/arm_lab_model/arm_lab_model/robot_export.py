"""Export one resolved physical tree to URDF and MuJoCo MJCF."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .mesh_physics import (finite_vector, mesh_scale, positive, read_stl, rotation,
                           transform)
from .physical_robot import validate_inertial


def _fmt(values):
    return ' '.join(f'{float(value):.15g}' for value in values)


def _element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {key:str(value) for key,value in attrs.items()})


def _xml(root):
    ET.indent(root, space='  ')
    return ET.tostring(root, encoding='unicode') + '\n'


def _origin(parent, origin=None):
    rot, pos = transform(origin)
    _element(parent, 'origin', xyz=_fmt(pos), rpy=_fmt((origin or {}).get('rpy', [0,0,0])))


def _geometry_urdf(parent, geometry):
    _origin(parent, geometry.get('origin'))
    geom = _element(parent, 'geometry')
    kind = geometry['type']
    if kind == 'box':
        _element(geom, 'box', size=_fmt(geometry['size']))
    elif kind == 'mesh':
        _element(geom, 'mesh', filename=Path(geometry['path']).resolve().as_uri(),
                 scale=_fmt(mesh_scale(geometry)))
    elif kind in ('sphere','cylinder'):
        attrs = {'radius':geometry['radius']}
        if kind == 'cylinder':
            attrs['length'] = geometry['length']
        _element(geom, kind, **attrs)
    else:
        raise ValueError(f'Unsupported geometry: {kind}')


def _link_geometries(link, kind):
    explicit = link.get('visuals' if kind == 'visual' else 'collisions')
    if explicit is not None:
        return explicit
    return [link['geometry']] if link.get('geometry') else []


def _control(robot, root, hardware_plugin):
    control = _element(root, 'ros2_control', name=robot['name']+'_system', type='system')
    hardware = _element(control, 'hardware')
    _element(hardware, 'plugin').text = hardware_plugin
    for joint in robot['joints']:
        if joint['type'] == 'fixed':
            continue
        item = _element(control, 'joint', name=joint['name'])
        _element(item, 'command_interface', name='position')
        state = _element(item, 'state_interface', name='position')
        _element(state, 'param', name='initial_value').text = '0'
        for name in ('velocity','effort'):
            _element(item, 'state_interface', name=name)


def _sensor_urdf(root, sensor):
    name = sensor['name']
    _element(root, 'link', name=name+'_link')
    joint = _element(root, 'joint', name=name+'_mount', type='fixed')
    _element(joint, 'parent', link=sensor['parent_link'])
    _element(joint, 'child', link=name+'_link')
    _origin(joint, sensor['transform'])
    _element(root, 'link', name=sensor['optical_frame'])
    optical = _element(root, 'joint', name=name+'_optical_joint', type='fixed')
    _element(optical, 'parent', link=name+'_link')
    _element(optical, 'child', link=sensor['optical_frame'])
    _origin(optical, {'xyz':[0,0,0],'rpy':[-math.pi/2,0,-math.pi/2]})


def build_robot_urdf(robot, ros2_control=False,
                     hardware_plugin='mock_components/GenericSystem',
                     controllers_file=None, sensors=()):
    """Fixed robots mount to world; floating robots keep their physical root link.

    Control is optional; default hardware plugin is explicitly mock components.
    No collision geometry is fabricated for unspecified link shapes.
    """
    root = ET.Element('robot', name=robot['name'])
    if robot['base']['type'] == 'fixed' and robot['base']['link'] != 'world':
        _element(root, 'link', name='world')
        mount = _element(root, 'joint', name='world_to_base', type='fixed')
        _element(mount, 'parent', link='world')
        _element(mount, 'child', link=robot['base']['link'])
        _origin(mount, robot['base'].get('origin'))
    for link in robot['links']:
        node = _element(root, 'link', name=link['name'])
        data = link['inertial']
        validate_inertial(data, link['name'])
        if data['mass'] > 0:
            inertial = _element(node, 'inertial')
            _origin(inertial, {'xyz':data['com'],'rpy':[0,0,0]})
            _element(inertial, 'mass', value=data['mass'])
            _element(inertial, 'inertia', **dict(zip(('ixx','iyy','izz','ixy','ixz','iyz'), data['inertia'])))
        for kind in ('visual','collision'):
            for geometry in _link_geometries(link, kind):
                _geometry_urdf(_element(node, kind), geometry)
    for joint in robot['joints']:
        node = _element(root, 'joint', name=joint['name'], type=joint['type'])
        _element(node, 'parent', link=joint['parent'])
        _element(node, 'child', link=joint['child'])
        _origin(node, joint['origin'])
        if joint['type'] == 'fixed':
            continue
        _element(node, 'axis', xyz=_fmt(joint['axis']))
        required = ('velocity','effort') + (('lower','upper') if joint['type'] != 'continuous' else ())
        missing = [key for key in required if joint['limits'].get(key) is None]
        if missing:
            raise ValueError(f"URDF joint {joint['name']}: missing limits {missing}")
        _element(node, 'limit', **{key:joint['limits'][key] for key in required})
        if joint.get('dynamics'):
            _element(node, 'dynamics', **joint['dynamics'])
    for sensor in sensors:
        _sensor_urdf(root, sensor)
    if ros2_control:
        _control(robot, root, hardware_plugin)
        if hardware_plugin == 'gz_ros2_control/GazeboSimSystem':
            plugin = _element(_element(root, 'gazebo'), 'plugin',
                              filename='gz_ros2_control-system', name='gz_ros2_control::GazeboSimROS2ControlPlugin')
            if controllers_file:
                _element(plugin, 'parameters').text = str(controllers_file)
    return _xml(root)


def _quat(rot):
    # MuJoCo quaternion order w,x,y,z. Stable near pi using largest diagonal.
    values = np.array([1+np.trace(rot), 1+2*rot[0,0]-np.trace(rot),
                       1+2*rot[1,1]-np.trace(rot), 1+2*rot[2,2]-np.trace(rot)])
    index = int(np.argmax(values))
    if index == 0:
        result = np.array([values[0], rot[2,1]-rot[1,2],rot[0,2]-rot[2,0],rot[1,0]-rot[0,1]])
    elif index == 1:
        result = np.array([rot[2,1]-rot[1,2],values[1],rot[0,1]+rot[1,0],rot[0,2]+rot[2,0]])
    elif index == 2:
        result = np.array([rot[0,2]-rot[2,0],rot[0,1]+rot[1,0],values[2],rot[1,2]+rot[2,1]])
    else:
        result = np.array([rot[1,0]-rot[0,1],rot[0,2]+rot[2,0],rot[1,2]+rot[2,1],values[3]])
    return result / np.linalg.norm(result)


def _pose(origin=None):
    rot, pos = transform(origin)
    return {'pos':_fmt(pos),'quat':_fmt(_quat(rot))}


def _geom_mjcf(body, asset, geometry, name, collision=True):
    attrs = {'name':name, **_pose(geometry.get('origin')), 'density':'0',
             'contype':'1' if collision else '0', 'conaffinity':'1' if collision else '0'}
    kind = geometry['type']
    if kind == 'box':
        attrs = {**attrs,'type':'box','size':_fmt(np.array(geometry['size'])/2)}
    elif kind == 'sphere':
        attrs = {**attrs,'type':'sphere','size':str(geometry['radius'])}
    elif kind == 'cylinder':
        attrs = {**attrs,'type':'cylinder','size':_fmt([geometry['radius'],geometry['length']/2])}
    elif kind == 'mesh':
        # Inline indexed mesh supports both ASCII/binary STL and portable MJCF.
        triangles = read_stl(geometry['path']) * mesh_scale(geometry)
        vertices, inverse = np.unique(triangles.reshape(-1,3), axis=0, return_inverse=True)
        _element(asset, 'mesh', name=name+'_mesh', vertex=_fmt(vertices.ravel()),
                 face=' '.join(str(int(x)) for x in inverse))
        attrs = {**attrs,'type':'mesh','mesh':name+'_mesh'}
    else:
        raise ValueError(f'Unsupported geometry: {kind}')
    _element(body, 'geom', **attrs)


def _body_mjcf(parent, link, origin, asset):
    body = _element(parent, 'body', name=link['name'], **_pose(origin))
    inertial = link['inertial']
    validate_inertial(inertial, link['name'])
    if inertial['mass']:
        _element(body, 'inertial', pos=_fmt(inertial['com']), mass=inertial['mass'],
                 fullinertia=_fmt(inertial['inertia']))
    collisions = _link_geometries(link, 'collision')
    visuals = _link_geometries(link, 'visual')
    for i, geom in enumerate(collisions):
        _geom_mjcf(body, asset, geom, link['name']+'_collision_'+str(i))
    for i, geom in enumerate(visuals):
        if geom not in collisions:
            _geom_mjcf(body, asset, geom, link['name']+'_visual_'+str(i), collision=False)
    return body


def _joint_mjcf(body, joint, actuators):
    kind = joint['type']
    if kind == 'fixed':
        return
    attrs = {'name':joint['name'],'type':'slide' if kind == 'prismatic' else 'hinge',
             'axis':_fmt(joint['axis'])}
    if kind != 'continuous':
        limits = [joint['limits'].get('lower'), joint['limits'].get('upper')]
        if any(value is None for value in limits):
            raise ValueError(f"MJCF joint {joint['name']}: missing lower/upper limits")
        attrs = {**attrs,'limited':'true','range':_fmt(limits)}
    dynamics = joint.get('dynamics', {})
    _element(body, 'joint', **attrs, damping=dynamics.get('damping',0), frictionloss=dynamics.get('friction',0))
    effort = joint['limits'].get('effort')
    motor = {'name':joint['name'],'joint':joint['name'],'gear':'1'}
    if effort is not None:
        effort = positive(effort, joint['name']+'.effort')
        motor = {**motor,'ctrllimited':'true','ctrlrange':_fmt([-effort,effort])}
    _element(actuators, 'motor', **motor)


def _camera_mjcf(body, sensor):
    mount_r, pos = transform(sensor['transform'])
    optical = rotation([-math.pi/2,0,-math.pi/2])
    # OpenGL/MuJoCo camera: -Z forward, +Y up; ROS optical: +Z forward, +Y down.
    rot = mount_r @ optical @ np.diag([1,-1,-1])
    width, height = sensor['resolution']
    intrinsics = sensor['intrinsics']
    # Pixel centres are integer ROS coordinates; OpenGL centre is (size-1)/2.
    # MuJoCo principalpixel offsets move the optical axis toward decreasing pixels.
    principal = [(width-1)/2-intrinsics['cx'], (height-1)/2-intrinsics['cy']]
    _element(body, 'camera', name=sensor['name'], pos=_fmt(pos), quat=_fmt(_quat(rot)),
             resolution=f'{width} {height}', sensorsize='1 1',
             focalpixel=_fmt([intrinsics['fx'], intrinsics['fy']]), principalpixel=_fmt(principal))


def build_robot_mjcf(robot, simulation=None, environment=(), sensors=()):
    """Explicit full inertia and unit-gear effort actuators; no density inference.

    STL collision is MuJoCo convex-hull contact approximation. Split nonconvex
    objects into convex components for faithful contacts. Base floating joint
    adds 7 position/6 velocity coordinates; frames are retained (fusestatic=false).
    """
    simulation = simulation or {}
    root = ET.Element('mujoco', model=robot['name'])
    _element(root, 'compiler', angle='radian', inertiafromgeom='false', fusestatic='false')
    if sensors:
        visual = _element(root, 'visual')
        _element(visual, 'global',
                 offwidth=max(640, max(camera['resolution'][0] for camera in sensors)),
                 offheight=max(480, max(camera['resolution'][1] for camera in sensors)))
    _element(root, 'option', timestep=positive(simulation.get('timestep', .002), 'simulation.timestep'),
             gravity=_fmt(finite_vector(simulation.get('gravity',[0,0,-9.81]))))
    asset, world = _element(root, 'asset'), _element(root, 'worldbody')
    actuators = _element(root, 'actuator')
    links = {link['name']:link for link in robot['links']}
    root_name = robot['base']['link']
    base = _body_mjcf(world, links[root_name], robot['base'].get('origin'), asset)
    if robot['base']['type'] == 'floating':
        _element(base, 'freejoint', name='floating_base')
    bodies = {root_name:base}
    pending = list(robot['joints'])
    while pending:
        ready = [j for j in pending if j['parent'] in bodies]
        if not ready:
            raise ValueError('robot joints are disconnected or cyclic')
        for joint in ready:
            body = _body_mjcf(bodies[joint['parent']], links[joint['child']], joint['origin'], asset)
            _joint_mjcf(body, joint, actuators)
            bodies[joint['child']] = body
        pending = [j for j in pending if j not in ready]
    for name, link in robot['end_effectors'].items():
        _element(bodies[link], 'site', name=name, pos='0 0 0', size='0.002')
    for item in environment:
        if not item.get('static', True):
            raise ValueError('MJCF environment currently requires static objects')
        body = _element(world, 'body', name=item['name'], **_pose(item['pose']))
        geometry = {**item['geometry']}
        if geometry['type'] == 'mesh':
            geometry = {**geometry,'units':geometry.get('units','m')}
        _geom_mjcf(body, asset, geometry, item['name']+'_collision')
    for sensor in sensors:
        if sensor['parent_link'] not in bodies:
            raise ValueError('sensor parent_link not in robot')
        _camera_mjcf(bodies[sensor['parent_link']], sensor)
    return _xml(root)
