"""Adapt the legacy, fully parameterized tube arm without changing its inertials."""
import xml.etree.ElementTree as ET

from .config import load_config
from .mesh_physics import rotation, tensor6, tensor_matrix
from .urdf_builder import build_urdf


def _values(value, fallback='0 0 0'):
    return [float(x) for x in (value or fallback).split()]


def _origin(element):
    origin = element.find('origin')
    return {'xyz':_values(origin.get('xyz')) if origin is not None else [0,0,0],
            'rpy':_values(origin.get('rpy')) if origin is not None else [0,0,0]}


def _geometry(element):
    shape = list(element.find('geometry'))[0]
    if shape.tag == 'box':
        result = {'type':'box','size':_values(shape.get('size'))}
    elif shape.tag in ('cylinder','sphere'):
        result = {'type':shape.tag, **{key:float(value) for key,value in shape.attrib.items()}}
    else:
        result = {'type':'mesh','path':shape.get('filename'), 'units':'m',
                  'scale':_values(shape.get('scale'), '1 1 1')}
    return {**result,'origin':_origin(element)}


def legacy_robot(path):
    cfg = load_config(str(path))
    xml = ET.fromstring(build_urdf(cfg, fixed_to_world=False))
    links = []
    for element in xml.findall('link'):
        inertial = element.find('inertial')
        origin = _origin(inertial)
        tensor = inertial.find('inertia')
        values = [float(tensor.get(key, '0')) for key in ('ixx','iyy','izz','ixy','ixz','iyz')]
        rot = rotation(origin['rpy'])
        data = {'mass':float(inertial.find('mass').get('value')),'com':origin['xyz'],
                'inertia':tensor6(rot @ tensor_matrix(values) @ rot.T),
                'source':'Legacy tube/assembly equations or configured measured inertial: '+str(path)}
        # This is a coordinate frame, not the old exporter tiny numerical mass.
        if element.get('name') == 'tcp_link':
            data = {'mass':0.,'com':[0,0,0],'inertia':[0]*6,'source':'Massless TCP coordinate frame'}
        visuals = [_geometry(g) for g in element.findall('visual')] if data['mass'] else []
        collisions = [_geometry(g) for g in element.findall('collision')]
        links.append({'name':element.get('name'),'inertial':data,
                      'visuals':visuals,'collisions':collisions})
    joints = []
    for element in xml.findall('joint'):
        joint = {'name':element.get('name'),'type':element.get('type'),
                 'parent':element.find('parent').get('link'),'child':element.find('child').get('link'),
                 'origin':_origin(element)}
        if joint['type'] != 'fixed':
            limit = element.find('limit')
            joint = {**joint,'axis':_values(element.find('axis').get('xyz')),
                     'limits':{**{k:float(v) for k,v in limit.attrib.items()},'acceleration':None}}
            dynamics = element.find('dynamics')
            if dynamics is not None:
                joint = {**joint,'dynamics':{k:float(v) for k,v in dynamics.attrib.items()}}
        joints.append(joint)
    return {'schema_version':1,'format':'tree','name':cfg.name,'family':'serial_arm',
            'source':str(path),'base':{'link':'base_link','type':'fixed',
            'origin':{'xyz':list(cfg.mount_xyz),'rpy':list(cfg.mount_rpy)}},
            'links':links,'joints':joints,'end_effectors':{'tcp':'tcp_link'}}
