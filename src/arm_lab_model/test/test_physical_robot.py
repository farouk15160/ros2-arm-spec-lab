from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

from arm_lab_model.physical_robot import resolve_robot, physical_report, robot_fingerprint


def pendulum():
    return {'schema_version':1,'format':'tree','name':'pendulum','family':'test','source':'analytic',
        'base':{'link':'base','type':'fixed'},'end_effectors':{'tcp':'arm'},
        'links':[{'name':'base','inertial':{'mass':0,'com':[0,0,0],'inertia':[0]*6,'source':'frame'}},
                 {'name':'arm','geometry':{'type':'box','size':[2,.1,.1],
                        'origin':{'xyz':[1,0,0],'rpy':[0,0,0]}},
                  'material':'test','inertial':{'mode':'auto'}}],
        'joints':[{'name':'hinge','type':'revolute','parent':'base','child':'arm',
                   'origin':{'xyz':[0,0,0],'rpy':[0,0,0]},'axis':[0,1,0],
                   'limits':{'lower':-3.,'upper':3.,'velocity':2.,'acceleration':4.,'effort':100.}}]}


def project(robot):
    return SimpleNamespace(sections={'robot':robot,'materials':{'materials':{'test':{'density':100,'source':'test'}}}},
                           files={'robot':Path('/tmp/robot.yaml')},legacy_source=None)


def test_resolve_then_static_torque_agrees_with_pendulum_and_payload():
    original = pendulum()
    robot = resolve_robot(project(original))
    report = physical_report(robot, payload={'link':'arm','mass':1,'com':[2,0,0]})
    assert 'mass' not in original['links'][1]['inertial']
    assert report['total_mass'] == pytest.approx(3)
    assert report['center_of_mass'] == pytest.approx([4/3,0,0])
    assert report['joint_static_effort']['hinge'] == pytest.approx(-39.24)
    assert len(robot_fingerprint(robot)) == 64


def test_missing_inertia_fails_clearly():
    robot = pendulum()
    robot['links'][1]['inertial'] = None
    with pytest.raises(ValueError, match='arm.*inertial'):
        resolve_robot(project(robot))


def test_complete_manual_override_ignores_density_and_stl_estimates():
    original = pendulum()
    original['links'][1]['inertial'] = {'mode':'manual','mass':3,'com':[.5,0,0],
        'inertia':[.1,1,1,0,0,0],'source':'weighed CAD assembly'}
    result = resolve_robot(project(original))
    assert result['links'][1]['inertial']['mass'] == 3
    assert physical_report(result)['joint_static_effort']['hinge'] == pytest.approx(-14.715)


def test_floating_massless_root_rejected():
    robot = pendulum()
    robot['base']['type'] = 'floating'
    with pytest.raises(ValueError, match='massless'):
        resolve_robot(project(robot))


def test_topology_accepts_auto_and_manual_fixed_frames():
    from arm_lab_model.robot_topology import validate_robot
    names, _, missing = validate_robot(pendulum())
    assert names == ('hinge',)
    assert missing == ()


def test_joint_angle_changes_world_com_and_static_effort():
    robot = resolve_robot(project(pendulum()))
    result = physical_report(robot, q={'hinge':np.pi/2})
    assert result['center_of_mass'] == pytest.approx([0,0,-1],abs=1e-12)
    assert result['joint_static_effort']['hinge'] == pytest.approx(0,abs=1e-12)


def test_branched_robot_loads_only_its_own_descendants():
    original = pendulum()
    original['links'].append({**original['links'][1],'name':'other'})
    original['joints'].append({**original['joints'][0],'name':'other_hinge','child':'other'})
    robot = resolve_robot(project(original))
    result = physical_report(robot, q=[0,np.pi/2])
    assert result['joint_static_effort']['hinge'] == pytest.approx(-19.62)
    assert result['joint_static_effort']['other_hinge'] == pytest.approx(0,abs=1e-12)
    assert result['total_mass'] == pytest.approx(4)
    assert result['center_of_mass'] == pytest.approx([.5,0,-.5])


@pytest.mark.parametrize('change,match', [
    ({'material':'missing'},'unknown material'),
    ({'material':None},'material density'),
    ({'geometry':None},'requires geometry'),
    ({'inertial':{'mode':'oops'}},'mode'),
    ({'inertial':{'mass':1,'com':[0,0,0],'inertia':[1]*6}},'source'),
    ({'inertial':{'mass':1,'com':[0,0,0],'inertia':[1,1,4,0,0,0],'source':'test'}},'triangle'),
])
def test_missing_or_invalid_physical_parameters_raise(change,match):
    original = pendulum()
    original['links'][1] = {**original['links'][1],**change}
    with pytest.raises(ValueError,match=match):
        resolve_robot(project(original))


def test_payload_frame_and_joint_state_validation():
    robot = resolve_robot(project(pendulum()))
    with pytest.raises(ValueError,match='payload.link'):
        physical_report(robot,payload={'link':'missing','mass':1,'com':[0,0,0]})
    with pytest.raises(ValueError,match='unknown joints'):
        physical_report(robot,q={'missing':1})
    with pytest.raises(ValueError,match='finite'):
        physical_report(robot,q={'hinge':float('nan')})
