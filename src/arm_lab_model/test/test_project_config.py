"""Public project configuration contract; no ROS or simulator required."""
from pathlib import Path
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest
import yaml

from arm_lab_model.project_config import load_project


CONFIG = Path(__file__).resolve().parents[1] / 'config'


def tree_robot():
    legs = ('front_left', 'front_right', 'rear_left', 'rear_right')
    return {
        'schema_version': 1, 'format': 'tree', 'name': 'dog_12dof',
        'family': 'quadruped', 'source': 'Illustrative topology; no hardware data',
        'base': {'link': 'body', 'type': 'floating'},
        'links': [{'name': name, 'inertial': None} for name in
                  ['body'] + [f'{leg}_{i}' for leg in legs for i in range(3)]],
        'joints': [{
            'name': f'{leg}_joint_{i}', 'type': 'revolute',
            'parent': 'body' if i == 0 else f'{leg}_{i - 1}',
            'child': f'{leg}_{i}',
            'origin': {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]},
            'axis': [0, 1, 0],
            'limits': dict.fromkeys(('lower', 'upper', 'velocity', 'acceleration', 'effort')),
        } for leg in legs for i in range(3)],
        'end_effectors': {leg: f'{leg}_2' for leg in legs},
    }


def write_project(tmp_path, robot, **components):
    (tmp_path / 'robot.yaml').write_text(yaml.safe_dump(robot))
    for name, data in components.items():
        (tmp_path / f'{name}.yaml').write_text(yaml.safe_dump(data))
    project = {'schema_version': 1, 'robot': 'robot.yaml',
               **{name: f'{name}.yaml' for name in components}}
    path = tmp_path / 'project.yaml'
    path.write_text(yaml.safe_dump(project))
    return path


def test_legacy_project_preserves_existing_robot(tmp_path):
    robot = tmp_path / 'robot.yaml'
    robot.write_text(yaml.safe_dump({
        'schema_version': 1, 'format': 'legacy_arm',
        'source': str(CONFIG / 'arm_config.yaml'),
    }))
    project = tmp_path / 'project.yaml'
    project.write_text('schema_version: 1\nrobot: robot.yaml\n')
    result = load_project(project)
    assert result.robot_name == 'rover_arm'
    assert result.joint_names == tuple(f'joint_{i}' for i in range(1, 7))
    assert result.base_dof == 0
    assert result.files['robot'] == robot
    assert result.legacy_source == CONFIG / 'arm_config.yaml'
    with pytest.raises(TypeError):
        result.sections['robot']['source'] = 'changed.yaml'


def test_branched_robot_preserves_unknown_physics_and_floating_base(tmp_path):
    result = load_project(write_project(tmp_path, tree_robot()))
    assert len(result.joint_names) == 12
    assert result.base_dof == 6
    assert result.legacy_source is None
    assert 'links.body.inertial' in result.missing_parameters
    assert 'joints.front_left_joint_0.limits.effort' in result.missing_parameters
    assert len(result.sections['robot']['end_effectors']) == 4
    with pytest.raises(TypeError):
        result.sections['robot']['joints'][0]['axis'][0] = 1


def test_components_are_loaded_relative_to_manifest_without_starting_runtime(tmp_path, monkeypatch):
    path = write_project(tmp_path, tree_robot(),
                         simulation={'schema_version': 1, 'backend': 'mujoco',
                                     'timestep': 0.001, 'gravity': [0, 0, -9.81], 'seed': 0},
                         benchmark={'schema_version': 1, 'enabled': False, 'robot': 'ur5e'},
                         moveit={'schema_version': 1, 'enabled': False},
                         materials={'schema_version': 1, 'materials': {
                             'example': {'density': 2700, 'source': 'Illustrative assumption'}}})
    monkeypatch.chdir(tmp_path.parent)
    result = load_project(path)
    assert result.sections['simulation']['gravity'] == (0, 0, -9.81)
    assert result.sections['benchmark']['robot'] == 'ur5e'
    assert result.files['simulation'] == tmp_path / 'simulation.yaml'


def test_benchmark_thresholds_allow_exact_checks_and_reject_unknown_channels(tmp_path):
    benchmark = {'schema_version': 1, 'enabled': True, 'robot': 'ur5e',
                 'tolerances': {'joint_position': 0}}
    assert load_project(write_project(tmp_path, tree_robot(), benchmark=benchmark))
    with pytest.raises(ValueError, match='unknown tolerance'):
        load_project(write_project(tmp_path, tree_robot(), benchmark={
            **benchmark, 'tolerances': {'joint_postion': .1}}))


def test_moveit_shared_defaults_and_invalid_options(tmp_path):
    options = {'schema_version': 1, 'enabled': True, 'groups': {'leg': {
        'joints': ['front_left_joint_0'], 'base_link': 'body',
        'tip_link': 'front_left_0', 'controller': 'leg_controller'}}}
    assert load_project(write_project(tmp_path, tree_robot(), moveit=options))
    for extra in ({'planner_id': 'unsupported'}, {'saved_trajectories_dir': ''}):
        with pytest.raises(ValueError):
            load_project(write_project(tmp_path, tree_robot(), moveit={**options, **extra}))


@pytest.mark.parametrize('component', ['moveit', 'environment', 'sensors', 'trajectories'])
def test_enabled_components_require_their_configuration(tmp_path, component):
    data = {'schema_version': 1, 'enabled': True}
    if component == 'benchmark':
        data['robot'] = 'ur5e'
    with pytest.raises(ValueError, match='missing|required'):
        load_project(write_project(tmp_path, tree_robot(), **{component: data}))


def test_cli_reports_configuration_validity_separately_from_runtime(tmp_path):
    path = write_project(tmp_path, tree_robot())
    command = [sys.executable, '-m', 'arm_lab_model.project_config', str(path)]
    env = {**os.environ, 'PYTHONPATH': str(CONFIG.parent)}
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['valid'] is True
    assert report['runtime_ready'] is False
    assert report['joint_dof'] == 12
    assert report['base_velocity_dof'] == 6
    assert report['missing_parameters']
    path.write_text('schema_version: 900\nrobot: robot.yaml\n')
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)['valid'] is False
    assert 'schema_version' in result.stdout
    assert not result.stderr


@pytest.mark.parametrize('filename,dof', [('project.yaml', 6), ('project_quadruped_12dof.yaml', 12)])
def test_shipped_projects(filename, dof):
    result = load_project(CONFIG / 'pipeline' / filename)
    assert len(result.joint_names) == dof
    assert len(result.sections) == 9
    with pytest.raises(FrozenInstanceError):
        result.robot_name = 'changed'


@pytest.mark.parametrize('contents,message', [
    ('schema_version: 2\nrobot: robot.yaml', 'schema_version'),
    ('schema_version: true\nrobot: robot.yaml', 'schema_version'),
    ('schema_version: 1\nrobot: robot.yaml\nrobot: ignored.yaml', 'duplicate'),
    ('schema_version: 1\nrobot: robot.yaml\nwrong_key: 0', 'unknown'),
    ('schema_version: 1\nrobot: missing.yaml', 'missing.yaml'),
    ('schema_version: 1\nrobot: null', 'project.robot'),
    ('[one, two]', 'mapping'),
    ('', 'mapping'),
    ('!!python/object/apply:os.system ["false"]', 'constructor'),
    ('1: bad', 'keys must be strings'),
    ('robot: &robot [*robot]', 'aliases'),
    ('robot: [', 'expected'),
])
def test_invalid_manifest_is_actionable(tmp_path, contents, message):
    path = write_project(tmp_path, tree_robot())
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        load_project(path)


def replace_joint(robot, index, **changes):
    return {**robot, 'joints': [dict(joint, **changes) if i == index else joint
                              for i, joint in enumerate(robot['joints'])]}


@pytest.mark.parametrize('changes,message', [
    ({'parent': 'missing'}, 'unknown link'),
    ({'child': 'missing'}, 'unknown link'),
    ({'parent': []}, 'string'),
    ({'child': 'body'}, 'single root'),
    ({'child': 'front_left_1'}, 'multiple parents'),
    ({'parent': 'front_left_1'}, 'cycle'),
    ({'name': 'front_left_joint_1'}, 'duplicate'),
    ({'axis': [0, 0, 0]}, 'unit vector'),
    ({'axis': [0, 0, float('nan')]}, 'finite'),
    ({'axis': [0, True, 0]}, 'finite'),
    ({'axis': [0, 1]}, '3 numbers'),
    ({'type': 'spherical'}, 'type'),
    ({'typo': 10}, 'unknown'),
    ({'origin': {'xyz': [0, 0, 0]}}, 'rpy'),
])
def test_invalid_joint_or_graph_is_rejected(tmp_path, changes, message):
    robot = replace_joint(tree_robot(), 0, **changes)
    with pytest.raises(ValueError, match=message):
        load_project(write_project(tmp_path, robot))


@pytest.mark.parametrize('limits,message', [
    ({'lower': 1, 'upper': -1}, 'lower must'),
    ({'velocity': -1}, 'positive'),
    ({'effort': True}, 'finite'),
    ({'acceleration': float('inf')}, 'finite'),
])
def test_invalid_limits_fail_even_when_other_inputs_are_unknown(tmp_path, limits, message):
    robot = tree_robot()
    joint = robot['joints'][0]
    robot = replace_joint(robot, 0, limits={**joint['limits'], **limits})
    with pytest.raises(ValueError, match=message):
        load_project(write_project(tmp_path, robot))


@pytest.mark.parametrize('inertial,message', [
    ({'mass': 0}, 'positive'),
    ({'com': [0, 0]}, '3 numbers'),
    ({'inertia': [1, 1, 4, 0, 0, 0]}, 'triangle'),
    ({'source': ''}, 'nonempty'),
])
def test_manual_inertials_require_physical_validity_and_provenance(tmp_path, inertial, message):
    data = {'mass': 2, 'com': [0.1, 0, 0], 'inertia': [0.02, 0.03, 0.04, 0, 0, 0],
            'source': 'Illustrative test', **inertial}
    robot = tree_robot()
    robot = {**robot, 'links': [{'name': 'body', 'inertial': data}] + robot['links'][1:]}
    with pytest.raises(ValueError, match=message):
        load_project(write_project(tmp_path, robot))


def test_fixed_continuous_prismatic_and_manual_inertials(tmp_path):
    robot = tree_robot()
    fixed = {key: value for key, value in robot['joints'][0].items()
             if key not in ('axis', 'limits')}
    manual = {'mass': 2, 'com': [0.1, 0, 0], 'inertia': [0.02, 0.03, 0.04, 0, 0, 0],
              'source': 'Illustrative test'}
    robot = {**robot, 'base': {'link': 'body', 'type': 'fixed'},
             'links': [{'name': 'body', 'inertial': manual}] + robot['links'][1:],
             'joints': [{**fixed, 'type': 'fixed'}] + robot['joints'][1:]}
    robot = replace_joint(robot, 1, type='continuous', limits={
        'velocity': 1, 'acceleration': 2, 'effort': 3})
    robot = replace_joint(robot, 2, type='prismatic', limits={
        'lower': 0, 'upper': 0.2, 'velocity': 0.1, 'acceleration': 0.2, 'effort': 50})
    result = load_project(write_project(tmp_path, robot))
    assert len(result.joint_names) == 11
    assert result.base_dof == 0
    assert 'links.body.inertial' not in result.missing_parameters
    assert not any(key.startswith('joints.front_left') for key in result.missing_parameters)


@pytest.mark.parametrize('component,data,message', [
    ('simulation', {'backend': 'bad'}, 'backend'),
    ('simulation', {'timestep': 0}, 'timestep'),
    ('simulation', {'gravity': [0, 0, float('inf')]}, 'gravity'),
    ('simulation', {'seed': -1}, 'seed'),
    ('simulation', {'seed': True}, 'seed'),
    ('moveit', {'enabled': 'false'}, 'boolean'),
    ('moveit', {'schema_version': 2}, 'schema_version'),
    ('benchmark', {'robot': '../ur5e'}, 'identifier'),
])
def test_invalid_component_parameters(tmp_path, component, data, message):
    defaults = ({'backend': 'mujoco', 'timestep': 0.001, 'gravity': [0, 0, -9.81], 'seed': 0}
                if component == 'simulation' else {'enabled': False})
    if component == 'benchmark':
        defaults = {**defaults, 'robot': None}
    value = {'schema_version': 1, **defaults, **data}
    with pytest.raises(ValueError, match=message):
        load_project(write_project(tmp_path, tree_robot(), **{component: value}))


def test_unrepresentable_numeric_input_returns_invalid_json(tmp_path, capsys):
    from arm_lab_model.project_config import main
    path = write_project(tmp_path, tree_robot(), simulation={
        'schema_version': 1, 'backend': 'mujoco', 'timestep': 10 ** 400,
        'gravity': [0, 0, -9.81], 'seed': 0})
    assert main([str(path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['valid'] is False
    assert 'simulation.timestep' in report['error']
