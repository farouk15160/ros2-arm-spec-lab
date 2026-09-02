"""The editor is generated from a schema, so the schema has to be right.

A path that does not exist in the configuration would silently create a new key
on first edit instead of changing the intended one, so every path is resolved
against the shipped configuration here.
"""

import os

import pytest
import yaml

from arm_lab_gui import schema
from arm_lab_gui.config_editor import get_path, set_path
from arm_lab_model.config import default_config_path, load_config


@pytest.fixture(scope='module')
def raw():
    with open(default_config_path()) as fh:
        return yaml.safe_load(fh)


def _all_fields(joint_index: int = 0):
    groups = (schema.robot_groups() + schema.joint_groups(joint_index)
              + schema.end_effector_groups() + schema.control_groups()
              + schema.error_groups() + schema.spec_groups())
    for group in groups:
        for field in group.fields:
            yield field


def test_every_schema_path_exists_in_the_config(raw):
    missing = [f.path for f in _all_fields()
               if get_path(raw, f.path, '__absent__') == '__absent__']
    assert not missing, f'schema references keys the config does not have: {missing}'


def test_every_joint_is_fully_addressable(raw):
    for index in range(len(raw['joints'])):
        for field in schema.joint_groups(index):
            for f in field.fields:
                assert get_path(raw, f.path, '__absent__') != '__absent__', f.path


def test_choice_fields_offer_the_value_the_config_uses(raw):
    for f in _all_fields():
        if f.kind != 'choice':
            continue
        value = get_path(raw, f.path)
        if f.choices:
            assert value in raw[f.choices], (
                f'{f.path} is {value!r}, absent from {f.choices}')
        elif f.options:
            assert value in f.options, f'{f.path} is {value!r}, not in {f.options}'


def test_set_path_round_trips(raw):
    import copy
    data = copy.deepcopy(raw)
    set_path(data, 'joints.1.link.length', 0.321)
    assert get_path(data, 'joints.1.link.length') == 0.321
    set_path(data, 'environment.gravity', 3.72)
    assert get_path(data, 'environment.gravity') == 3.72


def test_numeric_ranges_admit_the_shipped_values(raw):
    """A spin box that clamps the current value would silently change it."""
    offenders = []
    for f in _all_fields():
        if f.kind not in ('float', 'int'):
            continue
        value = get_path(f.path and raw, f.path)
        if value is None:
            continue
        if not (f.minimum <= float(value) <= f.maximum):
            offenders.append((f.path, value, f.minimum, f.maximum))
    assert not offenders, f'values outside their widget range: {offenders}'


def test_editing_through_the_schema_produces_a_loadable_config(raw, tmp_path):
    import copy
    data = copy.deepcopy(raw)
    set_path(data, 'joints.1.link.length', 0.38)
    set_path(data, 'end_effector.mass', 1.4)
    path = tmp_path / 'edited.yaml'
    with open(path, 'w') as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    cfg = load_config(str(path))
    assert cfg.joints[1].link.length == pytest.approx(0.38)
    assert cfg.end_effector.mass == pytest.approx(1.4)
