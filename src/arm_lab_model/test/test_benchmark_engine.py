"""Independent numerical examples for observation comparison."""
from copy import deepcopy
import math

import pytest

from arm_lab_model.benchmark_engine import compare_runs, load_observation


def observation(times=None, positions=None, **channels):
    return {'schema_version': 1, 'robot': 'test_arm', 'scenario_id': 'ramp',
            'joint_names': ['joint1'], 'frame': 'base', 'units': 'SI',
            'evidence': {'kind': 'synthetic', 'source': 'unit test, not real hardware'},
            'samples': {'time': times or [0., 1., 2.],
                        'joint_position': positions or [[0.], [1.], [2.]], **channels}}


def test_align_different_sample_rates_and_known_error():
    sim = observation([0., 0.5, 1., 1.5, 2.], [[0.1], [0.6], [1.1], [1.6], [2.1]])
    result = compare_runs(sim, observation(), {'joint_position': 0.11, 'duration': 0.01})
    assert result['status'] == 'passed'
    assert result['metrics']['joint_position']['rmse'] == pytest.approx(0.1)
    assert result['metrics']['joint_position']['max'] == pytest.approx(0.1)
    assert result['metrics']['joint_position']['final'] == pytest.approx(0.1)
    assert result['evidence']['reference']['kind'] == 'synthetic'


def test_quaternion_sign_is_same_rotation_and_slerp_is_correct():
    ref = observation([0., 2.], [[0.], [2.]], tcp_orientation=[[0., 0., 0., 1.], [0., 0., 1., 0.]])
    h = math.sqrt(0.5)
    sim = observation(tcp_orientation=[[0., 0., 0., -1.], [0., 0., -h, -h], [0., 0., -1., 0.]])
    result = compare_runs(sim, ref, {'tcp_orientation': 1e-6})
    assert result['metrics']['tcp_orientation']['max'] == pytest.approx(0., abs=1e-7)


def test_missing_requested_channel_never_passes():
    result = compare_runs(observation(), observation(), {'joint_torque': 0.1})
    assert result['status'] == 'incomplete'
    assert result['metrics']['joint_torque']['status'] == 'unavailable'


def test_no_tolerances_means_unassessed():
    assert compare_runs(observation(), observation(), {})['status'] == 'incomplete'


@pytest.mark.parametrize('field,value', [('robot', 'other'), ('scenario_id', 'other'),
    ('frame', 'tool'), ('units', 'degrees'), ('joint_names', ['other'])])
def test_refuse_incompatible_metadata(field, value):
    with pytest.raises(ValueError):
        compare_runs({**observation(), field: value}, observation(), {})


def test_partial_overlap_marked_and_full_duration_retained():
    result = compare_runs(observation([1., 2.], [[1.], [2.]]), observation(), {'joint_position': 0.01})
    assert result['status'] == 'incomplete'
    assert result['alignment']['partial_overlap'] is True
    assert result['duration']['simulation'] == 1.
    assert result['duration']['reference'] == 2.


def test_effort_integrals_known_constant_power():
    trace = observation(joint_velocity=[[2.], [2.], [2.]], joint_torque=[[3.], [3.], [3.]])
    result = compare_runs(trace, deepcopy(trace), {'energy': 0.001, 'effort': 0.001})
    assert result['energy']['simulation'] == pytest.approx(12.)
    assert result['effort']['simulation'] == pytest.approx(6.)
    assert result['status'] == 'passed'


def test_nonuniform_effort_and_work_without_removed_numpy_trapz(monkeypatch):
    import numpy as np
    monkeypatch.delattr(np, 'trapz', raising=False)  # NumPy 2.4 removed this API.
    trace = observation([0., 0.5, 2.], joint_velocity=[[2.], [2.], [2.]],
                        joint_torque=[[1.], [2.], [5.]])
    result = compare_runs(trace, deepcopy(trace), {'energy': 0.001, 'effort': 0.001})
    # Torque = 1 + 2t: area over [0,2] is 6 N m s; constant speed doubles work.
    assert result['effort']['simulation'] == pytest.approx(6.)
    assert result['energy']['simulation'] == pytest.approx(12.)
    assert result['status'] == 'passed'


def test_continuous_joint_unwrap_preserves_short_crossing():
    sim = {**observation([0., 1., 2.], [[3.], [3.14], [-3.]]), 'continuous_joints': ['joint1']}
    ref = {**observation([0., 1., 2.], [[3.], [3.14], [2 * math.pi - 3.]]), 'continuous_joints': ['joint1']}
    assert compare_runs(sim, ref, {'joint_position': 1e-6})['status'] == 'passed'


@pytest.mark.parametrize('times,positions', [([0., 0.], [[0.], [1.]]),
    ([0., 1.], [[0.], [float('nan')]]), ([0., 1.], [[0., 1.], [1., 2.]])])
def test_bad_samples_rejected(times, positions):
    with pytest.raises(ValueError):
        compare_runs(observation(times, positions), observation(), {})


def test_csv_ingest_requires_explicit_metadata(tmp_path):
    path = tmp_path / 'trace.csv'
    path.write_text('t,q\n0,0\n1,1\n2,2\n')
    meta = {key: value for key, value in observation().items() if key != 'samples'}
    loaded = load_observation(path, metadata=meta, columns={'time': 't', 'joint_position': ['q']})
    assert compare_runs(loaded, observation(), {'joint_position': 0.001})['status'] == 'passed'
    with pytest.raises(ValueError):
        load_observation(path)


def test_floating_clock_roundoff_is_not_missing_evidence():
    sim = observation([0., 1., 2.000000000000001])
    assert compare_runs(sim, observation(), {'joint_position': 0.001})['status'] == 'passed'


def test_excess_error_fails_with_expected_maximum():
    result = compare_runs(observation(positions=[[0.], [1.], [3.]]), observation(), {'joint_position': 0.1})
    assert result['status'] == 'failed'
    assert result['metrics']['joint_position']['max'] == 1.


def test_nonoverlapping_clocks_rejected():
    with pytest.raises(ValueError, match='overlap'):
        compare_runs(observation([3., 4.], [[3.], [4.]]), observation(), {})


@pytest.mark.parametrize('tolerances', [{'bogus': 1.}, {'joint_position': -1.}, {'joint_position': float('nan')}, []])
def test_bad_tolerances_rejected(tolerances):
    with pytest.raises(ValueError):
        compare_runs(observation(), observation(), tolerances)


def test_yaml_ingest_and_ambiguous_overrides(tmp_path):
    import yaml
    path = tmp_path / 'observations.yaml'
    path.write_text(yaml.safe_dump(observation()))
    assert load_observation(path) == observation()
    with pytest.raises(ValueError):
        load_observation(path, metadata={})


def test_nonunit_quaternion_rejected():
    with pytest.raises(ValueError, match='unit quaternion'):
        compare_runs(observation(tcp_orientation=[[0., 0., 0., 2.]] * 3), observation(), {})


def test_csv_duplicate_headers_rejected(tmp_path):
    path = tmp_path / 'trace.csv'
    path.write_text('t,t\n0,0\n1,1\n')
    meta = {key: value for key, value in observation().items() if key != 'samples'}
    with pytest.raises(ValueError, match='unique'):
        load_observation(path, metadata=meta, columns={'time': 't', 'joint_position': ['t']})


def test_prismatic_units_and_work():
    trace = {**observation(joint_velocity=[[2.]] * 3, joint_torque=[[3.]] * 3),
             'joint_types': ['prismatic']}
    result = compare_runs(trace, trace, {'joint_position': 0.01, 'energy': 0.01, 'effort': 0.01})
    assert result['metrics']['joint_position']['unit'] == 'm'
    assert result['metrics']['joint_torque']['unit'] == 'N'
    assert result['energy']['simulation'] == 12.
    assert result['effort']['unit'] == 'N s'


def test_mixed_joints_keep_error_units_separate_and_effort_unavailable():
    trace = {**observation(), 'joint_names': ['hinge', 'slider'], 'joint_types': ['revolute', 'prismatic'],
             'samples': {'time': [0., 1.], 'joint_position': [[0., 0.], [1., 0.1]],
                         'joint_torque': [[1., 2.], [1., 2.]]}}
    result = compare_runs(trace, trace, {'joint_position': 0.01, 'effort': 1.})
    assert result['status'] == 'incomplete'
    assert result['metrics']['joint_position']['rmse'] is None
    assert result['metrics']['joint_position']['per_joint']['slider']['unit'] == 'm'
    assert result['metrics']['effort']['status'] == 'unavailable'


def test_differing_joint_types_rejected():
    with pytest.raises(ValueError, match='joint_types'):
        compare_runs({**observation(), 'joint_types': ['prismatic']}, observation(), {})


def test_tcp_end_effector_identity_is_checked():
    trace = {**observation(tcp_position=[[0., 0., 0.]] * 3), 'end_effector': 'tool0'}
    assert compare_runs(trace, trace, {'tcp_position': 0.001})['status'] == 'passed'
    with pytest.raises(ValueError, match='end_effector'):
        compare_runs(trace, {**trace, 'end_effector': 'flange'}, {})
    with pytest.raises(ValueError, match='end_effector'):
        compare_runs(trace, observation(tcp_position=[[0., 0., 0.]] * 3), {})


def test_blank_end_effector_rejected():
    with pytest.raises(ValueError, match='end_effector'):
        compare_runs({**observation(), 'end_effector': ''}, observation(), {})
