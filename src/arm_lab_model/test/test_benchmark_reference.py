"""Public benchmark loading and independent UR reference evidence checks."""
from pathlib import Path

import pytest
import yaml

from arm_lab_model.benchmark_reference import load_benchmark_reference

CONFIG = Path(__file__).parents[1] / 'config' / 'benchmarks' / 'ur5e.yaml'


def test_ur5e_reference_is_sourced_and_never_claims_hardware_measurements():
    reference = load_benchmark_reference(CONFIG)
    assert reference.data['robot']['model'] == 'UR5e'
    assert len(reference.joint_names) == 6
    assert reference.data['payload']['maximum_mass_kg'] == 5.0
    assert not [d for d in reference.data['datasets'].values()
                if d['kind'] == 'hardware_measurement']
    assert all(j['acceleration_rad_s2'] is None
               for j in reference.data['joint_limits'].values())
    assert all(j['continuous_torque_nm'] is None
               for j in reference.data['joint_limits'].values())
    assert reference.data['sources']['ur_description']['revision'] == (
        'b48aa88ac18a17466e767929f05f45b23e332a8e')
    with pytest.raises(TypeError):
        reference.data['robot']['model'] = 'invented'


def _copy_catalogue(tmp_path):
    import shutil
    target = tmp_path / 'config'
    shutil.copytree(CONFIG.parent, target / 'benchmarks')
    shutil.copytree(CONFIG.parent.parent / 'pipeline', target / 'pipeline')
    return target / 'benchmarks' / CONFIG.name


def test_catalogue_rejects_tampered_upstream_evidence(tmp_path):
    path = _copy_catalogue(tmp_path)
    upstream = path.parent / 'upstream/ur5e/physical_parameters.yaml'
    upstream.write_text(upstream.read_text() + '\n# modified\n')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_benchmark_reference(path)


@pytest.mark.parametrize('mutation,message', [
    (lambda d: d.update(schema_version=2), 'version 1'),
    (lambda d: d.update(unknown=True), 'unknown'),
    (lambda d: d['joint_names'].append(d['joint_names'][0]), 'duplicate'),
    (lambda d: d['joint_limits'].pop('elbow_joint'), 'names must match'),
    (lambda d: d['joint_limits']['elbow_joint'].update(velocity_rad_s=-1), 'positive'),
    (lambda d: d['joint_limits']['elbow_joint'].update(position_rad=[1, -1]), 'lower'),
    (lambda d: d['joint_limits']['elbow_joint'].update(source='missing'), 'unknown source'),
    (lambda d: d['payload'].update(maximum_mass_kg=float('nan')), 'finite'),
    (lambda d: d['kinematics']['rows'].pop(), 'one DH row'),
    (lambda d: d['physical_properties'].update(inertia_units='g*mm^2'), 'inertia_units'),
    (lambda d: d['robot'].update(name='wrong_robot'), 'identity mismatch'),
])
def test_catalogue_rejects_invalid_declarations(tmp_path, mutation, message):
    path = _copy_catalogue(tmp_path)
    data = yaml.safe_load(path.read_text())
    mutation(data)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=message):
        load_benchmark_reference(path)


def test_analytical_cases_are_untimed_and_agree_with_published_zero_pose():
    import numpy as np
    reference = load_benchmark_reference(CONFIG)
    dataset = reference.data['datasets']['nominal_fk']
    assert dataset['kind'] == 'analytical_reference'
    cases = yaml.safe_load((CONFIG.parent / dataset['file']).read_text())
    assert 'time' not in cases
    assert cases['frame'] == 'base'
    zero = cases['cases'][0]
    np.testing.assert_allclose(zero['joint_position'], [0.] * 6)
    np.testing.assert_allclose(zero['tcp_position'], [-0.8172, -0.2329, 0.0628], atol=1e-12)
    np.testing.assert_allclose(zero['tcp_orientation'], [2**-.5, 0., 0., 2**-.5], atol=1e-12)


def test_manufacturer_parameters_cannot_be_relabelled_hardware_samples(tmp_path):
    path = _copy_catalogue(tmp_path)
    data = yaml.safe_load(path.read_text())
    data['datasets']['nominal_fk']['kind'] = 'hardware_measurement'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='hardware source'):
        load_benchmark_reference(path)


def test_robot_tree_matches_independent_manufacturer_dh_reference_cases():
    import numpy as np
    from arm_lab_model.physical_robot import forward_tree, resolve_robot
    from arm_lab_model.project_config import load_project
    reference = load_benchmark_reference(CONFIG)
    project = load_project(CONFIG.parent.parent / 'pipeline/project_ur5e.yaml')
    robot = resolve_robot(project)
    dataset = reference.data['datasets']['nominal_fk']
    fixture = yaml.safe_load((CONFIG.parent / dataset['file']).read_text())
    for case in fixture['cases']:
        frames, _ = forward_tree(robot, dict(zip(reference.joint_names, case['joint_position'])))
        base_r, base_p = frames['base']
        tcp_r, tcp_p = frames['tool0']
        np.testing.assert_allclose(base_r.T @ (tcp_p-base_p), case['tcp_position'], atol=5e-10)
        x, y, z, w = case['tcp_orientation']
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        np.testing.assert_allclose(base_r.T @ tcp_r, rotation, atol=5e-10)


def test_reference_report_distinguishes_nominal_agreement_and_known_mass_discrepancy():
    from arm_lab_model.benchmark_reference import check_reference_model
    from arm_lab_model.physical_robot import resolve_robot
    from arm_lab_model.project_config import load_project
    project = load_project(CONFIG.parent.parent / 'pipeline/project_ur5e.yaml')
    report = check_reference_model(resolve_robot(project), CONFIG)
    assert report['evidence_kind'] == 'analytical_reference'
    assert report['hardware_validation'] is False
    assert report['case_count'] == 5
    assert report['passed'] is True
    assert report['max_position_error_m'] < 5e-10
    assert report['max_orientation_error_rad'] < 5e-10
    assert report['mass']['model_kg'] == pytest.approx(21.7)
    assert report['mass']['manufacturer_kg'] == 20.7
    assert report['mass']['difference_kg'] == pytest.approx(1.)
