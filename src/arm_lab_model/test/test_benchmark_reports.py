"""Report publication covers data, visible provenance and comparison plots."""
import json

from arm_lab_model.benchmark_engine import compare_runs
from arm_lab_model.benchmark_reports import write_benchmark_report
from test_benchmark_engine import observation


def test_reports_retain_provenance_and_numeric_results(tmp_path):
    sim = observation(tcp_position=[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    result = compare_runs(sim, sim, {'joint_position': 0.01, 'tcp_position': 0.001})
    paths = write_benchmark_report(result, tmp_path, sim, sim)
    assert json.loads(paths['json'].read_text())['status'] == 'passed'
    markdown = paths['markdown'].read_text()
    assert 'synthetic' in markdown and 'not real hardware' in markdown
    assert 'joint_position' in markdown and 'TCP final' in markdown
    assert all(path.is_file() and path.stat().st_size > 100 for path in paths['plots'])
    assert {'joint_position.png', 'tcp_position.png', 'tcp_trajectory.png', 'errors.png'} <= {p.name for p in paths['plots']}


def test_unavailable_report_does_not_claim_pass(tmp_path):
    result = compare_runs(observation(), observation(), {'joint_torque': 0.1})
    paths = write_benchmark_report(result, tmp_path)
    assert 'incomplete' in paths['markdown'].read_text()
