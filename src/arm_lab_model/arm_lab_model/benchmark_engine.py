"""Time-aligned comparisons preserving provenance and incomplete evidence."""
import math

import numpy as np

from .benchmark_observations import CHANNELS, load_observation, validate_observation
from .config_contract import number


def _continuous(record):
    types = record.get('joint_types', ['revolute'] * len(record['joint_names']))
    return set(record.get('continuous_joints', [])) | {
        name for name, kind in zip(record['joint_names'], types) if kind == 'continuous'}


def _interpolate(record, channel, times):
    source_times = np.asarray(record['samples']['time'])
    values = np.asarray(record['samples'][channel], dtype=float)
    if channel == 'tcp_orientation':
        return _slerp(source_times, values, times)
    if channel == 'joint_position':
        continuous = _continuous(record)
        values = np.column_stack([np.unwrap(values[:, i]) if name in continuous else values[:, i]
                                  for i, name in enumerate(record['joint_names'])])
    return np.column_stack([np.interp(times, source_times, values[:, i]) for i in range(values.shape[1])])


def _slerp(source_times, values, times):
    result = []
    for time in times:
        index = int(np.clip(np.searchsorted(source_times, time, side='right') - 1, 0, len(source_times) - 2))
        a, b = values[index], values[index + 1]
        fraction = (time - source_times[index]) / (source_times[index + 1] - source_times[index])
        dot = float(np.dot(a, b))
        b = -b if dot < 0 else b
        dot = min(1., abs(dot))
        if dot > 0.9995:
            q = a + fraction * (b - a)
        else:
            angle = math.acos(dot)
            q = (math.sin((1 - fraction) * angle) * a + math.sin(fraction * angle) * b) / math.sin(angle)
        result.append(q / np.linalg.norm(q))
    return np.asarray(result)


def _metric(errors, unit, tolerance):
    errors = np.asarray(errors)
    maximum = float(np.max(np.abs(errors)))
    return {'rmse': float(np.sqrt(np.mean(errors ** 2))), 'max': maximum,
            'final': float(np.max(np.abs(errors[-1]))), 'unit': unit,
            'tolerance': tolerance,
            'status': ('unassessed' if tolerance is None else 'passed' if maximum <= tolerance else 'failed')}


def _compatible(simulation, reference, tolerances):
    sim, ref = validate_observation(simulation), validate_observation(reference)
    for key in ('robot', 'scenario_id', 'joint_names', 'frame', 'units'):
        if sim[key] != ref[key]:
            raise ValueError(f'incompatible {key}; transform/reorder observations explicitly')
    if sim.get('end_effector') != ref.get('end_effector'):
        raise ValueError('incompatible end_effector; both observations must identify the same TCP')
    if _continuous(sim) != _continuous(ref):
        raise ValueError('incompatible continuous_joints declarations')
    count = len(sim['joint_names'])
    if sim.get('joint_types', ['revolute'] * count) != ref.get('joint_types', ['revolute'] * count):
        raise ValueError('incompatible joint_types declarations')
    if not isinstance(tolerances, dict):
        raise ValueError('tolerances must be a mapping of channel to absolute maximum error')
    for key, tolerance in tolerances.items():
        if key not in set(CHANNELS) | {'duration', 'energy', 'effort'}:
            raise ValueError(f'unknown tolerance channel: {key}')
        if number(tolerance, 'tolerance.' + key) < 0:
            raise ValueError('tolerances must be nonnegative')
    return sim, ref


def _alignment(sim, ref):
    a, b = np.asarray(sim['samples']['time']), np.asarray(ref['samples']['time'])
    start, end = max(a[0], b[0]), min(a[-1], b[-1])
    if end <= start:
        raise ValueError('observation time ranges have no positive overlap')
    times = np.unique(np.concatenate(([start, end], a[(a >= start) & (a <= end)], b[(b >= start) & (b <= end)])))
    return times, {'start': float(start), 'end': float(end), 'samples': len(times),
                   'partial_overlap': not (math.isclose(a[0], b[0], rel_tol=0, abs_tol=1e-9)
                                           and math.isclose(a[-1], b[-1], rel_tol=0, abs_tol=1e-9)),
                   'method': 'linear channels; shortest-arc SLERP orientation; no extrapolation'}


def compare_runs(simulation: dict, reference: dict, tolerances: dict) -> dict:
    """Compare matching SI scenario clocks; missing requested evidence cannot pass.

    Joint errors use maximum per-component absolute error. TCP vector errors use
    Euclidean norm; orientation uses geodesic rotation angle. Tolerances apply to
    maximum error. RMSE is sample-weighted on the union of overlap sample times.
    """
    sim, ref = _compatible(simulation, reference, tolerances)
    times, alignment = _alignment(sim, ref)
    channels = (set(sim['samples']) | set(ref['samples']) | set(tolerances)) & set(CHANNELS)
    metrics, series = {}, {}
    for channel in sorted(channels):
        if channel not in sim['samples'] or channel not in ref['samples']:
            metrics[channel] = {'status': 'unavailable', 'reason': 'channel absent from simulation or reference'}
            continue
        a, b = _interpolate(sim, channel, times), _interpolate(ref, channel, times)
        errors = _errors(channel, a, b, sim)
        metrics[channel] = _channel_metric(channel, errors, sim, tolerances.get(channel))
        series[channel] = {'simulation': a.tolist(), 'reference': b.tolist(), 'error': errors.tolist()}
    duration = _durations(sim, ref, tolerances, metrics)
    integrals = _integrals(sim, ref, times, tolerances, metrics)
    statuses = [metrics[key]['status'] for key in tolerances]
    status = ('failed' if 'failed' in statuses else 'incomplete' if not statuses or
              alignment['partial_overlap'] or any(s != 'passed' for s in statuses) else 'passed')
    return {'schema_version': 1, 'robot': sim['robot'], 'scenario_id': sim['scenario_id'],
            'joint_names': sim['joint_names'], 'frame': sim['frame'], 'status': status,
            'evidence': {'simulation': sim['evidence'], 'reference': ref['evidence']},
            'alignment': alignment, 'metrics': metrics, 'duration': duration, **integrals,
            'missing_channels': sorted(key for key, value in metrics.items() if value['status'] == 'unavailable'),
            'series': {'time': times.tolist(), **series},
            'limitations': ['Only configured tolerances are acceptance criteria.',
                            'Joint torque × joint velocity is mechanical work, not electrical energy.',
                            'Comparison does not certify hardware safety or validate provenance authenticity.']}


def _channel_metric(channel, errors, record, tolerance):
    if not channel.startswith('joint_'):
        return _metric(errors, CHANNELS[channel][0], tolerance)
    linear = {'joint_position': 'm', 'joint_velocity': 'm/s',
              'joint_acceleration': 'm/s²', 'joint_torque': 'N'}
    types = record.get('joint_types', ['revolute'] * len(record['joint_names']))
    units = [linear[channel] if kind == 'prismatic' else CHANNELS[channel][0] for kind in types]
    if len(set(units)) == 1:
        return _metric(errors, units[0], tolerance)
    per_joint = {name: _metric(errors[:, i], units[i], tolerance)
                 for i, name in enumerate(record['joint_names'])}
    status = ('unassessed' if tolerance is None else
              'failed' if any(item['status'] == 'failed' for item in per_joint.values()) else 'passed')
    return {'rmse': None, 'max': None, 'final': None, 'unit': units,
            'tolerance': tolerance, 'status': status, 'per_joint': per_joint}


def _errors(channel, a, b, sim):
    if channel == 'tcp_orientation':
        return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=1)), 0., 1.))
    difference = a - b
    if channel == 'joint_position':
        continuous = _continuous(sim)
        difference = np.column_stack([
            np.arctan2(np.sin(difference[:, i]), np.cos(difference[:, i])) if name in continuous
            else difference[:, i] for i, name in enumerate(sim['joint_names'])])
    return np.linalg.norm(difference, axis=1) if channel.startswith('tcp_') else difference


def _durations(sim, ref, tolerances, metrics):
    a, b = [record['samples']['time'][-1] - record['samples']['time'][0] for record in (sim, ref)]
    metrics['duration'] = _metric(np.array([a - b]), 's', tolerances.get('duration'))
    return {'simulation': a, 'reference': b, 'difference': a - b, 'unit': 's'}


def _integrals(sim, ref, times, tolerances, metrics):
    result = {}
    types = sim.get('joint_types', ['revolute'] * len(sim['joint_names']))
    is_linear = [kind == 'prismatic' for kind in types]
    for name, required, unit in (('effort', ('joint_torque',), 'N m s'),
                                  ('energy', ('joint_torque', 'joint_velocity'), 'J')):
        if name == 'effort' and any(is_linear) and not all(is_linear):
            metrics[name] = {'status': 'unavailable', 'reason': 'cannot sum torque and force integrals with unlike units'}
            continue
        unit = 'N s' if name == 'effort' and all(is_linear) else unit
        if not all(channel in record['samples'] for channel in required for record in (sim, ref)):
            if name in tolerances:
                metrics[name] = {'status': 'unavailable', 'reason': 'requires ' + ', '.join(required)}
            continue
        values = [_integral(record, required, times) for record in (sim, ref)]
        result[name] = {'simulation': values[0], 'reference': values[1],
                        'difference': values[0] - values[1], 'unit': unit, 'interval': 'overlap'}
        metrics[name] = _metric(np.array([values[0] - values[1]]), unit, tolerances.get(name))
    return result


def _integral(record, channels, times):
    torque = _interpolate(record, channels[0], times)
    power = torque if len(channels) == 1 else torque * _interpolate(record, channels[1], times)
    # Absolute per-joint contributions: opposing joints do not cancel effort.
    contributions = np.sum(np.abs(power), axis=1)
    return float(np.sum(np.diff(times) * (contributions[:-1] + contributions[1:]) / 2))
