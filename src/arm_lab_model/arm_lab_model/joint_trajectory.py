"""Quintic joint interpolation with continuous polynomial limit checks."""
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from numpy.polynomial import polynomial as poly


def _array(value, size, label):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{label}: expected finite vector') from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{label}: expected {size} finite values')
    return result


def _segment(left, right, dt):
    p0, v0, a0 = left
    p1, v1, a1 = right
    initial = np.array([p0, v0 * dt, a0 * dt ** 2 / 2])
    residual = np.array([p1 - initial.sum(axis=0),
                         v1 * dt - initial[1] - 2 * initial[2],
                         a1 * dt ** 2 - 2 * initial[2]])
    rest = np.linalg.solve([[1, 1, 1], [3, 4, 5], [6, 12, 20]], residual)
    return tuple(tuple(float(v) for v in row) for row in np.vstack((initial, rest)))


@dataclass(frozen=True)
class JointPath:
    names: tuple
    times: tuple
    coefficients: tuple

    @classmethod
    def from_points(cls, names, points):
        if not names or len(set(names)) != len(names) or len(points) < 2:
            raise ValueError('trajectory needs unique joints and at least two points')
        count = len(names)
        times = _array([p['time_from_start'] for p in points], len(points), 'times')
        if times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError('trajectory times must start at zero and increase strictly')
        states = tuple(tuple(_array(p.get(key, [0] * count), count, key)
                             for key in ('positions', 'velocities', 'accelerations')) for p in points)
        if any('positions' not in p for p in points):
            raise ValueError('every point needs positions')
        coefficients = tuple(_segment(states[i], states[i+1], times[i+1] - times[i])
                             for i in range(len(points)-1))
        return cls(tuple(names), tuple(times), coefficients)

    @property
    def duration(self):
        return self.times[-1]

    def sample(self, time):
        if not np.isfinite(time):
            raise ValueError('sample time must be finite')
        index = min(max(int(np.searchsorted(self.times, time, side='right'))-1, 0),
                    len(self.coefficients)-1)
        dt = self.times[index+1] - self.times[index]
        u = np.clip((time-self.times[index])/dt, 0, 1)
        coeff = np.asarray(self.coefficients[index])
        return tuple(poly.polyval(u, poly.polyder(coeff, m=order)) / dt ** order
                     for order in (0, 1, 2))

    def check_limits(self, joints, acceleration_limits=None):
        """Check segment extrema, not only input knots; no hidden limit defaults."""
        acceleration_limits = acceleration_limits or {}
        if len(joints) != len(self.names):
            raise ValueError('limits must match trajectory joint order')
        if not isinstance(acceleration_limits, Mapping) or set(acceleration_limits) - set(self.names):
            raise ValueError('acceleration policy contains unknown joints')
        for name, joint in zip(self.names, joints):
            if name in acceleration_limits:
                value = acceleration_limits[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                    raise ValueError(f'{name}: invalid acceleration limit')
                rated = joint['limits'].get('acceleration')
                if rated is not None and value > rated:
                    raise ValueError(f'{name}: acceleration policy exceeds robot rating')
        for index, values in enumerate(self.coefficients):
            dt = self.times[index+1] - self.times[index]
            for j, joint in enumerate(joints):
                limits = joint['limits']
                for order, key in enumerate(('position', 'velocity', 'acceleration')):
                    coeff = poly.polyder(np.asarray(values)[:, j], m=order) / dt ** order
                    roots = poly.polyroots(poly.polyder(coeff))
                    u = [0, 1] + [r.real for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < 1]
                    values_at_extrema = poly.polyval(u, coeff)
                    self._check_extrema(joint, self.names[j], limits, key,
                                        values_at_extrema, acceleration_limits)

    @staticmethod
    def _check_extrema(joint, name, limits, key, values, accelerations):
        if key == 'position':
            if joint['type'] == 'continuous':
                return
            low, high = limits.get('lower'), limits.get('upper')
            if low is None or high is None:
                raise ValueError(f'{name}: missing position limits')
            outside = np.min(values) < low - 1e-8 or np.max(values) > high + 1e-8
        else:
            limit = accelerations.get(name, limits.get(key)) if key == 'acceleration' else limits.get(key)
            if limit is None:
                raise ValueError(f'{name}: missing {key} limit; supply an explicit test policy')
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError(f'{name}: invalid {key} limit')
            outside = np.max(np.abs(values)) > limit + 1e-8
        if outside:
            raise ValueError(f'{name}: trajectory exceeds {key} limit')
