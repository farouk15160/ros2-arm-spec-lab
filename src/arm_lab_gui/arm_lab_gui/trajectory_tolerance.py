"""Immutable FollowJointTrajectory tolerances, independent of ROS middleware."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TolerancePolicy:
    names: tuple
    path: tuple
    goal: tuple
    goal_time: float


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(field + ': expected finite number')
    return float(value)


def _entries(names, entries, defaults):
    by_name = {}
    for entry in entries:
        if entry.get('name') not in names or entry['name'] in by_name:
            raise ValueError('tolerance references an unknown or duplicate joint')
        values = []
        for key, default in zip(('position', 'velocity', 'acceleration'), defaults):
            value = _finite(entry.get(key, 0.), 'tolerance.' + key)
            if value < 0 and value != -1:
                raise ValueError('tolerance: only -1 (disable), 0 (default) or positive allowed')
            values.append(None if value == -1 else default if value == 0 else value)
        by_name = {**by_name, entry['name']: tuple(values)}
    return tuple(by_name.get(name, defaults) for name in names)


def tolerance_policy(names, *, path=(), goal=(), goal_time=0., path_position=.05,
                     goal_position=.02, goal_velocity=.01, default_goal_time=1.):
    """Zero requests select explicit defaults; -1 disables that joint component."""
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError('tolerances require unique nonempty trajectory joint names')
    for value, field in ((path_position, 'path_position'), (goal_position, 'goal_position'),
                         (goal_velocity, 'goal_velocity'), (default_goal_time, 'default_goal_time')):
        if _finite(value, field) <= 0:
            raise ValueError(field + ': default must be positive')
    goal_time = _finite(goal_time, 'goal_time')
    if goal_time < 0:
        raise ValueError('goal_time must not be negative')
    return TolerancePolicy(names, _entries(names, path, (path_position, None, None)),
                           _entries(names, goal, (goal_position, goal_velocity, None)),
                           goal_time or float(default_goal_time))


def tracking_violation(policy, desired, actual, *, final):
    """Return first violated joint/component, or None. Vectors follow joint order."""
    limits = policy.goal if final else policy.path
    if len(desired) != 3 or len(actual) != 3:
        raise ValueError('tracking state requires position, velocity and acceleration vectors')
    if any(len(vector) != len(policy.names) for vector in tuple(desired) + tuple(actual)):
        raise ValueError('tracking state dimension differs from trajectory joints')
    for index, name in enumerate(policy.names):
        for component, label in enumerate(('position', 'velocity', 'acceleration')):
            expected = _finite(float(desired[component][index]), 'desired.' + label)
            observed = _finite(float(actual[component][index]), 'actual.' + label)
            bound = limits[index][component]
            error = abs(expected - observed)
            if bound is not None and error > bound:
                return f'{name} {label} error {error:.6g} exceeds {bound:.6g}'
    return None


def validate_terminal_state(point):
    """A stopped terminal velocity is required for the controller's position hold.

    The final acceleration may be the incoming segment's one-sided acceleration;
    the runtime changes acceleration to zero once holding the endpoint.
    """
    if point.get('effort'):
        raise ValueError('effort trajectory commands are unsupported by the position servo')
    if any(abs(_finite(value, 'terminal velocity')) > 1e-6 for value in point.get('velocities', ())):
        raise ValueError('nonzero terminal velocity is unsupported; end trajectory at rest')
