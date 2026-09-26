"""Generic MuJoCo joint execution and reproducible telemetry, independent of ROS."""
from __future__ import annotations

from datetime import datetime, timezone
import math

import numpy as np

from .joint_trajectory import JointPath


class RobotSimulation:
    """Torque-limited joint servo; floating robots require external balance policies."""

    def __init__(self, robot, simulation=None, environment=(), sensors=()):
        import mujoco
        from .robot_export import build_robot_mjcf
        from .physical_robot import robot_fingerprint
        self.mj = mujoco
        self.robot = robot
        self.settings = dict(simulation or {})
        if self.settings.get('backend', 'mujoco') != 'mujoco':
            raise ValueError('generic runtime supports only the mujoco backend')
        self.m = mujoco.MjModel.from_xml_string(
            build_robot_mjcf(robot, simulation=self.settings, environment=environment, sensors=sensors))
        self.d = mujoco.MjData(self.m)
        self.inverse = mujoco.MjData(self.m)
        self.joints = tuple(j for j in robot['joints'] if j['type'] != 'fixed')
        self.names = tuple(j['name'] for j in self.joints)
        self.qidx = np.array([int(self.m.joint(n).qposadr[0]) for n in self.names])
        self.vidx = np.array([int(self.m.joint(n).dofadr[0]) for n in self.names])
        self.actidx = np.array([self._actuator(n) for n in self.names])
        effort = [j['limits'].get('effort') for j in self.joints]
        if any(v is None or not np.isfinite(v) or v <= 0 for v in effort):
            raise ValueError('controlled simulation requires positive joint effort limits')
        self.effort = np.asarray(effort)
        self.model_hash = robot_fingerprint(robot)
        self.path = None
        self.path_start = 0.0
        self.hold_position = self.d.qpos[self.qidx].copy()
        self.torque = np.zeros(len(self.names))
        self.saturated = False
        self.mj.mj_forward(self.m, self.d)

    def _actuator(self, name):
        joint_id = self.m.joint(name).id
        ids = np.flatnonzero(self.m.actuator_trnid[:, 0] == joint_id)
        if len(ids) != 1:
            raise ValueError(f'{name}: expected one joint actuator')
        return int(ids[0])

    @property
    def time(self):
        return float(self.d.time)

    def set_state(self, positions):
        value = np.asarray(positions, dtype=float)
        if value.shape != (len(self.names),) or not np.all(np.isfinite(value)):
            raise ValueError('initial positions must match robot joint count and be finite')
        for q, j in zip(value, self.joints):
            if j['type'] != 'continuous' and not j['limits']['lower'] <= q <= j['limits']['upper']:
                raise ValueError(f'{j["name"]}: initial state outside position limits')
        self.d.qpos[self.qidx] = value
        self.d.qvel[:] = 0
        self.hold_position = value.copy()
        self.path = None
        self.mj.mj_forward(self.m, self.d)

    def command(self, names, points, acceleration_limits=None, start_tolerance=0.05):
        if set(names) - set(self.names) or len(set(names)) != len(names):
            raise ValueError('trajectory contains unknown or duplicate joints')
        current = self.d.qpos[self.qidx].copy()
        lookup = {name: i for i, name in enumerate(names)}
        expanded = []
        for point in points:
            if len(point.get('positions', [])) != len(names):
                raise ValueError('positions must match trajectory joint names')
            full = {'time_from_start': point['time_from_start']}
            for key in ('positions', 'velocities', 'accelerations'):
                values = point.get(key, [0.0] * len(names))
                if len(values) != len(names):
                    raise ValueError(f'{key}: wrong trajectory dimension')
                full[key] = [values[lookup[n]] if n in lookup else
                             (current[i] if key == 'positions' else 0.0)
                             for i, n in enumerate(self.names)]
            expanded.append(full)
        if expanded and expanded[0]['time_from_start'] > 0:
            expanded = [{'time_from_start': 0, 'positions': current.tolist()}] + expanded
        path = JointPath.from_points(self.names, expanded)
        if np.max(np.abs(path.sample(0)[0] - current)) > start_tolerance:
            raise ValueError('trajectory start differs from current state; replay requires matching start')
        path.check_limits(self.joints, acceleration_limits)
        self.path, self.path_start = path, self.time
        self.apply_control()
        self.mj.mj_forward(self.m, self.d)
        return path

    def desired(self):
        if self.path is None:
            return self.hold_position.copy(), np.zeros(len(self.names)), np.zeros(len(self.names))
        if self.time - self.path_start >= self.path.duration:
            return self.path.sample(self.path.duration)[0], np.zeros(len(self.names)), np.zeros(len(self.names))
        return self.path.sample(self.time - self.path_start)

    def hold(self):
        self.hold_position = self.d.qpos[self.qidx].copy()
        self.path = None

    def apply_control(self):
        q, v, a = self.desired()
        bandwidth = float(self.settings.get('bandwidth_hz', 5.0))
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError('bandwidth_hz must be finite and positive')
        omega = 2 * math.pi * bandwidth
        desired_accel = a + omega ** 2 * (q-self.d.qpos[self.qidx]) + 2*omega*(v-self.d.qvel[self.vidx])
        self.inverse.qpos[:] = self.d.qpos
        self.inverse.qvel[:] = self.d.qvel
        self.inverse.qacc[:] = 0
        self.inverse.qacc[self.vidx] = desired_accel
        self.mj.mj_inverse(self.m, self.inverse)
        required = self.inverse.qfrc_inverse[self.vidx].copy()
        self.torque = np.clip(required, -self.effort, self.effort)
        self.saturated = bool(np.any(np.abs(required) > self.effort + 1e-8))
        self.d.ctrl[self.actidx] = self.torque

    def step(self):
        self.apply_control()
        self.mj.mj_step(self.m, self.d)
        # mj_step leaves Cartesian frames at the prior state; refresh for synchronized records.
        self.mj.mj_forward(self.m, self.d)
        if not np.all(np.isfinite(self.d.qpos)) or any(w.number for w in self.d.warning):
            raise RuntimeError('MuJoCo produced nonfinite state or a numerical warning')

    def endpoint(self, end_effector=None):
        endpoints = self.robot['end_effectors']
        endpoint = endpoints.get(end_effector, end_effector) if end_effector else next(iter(endpoints.values()), None)
        if endpoint is not None and endpoint not in endpoints.values():
            raise ValueError(f'unknown configured end effector: {end_effector}')
        return endpoint

    def sample(self, end_effector=None):
        endpoint = self.endpoint(end_effector)
        record = {'time': self.time, 'joint_position': self.d.qpos[self.qidx].tolist(),
                  'joint_velocity': self.d.qvel[self.vidx].tolist(),
                  'joint_acceleration': self.d.qacc[self.vidx].tolist(),
                  'joint_torque': self.torque.tolist()}
        if endpoint:
            body = self.m.body(endpoint).id
            quat = self.d.xquat[body]
            jacobian = np.zeros((3, self.m.nv))
            self.mj.mj_jacBody(self.m, self.d, jacobian, None, body)
            record.update(tcp_position=self.d.xpos[body].tolist(),
                          tcp_orientation=quat[[1, 2, 3, 0]].tolist(),
                          tcp_velocity=(jacobian @ self.d.qvel).tolist())
        return record


def run_trajectory(sim, points, *, scenario_id, acceleration_limits=None,
                   max_error=0.05, end_effector=None):
    """Execute from the current state; return simulation evidence and saveable record."""
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError('scenario_id must identify the same experiment as the reference')
    if not math.isfinite(max_error) or max_error <= 0:
        raise ValueError('max_error must be positive')
    path = sim.command(sim.names, points, acceleration_limits)
    endpoint = sim.endpoint(end_effector)
    base_state = None
    if sim.robot['base']['type'] == 'floating':
        base_state = {'qpos': sim.d.qpos[:7].tolist(), 'qvel': sim.d.qvel[:6].tolist()}
    start = sim.time
    rows = [sim.sample(end_effector)]
    saturation, contacts, worst = 0, 0, 0.0
    for _ in range(math.ceil(path.duration / sim.m.opt.timestep)):
        sim.step()
        rows.append(sim.sample(end_effector))
        saturation += int(sim.saturated)
        contacts = max(contacts, sim.d.ncon)
        worst = max(worst, float(np.max(np.abs(sim.desired()[0]-sim.d.qpos[sim.qidx]))))
    passed = worst <= max_error and saturation == 0 and contacts == 0
    samples = {key: [row[key] for row in rows] for key in rows[0]}
    samples['time'] = [time - start for time in samples['time']]
    observation = {'schema_version': 1, 'robot': sim.robot['name'], 'scenario_id': scenario_id,
                   'model_sha256': sim.model_hash, 'joint_names': list(sim.names), 'frame': 'world',
                   'joint_types': [j['type'] for j in sim.joints],
                   'units': 'SI', 'evidence': {'kind': 'synthetic', 'source': f'MuJoCo {sim.mj.__version__} simulation'},
                   'samples': samples}
    if endpoint:
        observation = {**observation, 'end_effector': endpoint}
    trajectory = {'schema_version': 1, 'robot': sim.robot['name'], 'model_sha256': sim.model_hash,
                  'scenario_id': scenario_id, 'timestamp': datetime.now(timezone.utc).isoformat(),
                  'joint_names': list(sim.names), 'start_state': list(points[0]['positions']),
                  'target': {'joint_positions': list(points[-1]['positions'])},
                  'trajectory': {'points': [
                      {'time_from_start': float(t), 'positions': path.sample(t)[0].tolist(),
                       'velocities': path.sample(t)[1].tolist(), 'accelerations': path.sample(t)[2].tolist()}
                      for t in path.times]}, 'planner': {'name': 'quintic_joint', 'parameters': {
                      'acceleration_limits': acceleration_limits or {}, 'max_error': max_error,
                      'end_effector': endpoint}},
                  'collision': {'status': 'clear' if contacts == 0 else 'collision'},
                  'execution': {'status': 'succeeded' if passed else 'failed'}, 'benchmark': None}
    if base_state is not None:
        trajectory = {**trajectory, 'base_start_state': base_state}
    return {'passed': passed, 'max_tracking_error': worst, 'saturated_steps': saturation,
            'max_contacts': int(contacts), 'observation': observation, 'trajectory': trajectory,
            'limitations': ['Simulation evidence, not hardware validation.',
                            'Joint servo does not provide floating-base balance or gait control.']}
