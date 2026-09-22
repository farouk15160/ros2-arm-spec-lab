"""ROS-free core of the interactive MuJoCo simulator.

`ArmSimulation` owns the MuJoCo model of the arm, its jaws and the world's
sample boxes, and steps it under a joint servo. `TrajectorySampler` turns a
joint trajectory into desired position, velocity and acceleration the way
ros2_control's joint_trajectory_controller does: a cubic Hermite segment when
both ends carry velocities, a straight line when they do not, and the last
point held afterwards. The ROS node in arm_lab_gui only wires these to topics.

The servo is computed-torque control using MuJoCo's own mass matrix and bias
forces, with PID gains placed at a chosen bandwidth and the result clipped to
each actuator's torque limit and torque-speed envelope. It stands in for a
well-tuned drive; it is not the Gazebo ros2_control PID, so tracking and
torque traces differ between the two benches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .actuator_model import ActuatorElectrical
from .kinematics import ArmModel
from .mujoco_backend import _engine, build_mjcf, load_world


def _full_inertia(mujoco, m, d, out):
    try:
        mujoco.mj_fullM(m, d, out)          # MuJoCo 3.3 and later
    except TypeError:
        mujoco.mj_fullM(m, out, d.qM)       # 3.2


@dataclass
class TrajectoryPoint:
    time: float                           # s from the start of the trajectory
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None


class TrajectorySampler:
    """Desired joint state along a trajectory that starts at `start_time`."""

    def __init__(self, start_time: float, q0: Sequence[float], v0: Sequence[float],
                 points: List[TrajectoryPoint]):
        self.start_time = float(start_time)
        knots = [TrajectoryPoint(0.0, np.asarray(q0, float), np.asarray(v0, float))]
        for p in sorted(points, key=lambda p: p.time):
            if p.time <= knots[-1].time + 1e-9:
                # A point at (or before) the previous time replaces it: the
                # controller jumps there, as a zero-length segment would.
                knots[-1] = TrajectoryPoint(knots[-1].time, p.positions, p.velocities)
            else:
                knots.append(p)
        self.knots = knots

    @property
    def duration(self) -> float:
        return self.knots[-1].time

    def done(self, t: float) -> bool:
        return t - self.start_time >= self.duration

    def sample(self, t: float):
        tau = t - self.start_time
        n = len(self.knots[0].positions)
        if tau >= self.duration or len(self.knots) == 1:
            return self.knots[-1].positions.copy(), np.zeros(n), np.zeros(n)
        tau = max(tau, 0.0)
        k = next(i for i in range(len(self.knots) - 1) if tau < self.knots[i + 1].time)
        a, b = self.knots[k], self.knots[k + 1]
        h = b.time - a.time
        s = (tau - a.time) / h
        if a.velocities is None or b.velocities is None:
            v = (b.positions - a.positions) / h
            return a.positions + v * s * h, v, np.zeros(n)
        p0, p1, m0, m1 = a.positions, b.positions, a.velocities * h, b.velocities * h
        q = ((2*s**3 - 3*s**2 + 1) * p0 + (s**3 - 2*s**2 + s) * m0
             + (-2*s**3 + 3*s**2) * p1 + (s**3 - s**2) * m1)
        dq = ((6*s**2 - 6*s) * p0 + (3*s**2 - 4*s + 1) * m0
              + (-6*s**2 + 6*s) * p1 + (3*s**2 - 2*s) * m1) / h
        ddq = ((12*s - 6) * p0 + (6*s - 4) * m0
               + (-12*s + 6) * p1 + (6*s - 2) * m1) / h**2
        return q, dq, ddq


class ArmSimulation:
    """The arm, its jaws and the world, stepped under a joint servo."""

    def __init__(self, cfg, *, payload: float = 0.0, world: Optional[str] = None,
                 initial_pose='home', timestep: float = 0.001,
                 bandwidth_hz: float = 5.0):
        mujoco = _engine()
        self.mujoco = mujoco
        self.cfg = cfg
        self.model = ArmModel(cfg)
        self.scene = load_world(world) if world else None
        self.xml = build_mjcf(cfg, payload, timestep, fingers=True, scene=self.scene)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        self.n = cfg.dof
        self.qpos = np.array([self.m.joint(j).qposadr[0] for j in cfg.joint_names])
        self.dof = np.array([self.m.joint(j).dofadr[0] for j in cfg.joint_names])
        self.arm_ctrl = np.array([self.m.actuator(j + '_motor').id for j in cfg.joint_names])
        ee = cfg.end_effector
        self.finger_names = list(ee.finger_joint_names)
        self.finger_qpos = [self.m.joint(j).qposadr[0] for j in self.finger_names]
        self.finger_dof = [self.m.joint(j).dofadr[0] for j in self.finger_names]
        self.finger_ctrl = [self.m.actuator(j + '_motor').id for j in self.finger_names]
        self.force_grip = ee.grasp_mode == 'force'

        q0 = self.model.resolve_pose(initial_pose)
        self.d.qpos[self.qpos] = q0
        for adr in self.finger_qpos:
            self.d.qpos[adr] = ee.stroke / 2.0         # start open
        mujoco.mj_forward(self.m, self.d)
        self.trajectory = TrajectorySampler(0.0, q0, np.zeros(self.n), [])
        self.generation = 0                            # bumps on every new command

        # PID placed at (s + w)^3, per unit of joint inertia.
        w = 2.0 * math.pi * bandwidth_hz
        self.kp, self.kd, self.ki = 3.0 * w * w, 3.0 * w, w ** 3
        self.integral = np.zeros(self.n)
        self.limits = np.array([j.effort_limit for j in cfg.joints])
        self.electrical = []
        for j in cfg.joints:
            a = j.actuator
            ok = min(a.torque_constant, a.phase_resistance, a.max_phase_current) > 0
            self.electrical.append(ActuatorElectrical(
                a.torque_constant, a.phase_resistance, a.bus_voltage,
                a.max_phase_current, a.gear_ratio, a.efficiency,
                peak_output_torque=a.output_peak_torque) if ok else None)
        self.grip_command = np.full(len(self.finger_names),
                                    ee.grip_force_min if self.force_grip else ee.stroke / 2.0)
        self._jaw_damping()
        self.applied = np.zeros(self.n)
        self.saturated = np.zeros(self.n, dtype=bool)
        self._M = np.zeros((self.m.nv, self.m.nv))

    # ------------------------------------------------------------ commands
    @property
    def time(self) -> float:
        return float(self.d.time)

    def desired(self):
        return self.trajectory.sample(self.time)

    def set_trajectory(self, joint_names: Sequence[str], points: List[TrajectoryPoint]):
        """Replace the active trajectory, starting from the current desired state.

        Joints left out of `joint_names` hold their current desired position.
        An empty point list stops the arm where it is commanded now.
        """
        index = {name: i for i, name in enumerate(self.cfg.joint_names)}
        unknown = [name for name in joint_names if name not in index]
        if unknown:
            raise ValueError(f'unknown joints {unknown}; have {self.cfg.joint_names}')
        q_d, v_d, _ = self.desired()
        if not points:
            v_d = np.zeros(self.n)
        full = []
        for p in points:
            q = q_d.copy()
            v = np.zeros(self.n) if p.velocities is not None else None
            for k, name in enumerate(joint_names):
                q[index[name]] = p.positions[k]
                if v is not None:
                    v[index[name]] = p.velocities[k]
            full.append(TrajectoryPoint(p.time, np.clip(q, self.model.lower,
                                                        self.model.upper), v))
        self.trajectory = TrajectorySampler(self.time, q_d, v_d, full)
        self.generation += 1

    def hold(self):
        self.set_trajectory([], [])

    def set_gripper(self, values: Sequence[float]):
        """Force mode: squeeze force per jaw, + opens. Position mode: half-opening."""
        values = list(values)
        if len(values) == 1:
            values = values * len(self.finger_names)
        if len(values) != len(self.finger_names):
            raise ValueError(f'expected {len(self.finger_names)} gripper values')
        self.grip_command = np.asarray(values, float)
        self._jaw_damping()

    def _jaw_damping(self):
        """Give the jaws a drive's force-speed line: full force at stall,
        `grip_speed` with no load.

        A bare force on a 60 g jaw slams it shut at metres per second, fast
        enough to tunnel through a box in one step when only the finger tips
        overlap it. The slope is written as joint damping so MuJoCo integrates
        it implicitly; an explicit c*v term this stiff would be unstable.
        """
        if not self.finger_dof:
            return
        ee = self.cfg.end_effector
        force = (np.maximum(np.abs(self.grip_command), ee.grip_force_min)
                 if self.force_grip else np.full(len(self.finger_dof), ee.grip_force_max))
        self.m.dof_damping[self.finger_dof] = force / max(ee.grip_speed, 1e-3)

    # ----------------------------------------------------------------- step
    def step(self):
        mujoco, m, d = self.mujoco, self.m, self.d
        mujoco.mj_step1(m, d)
        q, v = d.qpos[self.qpos], d.qvel[self.dof]
        q_d, v_d, a_d = self.desired()
        error = q_d - q
        _full_inertia(mujoco, m, d, self._M)
        M = self._M[np.ix_(self.dof, self.dof)]
        command = a_d + self.kp * error + self.kd * (v_d - v) + self.ki * self.integral
        requested = M @ command + d.qfrc_bias[self.dof]
        limits = np.array([min(lim, e.available_torque(v[i]) if e else lim)
                           for i, (lim, e) in enumerate(zip(self.limits, self.electrical))])
        self.saturated = np.abs(requested) > limits
        self.applied = np.clip(requested, -limits, limits)
        # Anti-windup: stop integrating a joint that is already saturated.
        self.integral += np.where(self.saturated, 0.0, error) * m.opt.timestep
        d.ctrl[self.arm_ctrl] = self.applied
        if self.finger_ctrl:
            d.ctrl[self.finger_ctrl] = self.grip_command
        mujoco.mj_step2(m, d)

    # ---------------------------------------------------------------- state
    def joint_state(self):
        """Names, positions, velocities and efforts, arm joints then jaws."""
        d = self.d
        names = list(self.cfg.joint_names) + self.finger_names
        pos = list(d.qpos[self.qpos]) + [d.qpos[a] for a in self.finger_qpos]
        vel = list(d.qvel[self.dof]) + [d.qvel[a] for a in self.finger_dof]
        eff = list(self.applied) + [d.actuator_force[a] for a in self.finger_ctrl]
        return names, [float(x) for x in pos], [float(x) for x in vel], [float(x) for x in eff]

    def object_poses(self):
        """(box, position, quaternion w-x-y-z) for each free box in the world."""
        if not self.scene:
            return []
        return [(box, self.d.body(box.name).xpos.copy(), self.d.body(box.name).xquat.copy())
                for box in self.scene.boxes]

    def tracking_error(self) -> np.ndarray:
        return self.desired()[0] - self.d.qpos[self.qpos]
