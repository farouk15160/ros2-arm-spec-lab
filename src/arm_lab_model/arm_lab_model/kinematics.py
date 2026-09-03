"""Kinematics, inverse dynamics and structural analysis for the configured arm.

Everything here works straight off `ArmConfig`, so it stays valid for whatever
chain the YAML describes -- any number of joints, any tube sizes, any gravity.

Conventions
-----------
All recursions are carried out in the world frame, which keeps the algebra
readable at the cost of a few extra rotations. Gravity enters the inverse
dynamics through the classic trick of accelerating the base upwards by g.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import ArmConfig, JointSpec

POISSON_RATIO = 0.33


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors.

    numpy's generic `cross` spends roughly ten times as long in axis
    normalisation as it does on the arithmetic, and this sits in the innermost
    loop of the Jacobian and the inverse dynamics.
    """
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-9:
        return (math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy),
                math.atan2(R[1, 0], R[0, 0]))
    return (math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0)


def axis_angle_to_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    c, s = math.cos(angle), math.sin(angle)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _tube_body_inertia(link, R_world: np.ndarray) -> np.ndarray:
    """Inertia tensor about the link CoM, expressed in the world frame.

    The principal axes are (perp, perp, along-tube); `R_world` maps the tube's
    local frame -- whose z axis runs along the tube -- into the world.
    """
    ixx, iyy, izz = link.inertia_about_com()
    I_local = np.diag([ixx, iyy, izz])
    return R_world @ I_local @ R_world.T


def _frame_from_direction(direction: np.ndarray) -> np.ndarray:
    """A rotation whose z axis is `direction` (choice of x/y is arbitrary)."""
    z = direction / np.linalg.norm(direction)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = cross3(ref, z)
    x /= np.linalg.norm(x)
    y = cross3(z, x)
    return np.column_stack((x, y, z))


@dataclass
class FrameSet:
    """Everything the analysis needs about one configuration q."""

    joint_origin: np.ndarray      # (n, 3) world position of each joint
    joint_axis: np.ndarray        # (n, 3) world direction of each joint axis
    link_com: np.ndarray          # (n, 3) world CoM of each moving link
    link_mass: np.ndarray         # (n,)
    link_inertia: List[np.ndarray]  # n x (3,3) about CoM, world frame
    link_dir: np.ndarray          # (n, 3) world direction of each tube
    link_frame_R: List[np.ndarray]  # n rotations, tube-local -> world
    distal: np.ndarray            # (n, 3) world position of each link's far end
    tcp: np.ndarray               # (3,)
    tcp_R: np.ndarray             # (3, 3)
    ee_com: np.ndarray            # (3,)
    ee_mass: float
    ee_inertia: np.ndarray        # (3, 3) about the EE CoM, world frame
    base_distal: np.ndarray       # (3,) top of the pedestal
    reach_origin: np.ndarray      # (3,) the "shoulder axis" reach is measured from


class ArmModel:
    """Analysis model built from an :class:`ArmConfig`."""

    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self.n = cfg.dof
        self.joints: List[JointSpec] = cfg.joints
        self.g = np.array([0.0, 0.0, -cfg.gravity])
        self.torque_limits = np.array([j.effort_limit for j in cfg.joints])
        self.continuous_limits = np.array(
            [j.actuator.output_continuous_torque for j in cfg.joints])
        self.velocity_limits = np.array([j.usable_speed for j in cfg.joints])
        self.lower = np.array([j.lower for j in cfg.joints])
        self.upper = np.array([j.upper for j in cfg.joints])
        self.reflected_inertia = np.array(
            [j.actuator.reflected_inertia for j in cfg.joints])
        self.friction = np.array([j.actuator.friction for j in cfg.joints])
        # Constant per-joint geometry, hoisted out of the frames() recursion:
        # these depend only on the config, never on q.
        self._origin_R = [rpy_to_matrix(j.origin_rpy) for j in cfg.joints]
        self._origin_p = [np.asarray(j.origin_xyz, dtype=float) for j in cfg.joints]
        self._axis = [np.asarray(j.axis, dtype=float) for j in cfg.joints]
        self._link_dir = [np.asarray(j.link.direction, dtype=float)
                          for j in cfg.joints]
        self._revolute = [j.jtype != 'prismatic' for j in cfg.joints]
        self._mount_R = rpy_to_matrix(cfg.mount_rpy)
        self._mount_p = np.asarray(cfg.mount_xyz, dtype=float)
        self._pedestal_dir = np.asarray(cfg.pedestal.direction, dtype=float)
        try:
            self.reach_index = cfg.joint_names.index(cfg.reach_reference_joint)
        except ValueError:
            self.reach_index = min(1, self.n - 1)

    # ---------------------------------------------------------------- frames
    def frames(self, q: Sequence[float], light: bool = False) -> FrameSet:
        """Pose of every frame at configuration `q`.

        `light` skips the world-frame inertia tensors, which cost more than the
        rest of the recursion put together and are only needed by the dynamics.
        Anything that consumes `link_inertia` re-derives the full set itself.
        """
        q = np.asarray(q, dtype=float)
        cfg = self.cfg

        R = self._mount_R
        p = self._mount_p
        # Pedestal: a fixed tube from the mount point.
        ped_dir = R @ self._pedestal_dir
        p = p + ped_dir * cfg.pedestal.length
        base_distal = p.copy()

        origins, axes, coms, masses, inertias = [], [], [], [], []
        dirs, frames_R, distals = [], [], []

        for i, joint in enumerate(self.joints):
            R = R @ self._origin_R[i]
            p = p + R @ self._origin_p[i]
            axis_w = R @ self._axis[i]

            if self._revolute[i]:
                # Rotating about its own axis leaves that axis unchanged.
                R = R @ axis_angle_to_matrix(self._axis[i], float(q[i]))
            else:
                p = p + axis_w * float(q[i])

            origins.append(p.copy())
            axes.append(axis_w.copy())

            link = joint.link
            d_w = R @ self._link_dir[i]
            dirs.append(d_w.copy())
            coms.append(p + d_w * link.com_distance)
            masses.append(link.mass)
            if not light:
                R_tube = _frame_from_direction(d_w)
                frames_R.append(R_tube)
                inertias.append(_tube_body_inertia(link, R_tube))
            p = p + d_w * link.length
            distals.append(p.copy())

        ee = cfg.end_effector
        flange_dir = dirs[-1] if dirs else np.array([0.0, 0.0, 1.0])
        tcp = p + flange_dir * ee.tcp_offset
        ee_com = p + flange_dir * ee.com_offset
        ee_mass = ee.mass          # already includes the jaws

        # The tool is a box, not a point. Ignoring its rotational inertia costs
        # nothing in a holding torque, but it is worth about 1e-2 N.m on the
        # wrist joints under acceleration -- and those are the joints with the
        # least torque to spare. The tensor matches the box the URDF exports.
        if light or not dirs:
            ee_inertia = np.zeros((3, 3))
        else:
            # Unlike a tube, the tool box is not transversely isotropic, so the
            # roll of its frame about the tool axis changes the tensor. Build it
            # exactly as the URDF does -- the link frame times the tube frame of
            # the local direction -- rather than from the world direction, whose
            # arbitrary choice of first axis would rotate the box.
            R_ee = R @ _frame_from_direction(
                np.asarray(cfg.joints[-1].link.direction, dtype=float))
            lx, ly, lz = ee.body_width, ee.body_height, ee.body_length
            local = np.diag([ee_mass * (ly * ly + lz * lz) / 12.0,
                             ee_mass * (lx * lx + lz * lz) / 12.0,
                             ee_mass * (lx * lx + ly * ly) / 12.0])
            ee_inertia = R_ee @ local @ R_ee.T

        return FrameSet(
            joint_origin=np.array(origins),
            joint_axis=np.array(axes),
            link_com=np.array(coms),
            link_mass=np.array(masses),
            link_inertia=inertias,
            link_dir=np.array(dirs),
            link_frame_R=frames_R,
            distal=np.array(distals),
            tcp=tcp,
            tcp_R=R.copy(),
            ee_com=ee_com,
            ee_mass=ee_mass,
            ee_inertia=ee_inertia,
            base_distal=base_distal,
            reach_origin=np.array(origins)[self.reach_index],
        )

    def fk(self, q: Sequence[float]) -> np.ndarray:
        return self.frames(q).tcp

    def reach(self, q: Sequence[float], fs: Optional[FrameSet] = None) -> float:
        """Straight-line distance from the shoulder axis to the TCP."""
        fs = fs or self.frames(q)
        return float(np.linalg.norm(fs.tcp - fs.reach_origin))

    @property
    def geometric_max_reach(self) -> float:
        """Sum of link lengths distal of the reach reference joint, plus TCP."""
        total = sum(j.link.length for j in self.joints[self.reach_index:])
        return total + self.cfg.end_effector.tcp_offset

    # ------------------------------------------------------------- jacobian
    def jacobian(self, q: Sequence[float],
                 fs: Optional[FrameSet] = None) -> np.ndarray:
        """6 x n geometric Jacobian at the TCP (linear rows first)."""
        fs = fs or self.frames(q, light=True)
        J = np.zeros((6, self.n))
        for i, joint in enumerate(self.joints):
            z = fs.joint_axis[i]
            if joint.jtype == 'prismatic':
                J[:3, i] = z
            else:
                J[:3, i] = cross3(z, fs.tcp - fs.joint_origin[i])
                J[3:, i] = z
        return J

    # ------------------------------------------------------- inverse dynamics
    def inverse_dynamics(self, q, qd=None, qdd=None,
                         payload: float = 0.0,
                         fs: Optional[FrameSet] = None,
                         include_friction: bool = True) -> np.ndarray:
        """Joint torques via recursive Newton-Euler, in the world frame.

        `payload` is an extra point mass held at the TCP. With qd = qdd = 0 this
        returns the pure gravity (holding) torques.
        """
        q = np.asarray(q, dtype=float)
        qd = np.zeros(self.n) if qd is None else np.asarray(qd, dtype=float)
        qdd = np.zeros(self.n) if qdd is None else np.asarray(qdd, dtype=float)
        fs = fs or self.frames(q)
        if not fs.link_inertia:          # handed a light FrameSet; needs the full one
            fs = self.frames(q)

        n = self.n
        # Bodies: the n links, then the end effector, then the payload.
        omega = np.zeros(3)
        alpha = np.zeros(3)
        # Base accelerates upward by g so that gravity falls out of the maths.
        acc_prev = -self.g
        p_prev = fs.base_distal

        omegas, alphas = [], []
        body_force = []      # net force on each body, world frame
        body_moment = []     # net moment about each body's CoM

        for i in range(n):
            p_i = fs.joint_origin[i]
            z = fs.joint_axis[i]
            r = p_i - p_prev
            acc_joint = acc_prev + cross3(alpha, r) + cross3(omega, cross3(omega, r))

            if self.joints[i].jtype == 'prismatic':
                acc_joint = acc_joint + z * qdd[i] + 2.0 * cross3(omega, z * qd[i])
                omega_i, alpha_i = omega.copy(), alpha.copy()
            else:
                omega_i = omega + z * qd[i]
                alpha_i = alpha + z * qdd[i] + cross3(omega, z * qd[i])

            c = fs.link_com[i] - p_i
            acc_com = (acc_joint + cross3(alpha_i, c)
                       + cross3(omega_i, cross3(omega_i, c)))
            m = fs.link_mass[i]
            I = fs.link_inertia[i]
            body_force.append(m * acc_com)
            body_moment.append(I @ alpha_i + cross3(omega_i, I @ omega_i))
            omegas.append(omega_i)
            alphas.append(alpha_i)

            omega, alpha, acc_prev, p_prev = omega_i, alpha_i, acc_joint, p_i

        # End effector and payload ride rigidly on the last link.
        tip_bodies = []
        if n:
            # The tool carries a real inertia tensor. A bare payload at the TCP
            # stays a point mass, which is the conservative reading of "2 kg at
            # the tool centre point" when its shape is unknown.
            for mass, pos, I_body in ((fs.ee_mass, fs.ee_com, fs.ee_inertia),
                                      (payload, fs.tcp, None)):
                if mass <= 0.0:
                    continue
                c = pos - p_prev
                acc = (acc_prev + cross3(alpha, c)
                       + cross3(omega, cross3(omega, c)))
                moment = np.zeros(3)
                if I_body is not None and I_body.any():
                    moment = I_body @ alpha + cross3(omega, I_body @ omega)
                tip_bodies.append((mass * acc, moment, pos))

        # Backward pass.
        f = np.zeros(3)
        nm = np.zeros(3)
        p_next = None
        tau = np.zeros(n)
        for i in range(n - 1, -1, -1):
            p_i = fs.joint_origin[i]
            if i == n - 1:
                for F_t, N_t, pos_t in tip_bodies:
                    nm = nm + cross3(pos_t - p_i, F_t) + N_t
                    f = f + F_t
            elif p_next is not None:
                nm = nm + cross3(p_next - p_i, f)

            F_i = body_force[i]
            nm = nm + body_moment[i] + cross3(fs.link_com[i] - p_i, F_i)
            f = f + F_i

            z = fs.joint_axis[i]
            tau[i] = float(f @ z) if self.joints[i].jtype == 'prismatic' else float(nm @ z)
            tau[i] += self.reflected_inertia[i] * qdd[i]
            if include_friction:
                tau[i] += self.friction[i] * math.tanh(qd[i] / 0.02)
            p_next = p_i

        return tau

    def gravity_torque(self, q, payload: float = 0.0,
                       fs: Optional[FrameSet] = None) -> np.ndarray:
        return self.inverse_dynamics(q, payload=payload, fs=fs, include_friction=False)

    # ------------------------------------------------------ payload capacity
    def payload_capacity(self, q, fs: Optional[FrameSet] = None,
                         reserve: Optional[float] = None) -> Tuple[float, int]:
        """Largest static payload holdable at the TCP, and the limiting joint.

        Gravity torque is affine in the payload mass, so this is exact and needs
        no search:  tau_i(m) = A_i + m * B_i,  |tau_i| <= tau_avail_i.
        """
        fs = fs or self.frames(q)
        if reserve is None:
            reserve = float(self.cfg.control.get('dynamic_torque_reserve', 0.0))
        avail = self.torque_limits * (1.0 - reserve)

        A = self.gravity_torque(q, payload=0.0, fs=fs)
        B = np.zeros(self.n)
        for i, joint in enumerate(self.joints):
            z = fs.joint_axis[i]
            r = fs.tcp - fs.joint_origin[i]
            if joint.jtype == 'prismatic':
                B[i] = float(z @ (-self.g)) * 1.0
            else:
                # d(tau_i)/dm for a point mass at the TCP.
                B[i] = float(z @ cross3(r, -self.g))

        best = math.inf
        limiting = -1
        for i in range(self.n):
            if abs(B[i]) < 1e-9:
                if abs(A[i]) > avail[i]:
                    return 0.0, i      # already overloaded by its own weight
                continue
            for bound in (avail[i], -avail[i]):
                m = (bound - A[i]) / B[i]
                if m >= 0.0 and m < best:
                    best, limiting = m, i
            # A negative-only solution means this joint is already saturated.
            if abs(A[i]) > avail[i]:
                return 0.0, i
        if not math.isfinite(best):
            return math.inf, -1
        return float(best), limiting

    # ----------------------------------------------------------- tcp speed
    def max_tcp_speed(self, q, fs: Optional[FrameSet] = None) -> Tuple[float, np.ndarray]:
        """Fastest TCP speed reachable at this pose within joint speed limits.

        The reachable velocity set is a zonotope, so its extreme point is a
        vertex of the joint-velocity box.
        """
        fs = fs or self.frames(q)
        Jv = self.jacobian(q, fs)[:3, :]
        w = self.velocity_limits
        if self.n <= 14:
            best, best_qd = 0.0, np.zeros(self.n)
            for signs in itertools.product((1.0, -1.0), repeat=self.n):
                qd = np.array(signs) * w
                s = float(np.linalg.norm(Jv @ qd))
                if s > best:
                    best, best_qd = s, qd
            return best, best_qd
        # Fall back to the sign heuristic for very long chains.
        u, _, _ = np.linalg.svd(Jv)
        d = u[:, 0]
        qd = np.sign(Jv.T @ d) * w
        return float(np.linalg.norm(Jv @ qd)), qd

    def tcp_velocity(self, q, qd, fs: Optional[FrameSet] = None) -> np.ndarray:
        fs = fs or self.frames(q)
        return self.jacobian(q, fs)[:3, :] @ np.asarray(qd, dtype=float)

    # -------------------------------------------------------- contact force
    def max_contact_force(self, q, direction: Optional[Sequence[float]] = None,
                          fs: Optional[FrameSet] = None) -> float:
        """Force the arm can exert at the TCP along `direction`, after holding
        its own weight. tau = J^T F, so F_max = min_i (headroom_i / |J^T_i . d|).
        """
        fs = fs or self.frames(q)
        if direction is None:
            direction = [0.0, 0.0, -1.0]
        d = np.asarray(direction, dtype=float)
        d = d / (np.linalg.norm(d) or 1.0)
        Jt = self.jacobian(q, fs)[:3, :].T @ d
        hold = np.abs(self.gravity_torque(q, fs=fs))
        headroom = np.maximum(self.torque_limits - hold, 0.0)
        best = math.inf
        for i in range(self.n):
            if abs(Jt[i]) > 1e-9:
                best = min(best, headroom[i] / abs(Jt[i]))
        return float(best) if math.isfinite(best) else 0.0

    # ------------------------------------------------- structural deflection
    def deflection(self, q, payload: float = 0.0,
                   fs: Optional[FrameSet] = None,
                   direction: Optional[Sequence[float]] = None,
                   samples: int = 24) -> dict:
        """TCP sag under load, split into tube bending, torsion and joint droop.

        Uses the unit-load (Castigliano) method along each tube, plus the
        elastic wind-up of each gearbox. This is the number that decides whether
        a +/-10 mm raw positioning accuracy is even physically available.
        """
        fs = fs or self.frames(q)
        if direction is None:
            direction = [0.0, 0.0, -1.0]
        e = np.asarray(direction, dtype=float)
        e = e / (np.linalg.norm(e) or 1.0)

        # Total tip load: payload weight only (self-weight handled separately
        # below by lumping each link's weight at its CoM).
        F_tip = payload * self.cfg.gravity
        loads = [(F_tip * -self.g / self.cfg.gravity, fs.tcp)] if payload > 0 else []
        loads.append((fs.ee_mass * self.g, fs.ee_com))
        for i in range(self.n):
            loads.append((fs.link_mass[i] * self.g, fs.link_com[i]))

        bending = 0.0
        torsion = 0.0
        stress = np.zeros(self.n)
        for i, joint in enumerate(self.joints):
            link = joint.link
            E = link.material.youngs_modulus
            I = link.second_moment_area
            G = E / (2.0 * (1.0 + POISSON_RATIO))
            Jp = 2.0 * I
            if I <= 0.0 or link.length <= 0.0:
                continue
            u = fs.link_dir[i]
            p0 = fs.joint_origin[i]
            ds = link.length / samples
            m_peak = 0.0
            for k in range(samples):
                s = (k + 0.5) * ds
                x = p0 + u * s
                # Real bending moment at this station from every load outboard.
                M = np.zeros(3)
                for F, pos in loads:
                    along = float((pos - p0) @ u)
                    if along > s:                       # only outboard loads
                        M += cross3(pos - x, F)
                # Virtual moment from a unit tip load along e.
                m_virt = cross3(fs.tcp - x, e)
                M_perp = M - float(M @ u) * u
                m_perp = m_virt - float(m_virt @ u) * u
                bending += float(M_perp @ m_perp) / (E * I) * ds
                torsion += float(M @ u) * float(m_virt @ u) / (G * Jp) * ds
                m_peak = max(m_peak, float(np.linalg.norm(M_perp)))
            stress[i] = m_peak / link.section_modulus if link.section_modulus > 0 else 0.0

        # Gearbox / joint elasticity: each joint winds up by tau/k.
        tau = self.gravity_torque(q, payload=payload, fs=fs)
        joint_droop = 0.0
        wind_up = np.zeros(self.n)
        for i, joint in enumerate(self.joints):
            k = joint.actuator.joint_stiffness
            if k <= 0.0:
                continue
            wind_up[i] = tau[i] / k
            z = fs.joint_axis[i]
            r = fs.tcp - fs.joint_origin[i]
            joint_droop += wind_up[i] * float(cross3(z, r) @ e)

        total = abs(bending) + abs(torsion) + abs(joint_droop)
        return {
            'bending': abs(bending),
            'torsion': abs(torsion),
            'joint_compliance': abs(joint_droop),
            'total': total,
            'joint_wind_up': wind_up,
            'bending_stress': stress,
            'stress_utilisation': np.array([
                stress[i] / self.joints[i].link.material.yield_strength
                for i in range(self.n)]),
        }

    # --------------------------------------------------------------- power
    def power(self, q, qd, payload: float = 0.0,
              fs: Optional[FrameSet] = None) -> dict:
        """Electrical power estimate: mechanical work / efficiency + overhead."""
        fs = fs or self.frames(q)
        qd = np.asarray(qd, dtype=float)
        tau = self.inverse_dynamics(q, qd=qd, payload=payload, fs=fs)
        mech = np.abs(tau * qd)
        elec = np.zeros(self.n)
        for i, joint in enumerate(self.joints):
            act = joint.actuator
            # Copper loss scales with torque^2; approximate it from the peak
            # operating point where the actuator is at its continuous rating.
            load = abs(tau[i]) / max(act.output_continuous_torque, 1e-6)
            elec[i] = mech[i] / max(act.efficiency, 1e-3) \
                + act.quiescent_power * (1.0 + 0.5 * load ** 2)
        buses = {}
        for i, joint in enumerate(self.joints):
            buses.setdefault(joint.actuator.bus_voltage, 0.0)
            buses[joint.actuator.bus_voltage] += elec[i]
        return {
            'joint_mech_w': mech,
            'joint_elec_w': elec,
            'total_w': float(elec.sum()),
            'per_bus_w': buses,
            'per_bus_a': {v: p / v for v, p in buses.items()},
            'torque': tau,
        }

    # ------------------------------------------------------------------- IK
    def ik_position(self, target: Sequence[float], q0: Optional[Sequence[float]] = None,
                    iterations: int = 300, damping: float = 0.05,
                    centring_gain: float = 0.35,
                    active: Optional[Sequence[bool]] = None) -> np.ndarray:
        """Damped least-squares position IK, clamped to the joint limits.

        A position task leaves three degrees of freedom unconstrained on a
        six-joint arm, and without a preference the solver is free to leave the
        wrist folded back into the forearm. A joint-centring term in the
        nullspace takes up that slack, pulling unused joints toward the middle
        of their travel without disturbing the TCP, which yields the laid-out
        poses the test cases are supposed to represent.

        `active` pins joints out of the solve: pass a boolean mask to let only
        some of them move, which turns an underdetermined task into a
        well-posed one.

        Aimed at an unreachable point it converges to the stretched-out pose,
        which is exactly what the "full reach" test case needs.
        """
        q = np.array(q0 if q0 is not None else np.clip(
            np.zeros(self.n), self.lower, self.upper), dtype=float)
        target = np.asarray(target, dtype=float)
        mask = (np.ones(self.n, dtype=bool) if active is None
                else np.asarray(active, dtype=bool))
        mid = 0.5 * (self.lower + self.upper)
        span = np.maximum(self.upper - self.lower, 1e-6)
        eye = np.eye(self.n)
        for _ in range(iterations):
            fs = self.frames(q, light=True)
            err = target - fs.tcp
            J = self.jacobian(q, fs)[:3, :]
            J = J * mask                      # frozen joints contribute nothing
            JT = J.T
            inv = np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(3))
            dq = JT @ (inv @ err)
            if centring_gain > 0.0:
                null = eye - (JT @ inv) @ J
                dq = dq + null @ (centring_gain * (mid - q) / span)
            step = np.clip(dq, -0.2, 0.2) * mask
            if np.linalg.norm(err) < 1e-6 and np.linalg.norm(step) < 1e-6:
                break
            q = np.clip(q + step, self.lower, self.upper)
        return q

    def resolve_pose(self, spec, _seen=None) -> np.ndarray:
        """Turn a pose entry from the config into joint angles.

        Accepts a list of angles, the name of an entry in `test_poses`,
        `auto_full_reach`, or `auto_reach_<metres>`.
        """
        if isinstance(spec, (list, tuple, np.ndarray)):
            vals = [float(v) for v in spec]
            vals += [0.0] * (self.n - len(vals))
            return np.clip(np.array(vals[:self.n]), self.lower, self.upper)
        if isinstance(spec, str):
            if spec == 'auto_full_reach':
                return self.stretched_pose(self.geometric_max_reach * 1.5)
            if spec.startswith('auto_reach_'):
                return self.stretched_pose(float(spec[len('auto_reach_'):]))
            seen = _seen or set()
            if spec in self.cfg.test_poses and spec not in seen:
                seen.add(spec)
                return self.resolve_pose(self.cfg.test_poses[spec], seen)
        raise ValueError(
            f'cannot interpret pose {spec!r}; known poses: '
            f'{sorted(self.cfg.test_poses)}')

    def stretched_pose(self, radius: float) -> np.ndarray:
        """Pose reaching `radius` from the shoulder axis, arm laid out straight.

        A pure position target on a six-joint arm leaves three degrees of
        freedom free, and the solver will spend them folding the forearm back
        across the wrist: the right radius, but a pose that self-collides and
        misrepresents the load.

        Pinning the wrist at zero does not help either, because the joint origin
        rotations mean a zeroed wrist is already bent. So the task here is
        position *plus* alignment of the tool axis with the direction being
        reached, which is what "laid out straight" actually means and which
        leaves no slack to fold.
        """
        fs0 = self.frames(np.zeros(self.n), light=True)
        origin = fs0.reach_origin
        direction = np.array([1.0, 0.0, 0.0])
        target = origin + direction * radius

        best_q, best_score = None, math.inf
        seeds = [np.zeros(self.n)]
        for a, b in ((0.3, -0.6), (-0.3, 0.6), (0.8, -1.2), (-0.8, 1.2)):
            seed = np.zeros(self.n)
            if self.n > 1:
                seed[1] = a
            if self.n > 2:
                seed[2] = b
            seeds.append(seed)

        for seed in seeds:
            q = np.clip(seed.copy(), self.lower, self.upper)
            for _ in range(400):
                fs = self.frames(q, light=True)
                J = self.jacobian(q, fs)
                tool = fs.link_dir[-1]
                e_pos = target - fs.tcp
                # Rotate the tool axis onto the reach direction. The cross
                # product vanishes exactly when they are parallel, and it has no
                # component about the tool axis, so spin stays free.
                e_rot = cross3(tool, direction)
                e = np.concatenate([e_pos, 0.15 * e_rot])
                Jt = np.vstack([J[:3, :], 0.15 * J[3:, :]])
                JT = Jt.T
                dq = JT @ np.linalg.solve(
                    Jt @ JT + (0.05 ** 2) * np.eye(6), e)
                step = np.clip(dq, -0.2, 0.2)
                if np.linalg.norm(step) < 1e-9:
                    break
                q = np.clip(q + step, self.lower, self.upper)

            fs = self.frames(q, light=True)
            reached = float(np.linalg.norm(fs.tcp - origin))
            score = abs(reached - radius)
            score += 0.5 * abs(fs.tcp[2] - origin[2])
            score += 0.3 * float(np.linalg.norm(cross3(fs.link_dir[-1], direction)))
            if score < best_score:
                best_q, best_score = q, score
        return best_q
