"""Independent verification of the physics.

Everything in this workspace rests on the inverse dynamics being right, so it is
checked against sources that share none of its code:

KDL
    Orocos KDL's recursive Newton-Euler solver, written by a different team, fed
    the *generated URDF* rather than the internal model. Agreement there checks
    the dynamics and the URDF export at the same time.

energy
    Mechanical power delivered at the joints must equal the rate of change of
    kinetic plus potential energy. That is a statement about the whole model at
    once and is derived from the Lagrangian side of mechanics, not the
    Newton-Euler side.

analytic
    Closed-form results for cases simple enough to write down: a point mass on a
    massless arm, and a uniform rod.

Run it with `ros2 run arm_lab_model verify_physics`.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ArmConfig, load_config
from .kinematics import ArmModel, rpy_to_matrix
from .urdf_builder import build_urdf

try:
    import PyKDL as kdl
    HAVE_KDL = True
except ImportError:                                   # pragma: no cover
    HAVE_KDL = False


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    worst: float = 0.0
    tolerance: float = 0.0


# ------------------------------------------------------------------- KDL
def _rpy(text: Optional[str]) -> Tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    r, p, y = (float(v) for v in text.split())
    return (r, p, y)


def _xyz(text: Optional[str]) -> Tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    x, y, z = (float(v) for v in text.split())
    return (x, y, z)


def kdl_chain_from_urdf(urdf_xml: str, base: str = 'base_link',
                        tip: str = 'tcp_link'):
    """Build a KDL chain from URDF text, the way kdl_parser does.

    Written out here because kdl_parser's Python bindings are not packaged for
    this distribution. Correctness is not assumed: `compare_forward_kinematics`
    checks the resulting chain against the model's own FK before any dynamics
    comparison is trusted.
    """
    root = ET.fromstring(urdf_xml)
    links: Dict[str, ET.Element] = {e.get('name'): e for e in root.findall('link')}
    joints = root.findall('joint')
    by_child: Dict[str, ET.Element] = {
        j.find('child').get('link'): j for j in joints}

    # Walk from the tip back to the base to get the chain order.
    order: List[ET.Element] = []
    node = tip
    while node != base:
        joint = by_child.get(node)
        if joint is None:
            raise ValueError(f'no path from {tip} back to {base} (stuck at {node})')
        order.append(joint)
        node = joint.find('parent').get('link')
    order.reverse()

    chain = kdl.Chain()
    for joint in order:
        origin = joint.find('origin')
        xyz = _xyz(origin.get('xyz') if origin is not None else None)
        rpy = _rpy(origin.get('rpy') if origin is not None else None)
        frame = kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))

        jtype = joint.get('type')
        name = joint.get('name')
        if jtype in ('revolute', 'continuous'):
            axis = _xyz(joint.find('axis').get('xyz'))
            kdl_joint = kdl.Joint(name, frame.p, frame.M * kdl.Vector(*axis),
                                  kdl.Joint.RotAxis)
        elif jtype == 'prismatic':
            axis = _xyz(joint.find('axis').get('xyz'))
            kdl_joint = kdl.Joint(name, frame.p, frame.M * kdl.Vector(*axis),
                                  kdl.Joint.TransAxis)
        else:
            kdl_joint = kdl.Joint(name, kdl.Joint.Fixed)

        child = links[joint.find('child').get('link')]
        inertial = child.find('inertial')
        if inertial is None:
            inertia = kdl.RigidBodyInertia(0.0, kdl.Vector(0, 0, 0),
                                           kdl.RotationalInertia())
        else:
            mass = float(inertial.find('mass').get('value'))
            iorigin = inertial.find('origin')
            com = _xyz(iorigin.get('xyz') if iorigin is not None else None)
            irpy = _rpy(iorigin.get('rpy') if iorigin is not None else None)
            tag = inertial.find('inertia')
            diag = np.array([[float(tag.get('ixx')), float(tag.get('ixy')),
                              float(tag.get('ixz'))],
                             [float(tag.get('ixy')), float(tag.get('iyy')),
                              float(tag.get('iyz'))],
                             [float(tag.get('ixz')), float(tag.get('iyz')),
                              float(tag.get('izz'))]])
            # URDF states the tensor in the inertial frame; KDL wants it about
            # the centre of mass but oriented with the link frame.
            R = rpy_to_matrix(irpy)
            I = R @ diag @ R.T
            inertia = kdl.RigidBodyInertia(
                mass, kdl.Vector(*com),
                kdl.RotationalInertia(I[0, 0], I[1, 1], I[2, 2],
                                      I[0, 1], I[0, 2], I[1, 2]))
        chain.addSegment(kdl.Segment(child.get('name'), kdl_joint, frame, inertia))
    return chain


def _to_jnt(values: Sequence[float]):
    array = kdl.JntArray(len(values))
    for i, v in enumerate(values):
        array[i] = float(v)
    return array


def _fingerless(cfg: ArmConfig) -> ArmConfig:
    """A copy with the jaws folded into the gripper body.

    KDL chains are serial, so branching finger links would simply be dropped and
    the comparison would be against a lighter arm.
    """
    out = copy.deepcopy(cfg)
    out.end_effector.simulate_fingers = False
    return out


def compare_forward_kinematics(cfg: ArmConfig, samples: int = 200,
                               seed: int = 0) -> Check:
    """The chain must agree before its dynamics mean anything."""
    model = ArmModel(cfg)
    chain = kdl_chain_from_urdf(build_urdf(cfg, fixed_to_world=False))
    solver = kdl.ChainFkSolverPos_recursive(chain)
    rng = np.random.default_rng(seed)

    worst = 0.0
    for _ in range(samples):
        q = rng.uniform(model.lower, model.upper)
        frame = kdl.Frame()
        solver.JntToCart(_to_jnt(q), frame)
        theirs = np.array([frame.p[0], frame.p[1], frame.p[2]])
        mine = model.frames(q, light=True).tcp
        worst = max(worst, float(np.linalg.norm(theirs - mine)))
    tol = 1e-9
    return Check('FK vs KDL', worst < tol,
                 f'worst TCP disagreement {worst:.3e} m over {samples} random '
                 'configurations', worst, tol)


def compare_dynamics(cfg: ArmConfig, samples: int = 200, seed: int = 0,
                     gravity_only: bool = False) -> Check:
    """Compare joint torques against KDL's recursive Newton-Euler solver."""
    model = ArmModel(cfg)
    # Stop at the gripper body: `tcp_link` is a massless frame marker that the
    # analytical model does not carry, and its 1e-6 kg would otherwise show up
    # as a spurious 1e-5 N.m disagreement.
    chain = kdl_chain_from_urdf(
        build_urdf(cfg, fixed_to_world=False),
        tip=f'{cfg.end_effector.name}_base_link')
    grav = kdl.Vector(0.0, 0.0, -cfg.gravity)
    solver = kdl.ChainIdSolver_RNE(chain, grav)
    rng = np.random.default_rng(seed)
    n = model.n

    worst = 0.0
    scale = 0.0
    for _ in range(samples):
        q = rng.uniform(model.lower, model.upper)
        qd = np.zeros(n) if gravity_only else rng.uniform(-1.5, 1.5, n)
        qdd = np.zeros(n) if gravity_only else rng.uniform(-3.0, 3.0, n)

        torques = kdl.JntArray(n)
        wrenches = [kdl.Wrench() for _ in range(chain.getNrOfSegments())]
        solver.CartToJnt(_to_jnt(q), _to_jnt(qd), _to_jnt(qdd), wrenches, torques)
        theirs = np.array([torques[i] for i in range(n)])

        # KDL knows nothing of the gearbox rotors, so compare the link dynamics.
        mine = model.inverse_dynamics(q, qd, qdd, include_friction=False)
        mine = mine - model.reflected_inertia * qdd

        worst = max(worst, float(np.max(np.abs(theirs - mine))))
        scale = max(scale, float(np.max(np.abs(theirs))))

    tol = 1e-7 * max(scale, 1.0)
    label = 'gravity torque vs KDL' if gravity_only else 'full dynamics vs KDL'
    return Check(label, worst < tol,
                 f'worst joint torque disagreement {worst:.3e} N.m '
                 f'(largest torque seen {scale:.1f} N.m) over {samples} states',
                 worst, tol)


# ---------------------------------------------------------------- energy
def check_energy_balance(cfg: ArmConfig, samples: int = 60,
                         seed: int = 0) -> Check:
    """Joint power must equal the rate of change of total mechanical energy.

    This is the Lagrangian statement of the same mechanics the Newton-Euler
    recursion implements, so agreement is a check of the model as a whole rather
    than of any one term.
    """
    model = ArmModel(cfg)
    rng = np.random.default_rng(seed)
    n = model.n
    h = 1e-6

    def energy(q, qd):
        fs = model.frames(q)
        potential = float(np.sum(fs.link_mass * fs.link_com[:, 2]))
        potential += fs.ee_mass * fs.ee_com[2]
        potential *= cfg.gravity

        kinetic = 0.0
        omega = np.zeros(3)
        v_prev = np.zeros(3)
        p_prev = fs.base_distal
        for i in range(n):
            z = fs.joint_axis[i]
            p_i = fs.joint_origin[i]
            v_joint = v_prev + np.cross(omega, p_i - p_prev)
            omega = omega + z * qd[i]
            v_com = v_joint + np.cross(omega, fs.link_com[i] - p_i)
            kinetic += 0.5 * fs.link_mass[i] * float(v_com @ v_com)
            kinetic += 0.5 * float(omega @ (fs.link_inertia[i] @ omega))
            v_prev, p_prev = v_joint, p_i
        v_ee = v_prev + np.cross(omega, fs.ee_com - p_prev)
        kinetic += 0.5 * fs.ee_mass * float(v_ee @ v_ee)
        kinetic += 0.5 * float(omega @ (fs.ee_inertia @ omega))
        return kinetic + potential

    worst = 0.0
    scale = 0.0
    for _ in range(samples):
        q = rng.uniform(model.lower * 0.8, model.upper * 0.8)
        qd = rng.uniform(-1.0, 1.0, n)
        qdd = rng.uniform(-2.0, 2.0, n)

        tau = model.inverse_dynamics(q, qd, qdd, include_friction=False)
        tau = tau - model.reflected_inertia * qdd      # rotors are not in E
        power = float(tau @ qd)

        # dE/dt along the trajectory implied by (q, qd, qdd).
        forward = energy(q + qd * h + 0.5 * qdd * h * h, qd + qdd * h)
        backward = energy(q - qd * h + 0.5 * qdd * h * h, qd - qdd * h)
        rate = (forward - backward) / (2.0 * h)

        worst = max(worst, abs(power - rate))
        scale = max(scale, abs(power))

    tol = 1e-4 * max(scale, 1.0)
    return Check('energy balance', worst < tol,
                 f'worst mismatch between joint power and dE/dt {worst:.3e} W '
                 f'(largest power {scale:.1f} W) over {samples} states',
                 worst, tol)


# -------------------------------------------------------------- analytic
def check_point_mass_pendulum() -> Check:
    """A mass on a massless arm: tau = m g L cos(theta), exactly."""
    from .config import (Actuator, ArmConfig, EndEffector, JointSpec, Material,
                         TubeLink)

    material = Material('massless', 0.0, 70e9, 200e6, [0.5, 0.5, 0.5, 1.0])
    actuator = Actuator('ideal', 1.0, 1.0, 1.0, 1.0, 0.0, 100.0, 0.0, 1e12,
                        0.0, 0.0, 48.0)
    length = 0.75
    mass = 2.5
    link = TubeLink('rod', length, 0.02, 0.019, material, [1.0, 0.0, 0.0],
                    0.0, 0.0)
    joint = JointSpec('j1', 'revolute', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], actuator, -3.0, 3.0, 1.0, link)
    pedestal = TubeLink('base', 0.0, 0.02, 0.019, material, [0, 0, 1], 0.0, 0.0)
    ee = EndEffector('tip', mass, 0.0, 0.0, 0.01, 0.01, 0.01, material,
                     False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cfg = ArmConfig('<synthetic>', {}, 'pendulum', [0, 0, 0], [0, 0, 0],
                    pedestal, 9.81, 'j1', {'massless': material},
                    {'ideal': actuator}, [joint], ee, {}, {}, {}, {})
    model = ArmModel(cfg)

    worst = 0.0
    for theta in np.linspace(-1.2, 1.2, 25):
        # Rotating about +y swings the arm from +x toward -z.
        expected = -mass * cfg.gravity * length * math.cos(theta)
        got = float(model.gravity_torque([theta])[0])
        worst = max(worst, abs(got - expected))
    tol = 1e-9
    return Check('point-mass pendulum', worst < tol,
                 f'worst deviation from m*g*L*cos(theta) is {worst:.3e} N.m',
                 worst, tol)


def check_uniform_rod_inertia(cfg: ArmConfig) -> Check:
    """Tube inertia must reduce to the textbook rod and disc limits.

    Thick-walled cylinder about its own centre:
        I_axial      = m/2 * (r_outer^2 + r_inner^2)
        I_transverse = m/12 * (3*(r_outer^2 + r_inner^2) + h^2)
    """
    from .config import TubeLink
    material = list(cfg.materials.values())[0]
    worst = 0.0
    for outer, wall, length in ((0.05, 0.05, 0.60), (0.04, 0.004, 0.35),
                                (0.10, 0.002, 1.20)):
        link = TubeLink('t', length, outer, outer - wall, material,
                        [0, 0, 1], 0.0, 0.0)
        ixx, iyy, izz = link.inertia_about_com()
        m = link.tube_mass
        ro2, ri2 = outer ** 2, (outer - wall) ** 2
        worst = max(worst, abs(izz - 0.5 * m * (ro2 + ri2)))
        worst = max(worst, abs(ixx - m * (3 * (ro2 + ri2) + length ** 2) / 12.0))
        worst = max(worst, abs(iyy - ixx))
    tol = 1e-12
    return Check('tube inertia vs textbook', worst < tol,
                 f'worst deviation from the closed-form tensor {worst:.3e} '
                 'kg.m^2', worst, tol)


def check_jacobian_against_finite_differences(cfg: ArmConfig,
                                              samples: int = 50) -> Check:
    model = ArmModel(cfg)
    rng = np.random.default_rng(7)
    h = 1e-7
    worst = 0.0
    for _ in range(samples):
        q = rng.uniform(model.lower * 0.8, model.upper * 0.8)
        J = model.jacobian(q)
        for i in range(model.n):
            step = np.zeros(model.n)
            step[i] = h
            numeric = (model.fk(q + step) - model.fk(q - step)) / (2 * h)
            worst = max(worst, float(np.max(np.abs(numeric - J[:3, i]))))
    tol = 1e-6
    return Check('Jacobian vs finite differences', worst < tol,
                 f'worst column disagreement {worst:.3e} m/rad', worst, tol)


# ------------------------------------------------------------------ main
def run_all(cfg: ArmConfig, samples: int = 200) -> List[Check]:
    plain = _fingerless(cfg)
    checks = [
        check_uniform_rod_inertia(cfg),
        check_point_mass_pendulum(),
        check_jacobian_against_finite_differences(cfg),
        check_energy_balance(cfg),
    ]
    if HAVE_KDL:
        checks.insert(0, compare_forward_kinematics(plain, samples))
        checks.insert(1, compare_dynamics(plain, samples, gravity_only=True))
        checks.insert(2, compare_dynamics(plain, samples, gravity_only=False))
    else:
        checks.append(Check('KDL cross-check', False,
                            'PyKDL not importable; install python3-pykdl', 0, 0))
    return checks


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog='verify_physics',
        description='Cross-check the dynamics against KDL, energy conservation '
                    'and closed-form results.')
    p.add_argument('--config', default=None)
    p.add_argument('--samples', type=int, default=200)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    checks = run_all(cfg, args.samples)

    W = 96
    print('=' * W)
    print('  PHYSICS VERIFICATION')
    print(f'  model: {cfg.source_path}')
    print(f'  KDL available: {HAVE_KDL}')
    print('=' * W)
    for c in checks:
        print(f'  [{"PASS" if c.passed else "FAIL"}]  {c.name}')
        print(f'          {c.detail}')
        if c.tolerance:
            print(f'          tolerance {c.tolerance:.3e}')
    print('-' * W)
    failed = [c.name for c in checks if not c.passed]
    if failed:
        print('  FAILED: ' + ', '.join(failed))
    else:
        print(f'  All {len(checks)} checks passed.')
    print('=' * W)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
