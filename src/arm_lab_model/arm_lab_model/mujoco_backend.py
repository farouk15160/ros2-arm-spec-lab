"""Optional, ROS-free MuJoCo export, inverse-dynamics checks and simulation.

The arm exporter uses a rigid tool (jaws included in its total mass). Contact
and grasp validation are separate from the contact-free arm dynamics benchmark.
Arbitrary MJCF trees can be stepped by the generic smoke-test runner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .config import load_config, _number
from .kinematics import ArmModel, _frame_from_direction, rpy_to_matrix


def _engine():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError('MuJoCo is optional. Install with: pip install "mujoco>=3.2,<4"') from exc
    return mujoco


def _fmt(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def _orientation(R):
    return {'xyaxes': _fmt(np.concatenate((R[:, 0], R[:, 1])))}


def _inertial(body, mass, com, I):
    ET.SubElement(body, 'inertial', mass=str(mass), pos=_fmt(com),
                  fullinertia=_fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))


def build_mjcf(cfg, payload=0.0, timestep=0.001):
    """Direct MJCF export with output-side motors and reflected rotor inertia."""
    _number(payload, 'payload')
    _number(timestep, 'timestep', positive=True)
    root = ET.Element('mujoco', model=cfg.name)
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false')
    ET.SubElement(root, 'option', timestep=str(timestep), gravity=f'0 0 {-cfg.gravity}',
                  integrator='implicitfast')
    world = ET.SubElement(root, 'worldbody')
    parent = ET.SubElement(world, 'body', name='base_link', pos=_fmt(cfg.mount_xyz),
                           **_orientation(rpy_to_matrix(cfg.mount_rpy)))

    def tube(body, link):
        _inertial(body, link.mass, link.com_xyz, link.inertia_in_link_frame())
        ET.SubElement(body, 'geom', type='cylinder',
                      size=_fmt([link.outer_radius, link.length / 2]),
                      pos=_fmt(np.array(link.direction) * link.length / 2),
                      zaxis=_fmt(link.direction), rgba=_fmt(link.material.color))

    tube(parent, cfg.pedestal)
    previous = cfg.pedestal
    motors = ET.SubElement(root, 'actuator')
    for joint in cfg.joints:
        offset = np.array(previous.direction) * previous.length + joint.origin_xyz
        parent = ET.SubElement(parent, 'body', name=joint.link.name, pos=_fmt(offset),
                               **_orientation(rpy_to_matrix(joint.origin_rpy)))
        ET.SubElement(parent, 'joint', name=joint.name, type='hinge', axis=_fmt(joint.axis),
                      limited='false' if joint.jtype == 'continuous' else 'true',
                      range=_fmt([joint.lower, joint.upper]),
                      armature=str(joint.actuator.reflected_inertia),
                      damping=str(joint.actuator.viscous_damping),
                      frictionloss=str(joint.actuator.friction))
        tube(parent, joint.link)
        ET.SubElement(motors, 'motor', name=joint.name + '_motor', joint=joint.name,
                      gear='1', ctrllimited='true',
                      ctrlrange=_fmt([-joint.effort_limit, joint.effort_limit]))
        previous = joint.link
    ee = cfg.end_effector
    direction = np.array(previous.direction)
    tool = ET.SubElement(parent, 'body', name=ee.name + '_rigid_tool',
                         pos=_fmt(direction * previous.length))
    R = _frame_from_direction(direction)
    x, y, z = ee.body_width, ee.body_height, ee.body_length
    I = R @ np.diag([ee.mass * (y*y + z*z) / 12,
                     ee.mass * (x*x + z*z) / 12,
                     ee.mass * (x*x + y*y) / 12]) @ R.T
    if ee.mass > 0:
        _inertial(tool, ee.mass, direction * ee.com_offset, I)
    ET.SubElement(tool, 'geom', type='box', size=_fmt([x/2, y/2, z/2]),
                  pos=_fmt(direction * z/2), **_orientation(R))
    ET.SubElement(tool, 'site', name='tcp', pos=_fmt(direction * ee.tcp_offset), size='0.006')
    if payload:
        # The analytical payload is a point mass. Tiny positive inertia is
        # required by the compiler; its approximation error is tested explicitly.
        body = ET.SubElement(tool, 'body', name='test_payload', pos=_fmt(direction * ee.tcp_offset))
        _inertial(body, payload, [0, 0, 0], np.eye(3) * 1e-12)
    ET.indent(root)
    return ET.tostring(root, encoding='unicode') + '\n'


def _indices(mj, cfg):
    return (np.array([mj.joint(n).qposadr[0] for n in cfg.joint_names]),
            np.array([mj.joint(n).dofadr[0] for n in cfg.joint_names]))


def _metadata(xml):
    return {'mujoco_version': _engine().__version__,
            'model_sha256': hashlib.sha256(xml.encode()).hexdigest()}


def _actuator_digest(cfg):
    data = json.dumps([asdict(j.actuator) for j in cfg.joints], sort_keys=True, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def crosscheck(cfg, samples=150, seed=0, payload=0.0, atol=1e-7, rtol=1e-8):
    """Compare each joint at seeded states, retaining rotor inertia in both engines."""
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError('samples must be a positive integer')
    _number(atol, 'atol', positive=True)
    _number(rtol, 'rtol')
    mujoco = _engine()
    xml = build_mjcf(cfg, payload)
    mj = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(mj)
    # Isolate rigid-body equations; constraints and friction are separate tests.
    mj.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    mj.dof_damping[:] = 0
    model = ArmModel(cfg)
    qp, dof = _indices(mj, cfg)
    rng = np.random.default_rng(seed)
    errors = {'gravity': 0.0, 'full_dynamics': 0.0}
    tcp_error = 0.0
    passed = True
    for _ in range(samples):
        q = rng.uniform(model.lower, model.upper)
        v = rng.uniform(-model.velocity_limits, model.velocity_limits)
        a = rng.uniform(-3, 3, model.n)
        for label, velocity, acceleration in (
                ('gravity', np.zeros(model.n), np.zeros(model.n)), ('full_dynamics', v, a)):
            data.qpos[qp] = q
            data.qvel[dof] = velocity
            mujoco.mj_forward(mj, data)
            data.qacc[dof] = acceleration
            mujoco.mj_inverse(mj, data)
            ours = model.inverse_dynamics(q, velocity, acceleration, payload=payload, include_friction=False)
            theirs = data.qfrc_inverse[dof]
            error = np.abs(ours - theirs)
            errors[label] = max(errors[label], float(np.max(error)))
            passed &= bool(np.all(np.isfinite(error)) and np.all(error <= atol + rtol * np.abs(theirs)))
        tcp_error = max(tcp_error, float(np.max(np.abs(data.site('tcp').xpos - model.fk(q)))))
    return {**_metadata(xml), 'test': 'rigid_body_crosscheck', 'samples': samples, 'seed': seed,
            'payload_kg': payload, 'atol_nm': atol, 'rtol': rtol,
            'max_gravity_error_nm': errors['gravity'],
            'max_full_dynamics_error_nm': errors['full_dynamics'],
            'max_tcp_error_m': tcp_error, 'passed': bool(passed and tcp_error < 1e-9),
            'scope': 'Fixed base, rigid tool, no contacts or joint friction; rotor inertia included.'}


def _steps(duration, timestep):
    _number(duration, 'duration', positive=True)
    _number(timestep, 'timestep', positive=True)
    return math.ceil(duration / timestep)


def _warnings(data):
    return [int(w.number) for w in data.warning]


def simulate_arm(cfg, duration=2.0, timestep=0.001, pose='home', target=None,
                 payload=0.0, max_error=0.05):
    """Quintic point-to-point/hold test with feedforward + PD and torque saturation.

    Contacts are disabled for this actuator/motion benchmark. Electrical speed
    envelopes are applied when data exist. Thermal behaviour is reported by the
    engineering tools, not integrated into this short-duration test.
    """
    steps = _steps(duration, timestep)
    _number(max_error, 'max_error', positive=True)
    mujoco = _engine()
    xml = build_mjcf(cfg, payload, timestep)
    mj = mujoco.MjModel.from_xml_string(xml)
    mj.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    data = mujoco.MjData(mj)
    model = ArmModel(cfg)
    qp, dof = _indices(mj, cfg)
    start = model.resolve_pose(pose, strict=True)
    end = start if target is None else model.resolve_pose(target, strict=True)
    for q in (start, end):
        if not np.all(np.isfinite(q)) or np.any(q < model.lower) or np.any(q > model.upper):
            raise ValueError('test poses must be finite and inside joint limits')
    data.qpos[qp] = start
    mujoco.mj_forward(mj, data)
    from .actuator_model import ActuatorElectrical
    electrical = []
    for j in cfg.joints:
        a = j.actuator
        electrical.append(ActuatorElectrical(a.torque_constant, a.phase_resistance,
                          a.bus_voltage, a.max_phase_current, a.gear_ratio, a.efficiency,
                          peak_output_torque=a.output_peak_torque)
                          if min(a.torque_constant, a.phase_resistance, a.max_phase_current) > 0 else None)
    peak = np.zeros(model.n)
    squared = np.zeros(model.n)
    worst_error = worst_speed_ratio = 0.0
    saturated = 0
    completed = 0
    finite = True
    for step in range(steps):
        s = min(data.time / duration, 1.0)
        blend = 10*s**3 - 15*s**4 + 6*s**5
        desired = start + (end-start)*blend
        velocity = (end-start)*(30*s**2 - 60*s**3 + 30*s**4)/duration
        acceleration = (end-start)*(60*s - 180*s**2 + 120*s**3)/duration**2
        q, v = data.qpos[qp].copy(), data.qvel[dof].copy()
        error = desired-q
        requested = model.inverse_dynamics(q, velocity, acceleration, payload=payload) + 80*error + 15*(velocity-v)
        limits = np.array([min(j.effort_limit, e.available_torque(v[i]) if e else j.effort_limit)
                           for i, (j, e) in enumerate(zip(cfg.joints, electrical))])
        saturated += int(np.any(np.abs(requested) > limits + 1e-9))
        applied = np.clip(requested, -limits, limits)
        data.ctrl[:] = applied
        mujoco.mj_step(mj, data)
        completed += 1
        finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        if not finite or any(_warnings(data)):
            break
        next_s = min(data.time / duration, 1.0)
        next_desired = start + (end-start)*(10*next_s**3 - 15*next_s**4 + 6*next_s**5)
        worst_error = max(worst_error, float(np.max(np.abs(next_desired - data.qpos[qp]))))
        worst_speed_ratio = max(worst_speed_ratio, float(np.max(np.abs(data.qvel[dof]) / model.velocity_limits)))
        peak = np.maximum(peak, np.abs(applied))
        squared += applied**2
    rms = np.sqrt(squared / max(completed, 1))
    warnings = _warnings(data)
    return {**_metadata(xml), 'test': 'arm_motion', 'duration_s': float(data.time),
            'requested_duration_s': duration, 'initial_q_rad': start.tolist(),
            'target_q_rad': end.tolist(),
            'final_q_rad': data.qpos[qp].tolist() if finite else None,
            'controller': {'kp_nm_per_rad': 80, 'kd_nm_s_per_rad': 15,
                           'trajectory': 'quintic'},
            'actuator_parameters_sha256': _actuator_digest(cfg),
            'timestep_s': timestep, 'steps': completed, 'joint_names': cfg.joint_names,
            'payload_kg': payload, 'max_tracking_error_rad': worst_error,
            'tracking_tolerance_rad': max_error, 'max_speed_limit_ratio': worst_speed_ratio,
            'peak_torque_nm': peak.tolist(), 'rms_torque_nm': rms.tolist(),
            'continuous_torque_exceeded': (rms > model.continuous_limits).tolist(),
            'saturation_fraction': saturated / max(completed, 1), 'warnings': warnings,
            'passed': bool(finite and not any(warnings) and completed == steps and saturated == 0
                           and worst_error <= max_error and worst_speed_ratio <= 1.001),
            'scope': 'Contact-free fixed-base arm; rigid tool; feedforward + PD; peak/current/voltage torque limits. No thermal integration or ROS controller.'}


def simulate_mjcf(path, duration=2.0, timestep=0.001, keyframe=None):
    """Step any MJCF tree. This checks numerical health, not locomotion success."""
    steps = _steps(duration, timestep)
    mujoco = _engine()
    mj = mujoco.MjModel.from_xml_path(str(path))
    mj.opt.timestep = timestep
    data = mujoco.MjData(mj)
    if keyframe is not None:
        key = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key < 0:
            raise ValueError(f'unknown keyframe: {keyframe}')
        mujoco.mj_resetDataKeyframe(mj, data, key)
    max_contacts = 0
    finite = True
    completed = 0
    for _ in range(steps):
        mujoco.mj_step(mj, data)
        completed += 1
        max_contacts = max(max_contacts, data.ncon)
        finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        if not finite or any(_warnings(data)):
            break
    return {**_metadata(Path(path).read_text()), 'test': 'mjcf_smoke',
            'source': str(Path(path).resolve()), 'nq': mj.nq, 'nv': mj.nv, 'nu': mj.nu,
            'steps': completed, 'duration_s': float(data.time), 'timestep_s': timestep,
            'requested_duration_s': duration, 'keyframe': keyframe,
            'max_contacts': max_contacts, 'warnings': _warnings(data),
            'final_qpos': data.qpos.tolist() if finite else None,
            'passed': bool(finite and not any(_warnings(data)) and completed == steps),
            'scope': 'Numerical smoke test only. Zero controls or constant keyframe controls; no gait/balance controller. Model hash covers root XML only.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('export', 'check', 'simulate', 'smoke'):
        p = commands.add_parser(name)
        p.add_argument('--output', '-o', help='MJCF for export; JSON report otherwise')
        if name != 'smoke':
            p.add_argument('--config')
            p.add_argument('--payload', type=float, default=0.0)
        if name == 'check':
            p.add_argument('--samples', type=int, default=150)
            p.add_argument('--seed', type=int, default=0)
        else:
            p.add_argument('--timestep', type=float, default=0.001)
        if name in ('simulate', 'smoke'):
            p.add_argument('--duration', type=float, default=2.0)
        if name == 'simulate':
            p.add_argument('--pose', default='home')
            p.add_argument('--target', help='named target pose; omit for a holding test')
            p.add_argument('--max-error', type=float, default=0.05)
        if name == 'smoke':
            p.add_argument('model', help='MJCF model, including branched/floating-base robots')
            p.add_argument('--keyframe')
    args = parser.parse_args(argv)
    try:
        if args.command == 'smoke':
            result = simulate_mjcf(args.model, args.duration, args.timestep, args.keyframe)
        else:
            cfg = load_config(args.config)
            if args.command == 'export':
                result = build_mjcf(cfg, args.payload, args.timestep)
            elif args.command == 'check':
                result = crosscheck(cfg, args.samples, args.seed, args.payload)
            else:
                result = simulate_arm(cfg, args.duration, args.timestep, args.pose,
                                      args.target, args.payload, args.max_error)
        output = result if isinstance(result, str) else json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output, end='')
        return 0 if isinstance(result, str) or result['passed'] else 1
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f'robot_test: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
