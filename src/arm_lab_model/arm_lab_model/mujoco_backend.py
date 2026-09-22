"""Optional, ROS-free MuJoCo export, inverse-dynamics checks and simulation.

By default the arm exporter uses a rigid tool (jaws included in its total
mass), which is what the contact-free dynamics benchmark compares against.
`build_mjcf(..., fingers=True, scene=load_world(sdf))` adds force-driven jaws,
the ground and the world's sample boxes for the interactive ROS simulator.
Arbitrary MJCF trees can be stepped by the generic smoke-test runner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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


@dataclass
class SceneBox:
    """A free box from the world file: a pick target for the simulator."""

    name: str
    pos: tuple
    rpy: tuple
    size: tuple
    mass: float
    friction: float
    rgba: tuple = (0.9, 0.45, 0.1, 1.0)


@dataclass
class Scene:
    ground_friction: float = 1.0
    boxes: tuple = ()


def _floats(text, count, default):
    values = [float(v) for v in (text or '').split()]
    return tuple(values) if len(values) == count else default


def load_world(path):
    """Read the ground and the free boxes out of a Gazebo world SDF.

    Only what the arm can touch is taken: a plane becomes the ground, and each
    non-static model whose collision is a box becomes a free body with its
    mass and friction. Lights, plugins and other shapes are ignored, so the
    Gazebo and MuJoCo benches share one world file.
    """
    world = ET.parse(str(path)).getroot().find('world')
    if world is None:
        raise ValueError(f'{path}: no <world> element')
    scene = Scene()
    boxes = []
    for model in world.findall('model'):
        link = model.find('link')
        collision = link.find('collision') if link is not None else None
        geometry = collision.find('geometry') if collision is not None else None
        if geometry is None:
            continue
        mu = collision.findtext('surface/friction/ode/mu')
        friction = float(mu) if mu else 1.0
        if geometry.find('plane') is not None:
            scene.ground_friction = friction
            continue
        size = geometry.findtext('box/size')
        if size is None or (model.findtext('static') or '').strip() == 'true':
            continue
        pose = _floats(model.findtext('pose'), 6, (0.0,) * 6)
        colour = _floats(link.findtext('visual/material/diffuse'), 4, SceneBox.rgba)
        boxes.append(SceneBox(model.get('name'), pose[:3], pose[3:],
                              _floats(size, 3, (0.05,) * 3),
                              float(link.findtext('inertial/mass') or 1.0),
                              friction, colour))
    scene.boxes = tuple(boxes)
    return scene


def build_mjcf(cfg, payload=0.0, timestep=0.001, *, fingers=False, scene=None):
    """Direct MJCF export with output-side motors and reflected rotor inertia.

    `fingers` models the jaws as two slide joints driven like the Gazebo
    gripper controller (a squeeze force in force mode, an opening in position
    mode), with the tool mass split exactly as the URDF does. `scene` adds the
    ground and free boxes; arm and jaw geoms then collide with the scene but
    never with each other. Both default off, which keeps the benchmark model.
    """
    _number(payload, 'payload')
    _number(timestep, 'timestep', positive=True)
    root = ET.Element('mujoco', model=cfg.name)
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false')
    option = ET.SubElement(root, 'option', timestep=str(timestep),
                           gravity=f'0 0 {-cfg.gravity}', integrator='implicitfast')
    if scene is not None:
        # Elliptic cones and a high impedance ratio keep a friction grasp from
        # creeping out of the jaws.
        option.set('cone', 'elliptic')
        option.set('impratio', '10')
    world = ET.SubElement(root, 'worldbody')
    arm_contact = {'contype': '1', 'conaffinity': '0'} if scene is not None else {}
    if scene is not None:
        _scene(root, world, scene)
    parent = ET.SubElement(world, 'body', name='base_link', pos=_fmt(cfg.mount_xyz),
                           **_orientation(rpy_to_matrix(cfg.mount_rpy)))

    def tube(body, link):
        _inertial(body, link.mass, link.com_xyz, link.inertia_in_link_frame())
        ET.SubElement(body, 'geom', type='cylinder',
                      size=_fmt([link.outer_radius, link.length / 2]),
                      pos=_fmt(np.array(link.direction) * link.length / 2),
                      zaxis=_fmt(link.direction), rgba=_fmt(link.material.color),
                      **arm_contact)

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
    jaws = fingers and ee.simulate_fingers
    body_mass, com = ee.mass, ee.com_offset
    if jaws:
        # Same split as the URDF: the jaws sit out at the jaw position, so the
        # body moves inboard to keep the whole tool's CoM where it is configured.
        body_mass = max(ee.mass - 2.0 * ee.finger_mass, 1e-3)
        com = (ee.mass * ee.com_offset - 2.0 * ee.finger_mass
               * (ee.body_length + ee.finger_length / 2.0)) / body_mass
    x, y, z = ee.body_width, ee.body_height, ee.body_length
    I = R @ np.diag([body_mass * (y*y + z*z) / 12,
                     body_mass * (x*x + z*z) / 12,
                     body_mass * (x*x + y*y) / 12]) @ R.T
    if body_mass > 0:
        _inertial(tool, body_mass, direction * com, I)
    ET.SubElement(tool, 'geom', type='box', size=_fmt([x/2, y/2, z/2]),
                  pos=_fmt(direction * z/2), **_orientation(R), **arm_contact)
    ET.SubElement(tool, 'site', name='tcp', pos=_fmt(direction * ee.tcp_offset), size='0.006')
    if jaws:
        _jaws(tool, motors, ee, direction, R, arm_contact)
    if payload:
        # The analytical payload is a point mass. Tiny positive inertia is
        # required by the compiler; its approximation error is tested explicitly.
        body = ET.SubElement(tool, 'body', name='test_payload', pos=_fmt(direction * ee.tcp_offset))
        _inertial(body, payload, [0, 0, 0], np.eye(3) * 1e-12)
    ET.indent(root)
    return ET.tostring(root, encoding='unicode') + '\n'


def _jaws(tool, motors, ee, flange, R, contact):
    """Two slide joints from shut (0) to open (stroke / 2), as in the URDF."""
    open_axis = R[:, 0]
    half = ee.stroke / 2.0
    t, w, length = ee.finger_thickness, ee.finger_width, ee.finger_length
    I = R @ np.diag([ee.finger_mass * (w*w + length*length) / 12,
                     ee.finger_mass * (t*t + length*length) / 12,
                     ee.finger_mass * (t*t + w*w) / 12]) @ R.T
    for sign, side in ((1.0, 'left'), (-1.0, 'right')):
        name = f'{ee.name}_{side}_joint'
        axis = open_axis * sign
        body = ET.SubElement(tool, 'body', name=f'{ee.name}_{side}_finger',
                             pos=_fmt(flange * ee.body_length))
        centre = axis * t / 2.0 + flange * length / 2.0
        _inertial(body, ee.finger_mass, centre, I)
        ET.SubElement(body, 'joint', name=name, type='slide', axis=_fmt(axis),
                      limited='true', range=_fmt([0.0, half]),
                      damping='5', frictionloss='1',
                      # A 60 g jaw alone makes the soft limit spongy: under the
                      # full squeeze it overran by 22 mm. The drive's reflected
                      # inertia and a stiffer limit keep it within a millimetre.
                      armature='0.05', solreflimit='0.004 1')
        ET.SubElement(body, 'geom', type='box', size=_fmt([t/2, w/2, length/2]),
                      pos=_fmt(centre), **_orientation(R), condim='4',
                      friction='1.5 0.02 0.0001', rgba='0.25 0.25 0.28 1', **contact)
        if ee.grasp_mode == 'force':
            ET.SubElement(motors, 'motor', name=name + '_motor', joint=name, gear='1',
                          ctrllimited='true',
                          ctrlrange=_fmt([-ee.grip_force_max, ee.grip_force_max]))
        else:
            ET.SubElement(motors, 'position', name=name + '_motor', joint=name,
                          kp='2000', ctrllimited='true', ctrlrange=_fmt([0.0, half]),
                          forcelimited='true',
                          forcerange=_fmt([-ee.grip_force_max, ee.grip_force_max]))


def _scene(root, world, scene):
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'headlight', ambient='0.4 0.4 0.4', diffuse='0.6 0.6 0.6')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', name='ground', type='2d', builtin='checker',
                  rgb1='0.52 0.47 0.40', rgb2='0.45 0.41 0.35', width='512', height='512')
    ET.SubElement(asset, 'material', name='ground', texture='ground', texrepeat='20 20')
    ET.SubElement(world, 'light', pos='0 0 4', dir='0 0 -1', diffuse='0.6 0.6 0.6')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='30 30 0.1',
                  material='ground', friction=_fmt([scene.ground_friction, 0.005, 0.0001]),
                  contype='0', conaffinity='1')
    for box in scene.boxes:
        body = ET.SubElement(world, 'body', name=box.name, pos=_fmt(box.pos),
                             **_orientation(rpy_to_matrix(box.rpy)))
        ET.SubElement(body, 'freejoint', name=box.name + '_free')
        x, y, z = box.size
        ET.SubElement(body, 'inertial', pos='0 0 0', mass=str(box.mass),
                      diaginertia=_fmt([box.mass * (y*y + z*z) / 12,
                                        box.mass * (x*x + z*z) / 12,
                                        box.mass * (x*x + y*y) / 12]))
        ET.SubElement(body, 'geom', type='box', size=_fmt([x/2, y/2, z/2]), condim='4',
                      friction=_fmt([box.friction, 0.02, 0.0001]), rgba=_fmt(box.rgba),
                      contype='1', conaffinity='1')


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
