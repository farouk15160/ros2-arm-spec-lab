#!/usr/bin/env python3
"""Minimum torque each joint must deliver, and the catalogue parts that cover it.

    python tools/joint_sizing.py
    python tools/joint_sizing.py --config src/arm_lab_model/config/rover_arm_1m_2kg.yaml
    python tools/joint_sizing.py --tilt 20 --json

The smallest part from each catalogue (Robstride, eRob, Harmonic Drive CSF/CSG
gear frame) that covers both requirements is listed per joint.

Load cases, all with the configured gripper and the spec payloads (2 kg
anywhere, 3 kg within 700 mm of the shoulder axis):

  hold        worst static gravity torque over random joint configurations
              plus the spec report's reach sweep, level ground
  hold tilt   the same with the base tilted --tilt degrees in the worst
              direction (a rover parked on a slope); this is what loads the
              base yaw joint
  dyn TCP     gravity + inertia at the spec TCP acceleration over the reach
              sweep, plus moving friction (the spec report's method)
  dyn joint   gravity + each joint alone at motion.joint_acceleration, at full
              reach, plus moving friction

  required continuous = max(hold, hold tilt)   -- no brakes: holding is continuous
  required peak       = max(dyn TCP, dyn joint, continuous / (1 - reserve))

The droop split uses the Harmonic Drive three-segment wind-up curve for the
gear (seen and corrected by an output encoder) and the configured bracket and
bearing stiffness (outboard of the encoder, so never corrected).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'arm_lab_model'))

from arm_lab_model.config import load_config  # noqa: E402
from arm_lab_model.kinematics import ArmModel  # noqa: E402
from arm_lab_model.reference import gearbox_table  # noqa: E402
from arm_lab_model.spec_report import SpecReport  # noqa: E402

# (name, continuous N.m, peak N.m, mass kg), smallest first.
CATALOGUES = {
    # Robstride QDD. RS04 continuous is its 220 x 200 mm heat-sink rating
    # (40 N.m needs a 345 x 345 mm plate).
    'Robstride': [
        ('RS05', 1.6, 5.5, 0.191),
        ('RS00', 5.0, 14.0, 0.310),
        ('RS02', 6.0, 17.0, 0.405),
        ('RS06', 11.0, 36.0, 0.621),
        ('RS04', 35.0, 120.0, 1.420),
    ],
    # ZeroErr eRob, ratio 100: the ratings of the Harmonic Drive CSG gear inside.
    'eRob': [
        ('eRob70I', 10.0, 36.0, 0.88),
        ('eRob80I', 31.0, 70.0, 1.09),
        ('eRob90I', 52.0, 107.0, 1.64),
        ('eRob110I', 87.0, 204.0, 2.68),
    ],
}
# Harmonic Drive catalogue, ratio 100: (rated at 2000 rpm, repeated peak).
CSF_100 = {14: (7.8, 28), 17: (24, 54), 20: (40, 82), 25: (67, 157), 32: (137, 333)}
CSG_100 = {14: (10, 36), 17: (31, 70), 20: (52, 107), 25: (87, 204), 32: (178, 433)}


def payload_for(reach: float, cfg) -> float:
    t = cfg.spec_targets
    full = float(t.get('payload_at_full_reach', 0.0))
    near = float(t.get('payload_at_700mm', full))
    return max(full, near) if reach <= 0.700 + 1e-9 else full


def reach_sweep(model: ArmModel):
    """The spec report's vertical-plane IK sweep, as joint configurations."""
    r_max = model.geometric_max_reach
    origin = model.frames(np.zeros(model.n)).reach_origin
    poses = []
    for r in np.linspace(0.25 * r_max, r_max, 12):
        for h in np.linspace(-0.45 * r_max, 0.45 * r_max, 9):
            if math.hypot(r, h) > r_max:
                continue
            target = origin + np.array([r, 0.0, h])
            q = model.ik_position(target)
            if np.linalg.norm(model.frames(q, light=True).tcp - target) <= 0.02:
                poses.append(q)
    return poses


def static_worst(model, cfg, poses, g_vectors):
    worst = np.zeros(model.n)
    for g in g_vectors:
        model.g = g
        for q in poses:
            fs = model.frames(q)
            m = payload_for(model.reach(q, fs), cfg)
            worst = np.maximum(worst, np.abs(model.gravity_torque(q, payload=m, fs=fs)))
    model.g = np.array([0.0, 0.0, -cfg.gravity])
    return worst


def tilted_gravity(cfg, tilt_deg: float, steps: int = 12):
    g0 = np.array([0.0, 0.0, -cfg.gravity])
    t = math.radians(tilt_deg)
    out = []
    for k in range(steps):
        a = 2.0 * math.pi * k / steps
        axis = np.array([math.cos(a), math.sin(a), 0.0])
        # Rodrigues rotation of g about a horizontal axis.
        out.append(g0 * math.cos(t) + np.cross(axis, g0) * math.sin(t)
                   + axis * (axis @ g0) * (1.0 - math.cos(t)))
    return out


def droop_split(model, cfg):
    q = model.resolve_pose('full_reach')
    fs = model.frames(q)
    payload = float(cfg.spec_targets.get('payload_at_full_reach', 0.0))
    tau = model.gravity_torque(q, payload=payload, fs=fs)
    rows, gear_total, struct_total = [], 0.0, 0.0
    for i, joint in enumerate(cfg.joints):
        act = joint.actuator
        r = fs.tcp - fs.joint_origin[i]
        lever = float(np.linalg.norm(np.cross(fs.joint_axis[i], r)))
        entry = (gearbox_table(act.gear_ratio).get(act.gearbox_size)
                 if act.gearbox_series else None)
        struct = sum(abs(tau[i]) / k for k in (act.bracket_stiffness,
                                               act.bearing_stiffness) if k > 0)
        if entry:
            gear = entry.windup(tau[i])
        else:
            # No catalogue gear: joint_stiffness is the whole joint, so the
            # gear is what is left after the bracket and bearing.
            gear = max(abs(tau[i]) / act.joint_stiffness - struct, 0.0)
        gear_total += gear * lever
        struct_total += struct * lever
        rows.append({'joint': joint.name, 'torque_nm': float(tau[i]),
                     'lever_m': lever, 'gear_mm': gear * lever * 1e3,
                     'structure_mm': struct * lever * 1e3})
    bending = model.deflection(q, payload=payload, fs=fs)
    tube = (bending['bending'] + bending['torsion'])
    return {'rows': rows, 'gear_mm': gear_total * 1e3,
            'structure_mm': struct_total * 1e3, 'tube_mm': tube * 1e3,
            'raw_mm': (gear_total + struct_total + tube) * 1e3,
            'with_output_encoder_mm': (struct_total + tube) * 1e3}


def smallest(table, cont, peak):
    for size in sorted(table):
        rated, rep = table[size]
        if rated >= cont and rep >= peak:
            return size
    return None


def run(cfg, tilt: float, samples: int, seed: int):
    model = ArmModel(cfg)
    rng = np.random.default_rng(seed)
    poses = reach_sweep(model)
    poses += [rng.uniform(model.lower, model.upper) for _ in range(samples)]
    g_level = [np.array([0.0, 0.0, -cfg.gravity])]

    hold = static_worst(model, cfg, poses, g_level)
    hold_tilt = static_worst(model, cfg, poses, tilted_gravity(cfg, tilt))

    rep = SpecReport(cfg)
    friction = model.friction
    dyn_tcp = rep._worst_case_torque(float(cfg.spec_targets.get(
        'payload_at_full_reach', 0.0)))['peak'] + friction

    a_joint = float((cfg.raw.get('motion', {}) or {}).get('joint_acceleration', 3.0))
    q_full = model.resolve_pose('full_reach')
    fs_full = model.frames(q_full)
    m_full = float(cfg.spec_targets.get('payload_at_full_reach', 0.0))
    dyn_joint = np.zeros(model.n)
    for i in range(model.n):
        qdd = np.zeros(model.n)
        for sign in (1.0, -1.0):
            qdd[i] = sign * a_joint
            tau = model.inverse_dynamics(q_full, qdd=qdd, payload=m_full,
                                         fs=fs_full, include_friction=False)
            dyn_joint[i] = max(dyn_joint[i], abs(tau[i]) + friction[i])

    reserve = float(cfg.control.get('dynamic_torque_reserve', 0.0))
    need_cont = np.maximum(hold, hold_tilt)
    need_peak = np.maximum.reduce([dyn_tcp, dyn_joint, need_cont / (1.0 - reserve)])

    joints = []
    for i, joint in enumerate(cfg.joints):
        act = joint.actuator
        picks = {family: next((name for name, cont, peak, _ in parts
                               if cont >= need_cont[i] and peak >= need_peak[i]), None)
                 for family, parts in CATALOGUES.items()}
        joints.append({
            'joint': joint.name,
            'hold_nm': float(hold[i]), 'hold_tilt_nm': float(hold_tilt[i]),
            'dyn_tcp_nm': float(dyn_tcp[i]), 'dyn_joint_nm': float(dyn_joint[i]),
            'required_continuous_nm': float(need_cont[i]),
            'required_peak_nm': float(need_peak[i]),
            'fitted': act.name,
            'fitted_rated_nm': act.output_continuous_torque,
            'fitted_peak_nm': act.output_peak_torque,
            'smallest': picks,
            'smallest_csf100': smallest(CSF_100, need_cont[i], need_peak[i]),
            'smallest_csg100': smallest(CSG_100, need_cont[i], need_peak[i]),
        })
    return {'config': cfg.source_path, 'tilt_deg': tilt, 'reserve': reserve,
            'joint_acceleration': a_joint, 'samples': len(poses),
            'joints': joints, 'droop': droop_split(model, cfg)}


def render(res) -> str:
    out = [f"JOINT SIZING  --  {res['config']}",
           f"  2 kg anywhere, 3 kg within 700 mm; tilt {res['tilt_deg']:.0f} deg; "
           f"{res['reserve'] * 100:.0f} % peak reserve; "
           f"{res['joint_acceleration']:.1f} rad/s^2 joint accel; "
           f"{res['samples']} configurations", '',
           f"{'joint':<9}{'hold':>7}{'tilt':>7}{'dynTCP':>8}{'dynJnt':>8}"
           f"{'REQ cont':>10}{'REQ peak':>10}   {'fitted (cont/peak)':<27}"
           + ''.join(f'{family:<11}' for family in CATALOGUES) + 'CSF/CSG-100']
    for j in res['joints']:
        ok = (j['fitted_rated_nm'] >= j['required_continuous_nm']
              and j['fitted_peak_nm'] >= j['required_peak_nm'])
        fitted = (f"{j['fitted']} {j['fitted_rated_nm']:.0f}/{j['fitted_peak_nm']:.0f}"
                  + ('' if ok else ' <-- SHORT'))
        out.append(
            f"{j['joint']:<9}{j['hold_nm']:>7.1f}{j['hold_tilt_nm']:>7.1f}"
            f"{j['dyn_tcp_nm']:>8.1f}{j['dyn_joint_nm']:>8.1f}"
            f"{j['required_continuous_nm']:>10.1f}{j['required_peak_nm']:>10.1f}   "
            f"{fitted:<27}"
            + ''.join(f"{str(j['smallest'][family]):<11}" for family in CATALOGUES)
            + f"{j['smallest_csf100']}/{j['smallest_csg100']}")
    d = res['droop']
    out += ['', 'DROOP AT FULL REACH, 2 kg  (gear curve from the catalogue, K1/K2/K3)',
            f"{'joint':<9}{'torque':>9}{'lever m':>9}{'gear mm':>9}{'struct mm':>11}"]
    for r in d['rows']:
        out.append(f"{r['joint']:<9}{r['torque_nm']:>9.1f}{r['lever_m']:>9.3f}"
                   f"{r['gear_mm']:>9.2f}{r['structure_mm']:>11.2f}")
    out += [f"  gear {d['gear_mm']:.1f} mm + bracket/bearing {d['structure_mm']:.1f} mm"
            f" + tube {d['tube_mm']:.2f} mm = raw {d['raw_mm']:.1f} mm",
            f"  with output encoders (gear wind-up measured and removed): "
            f"{d['with_output_encoder_mm']:.1f} mm"]
    return '\n'.join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config', default=None)
    p.add_argument('--tilt', type=float, default=15.0, help='base tilt, degrees')
    p.add_argument('--samples', type=int, default=4000, help='random configurations')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--json', action='store_true')
    args = p.parse_args(argv)
    res = run(load_config(args.config), args.tilt, args.samples, args.seed)
    print(json.dumps(res, indent=2) if args.json else render(res))
    return 0


if __name__ == '__main__':
    sys.exit(main())
