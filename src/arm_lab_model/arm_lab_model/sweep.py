"""Sweep one config parameter and watch the headline specs move.

    ros2 run arm_lab_model sweep --param joints.1.link.length --values 0.35:0.55:5
    ros2 run arm_lab_model sweep --param end_effector.mass --values 0.5,1.0,1.5,2.0
    ros2 run arm_lab_model sweep --param environment.gravity --values 1.62,3.72,9.81

Paths are dotted, with integers indexing into lists, exactly matching the
structure of arm_config.yaml.
"""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
from typing import Any, List, Optional

import numpy as np
import yaml

from .config import load_config, default_config_path
from .kinematics import ArmModel


def set_path(data: Any, path: str, value: Any) -> None:
    """Assign into a nested dict/list using a dotted path."""
    keys = path.split('.')
    node = data
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        if last not in node:
            raise KeyError(f'{path!r}: no such key {last!r} (have {sorted(node)})')
        node[last] = value


def get_path(data: Any, path: str) -> Any:
    node = data
    for key in path.split('.'):
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


def parse_values(text: str) -> List[float]:
    """`a,b,c` for an explicit list, or `start:stop:count` for a linear range."""
    if ':' in text:
        start, stop, count = text.split(':')
        return list(np.linspace(float(start), float(stop), int(count)))
    return [float(v) for v in text.split(',')]


def evaluate(raw: dict, tmpdir: str, index: int) -> dict:
    path = os.path.join(tmpdir, f'variant_{index}.yaml')
    with open(path, 'w') as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    cfg = load_config(path)
    m = ArmModel(cfg)

    q_full = m.resolve_pose(cfg.test_poses.get('full_reach', 'auto_full_reach'))
    fs = m.frames(q_full)
    cap, lim = m.payload_capacity(q_full, fs)
    want = float(cfg.spec_targets.get('payload_at_full_reach', 2.0))
    defl = m.deflection(q_full, payload=want, fs=fs)
    tau = m.gravity_torque(q_full, payload=want, fs=fs)
    util = float(np.max(np.abs(tau) / m.torque_limits))
    speed, _ = m.max_tcp_speed(q_full, fs)

    q_700 = m.resolve_pose(cfg.test_poses.get('reach_700', 'auto_reach_0.700'))
    cap700, _ = m.payload_capacity(q_700)

    return {
        'reach_mm': m.reach(q_full, fs) * 1000.0,
        'mass_kg': cfg.arm_mass,
        'payload_full_kg': cap,
        'payload_700_kg': cap700,
        'droop_mm': defl['total'] * 1000.0,
        'peak_util_pct': util * 100.0,
        'tcp_speed_ms': speed,
        'limiting': cfg.joint_names[lim] if lim >= 0 else '-',
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog='sweep',
        description='Sweep one parameter of the arm config and tabulate the specs.')
    p.add_argument('--config', default=None)
    p.add_argument('--param', required=True,
                   help='dotted path, e.g. joints.1.link.length')
    p.add_argument('--values', required=True,
                   help='a,b,c  or  start:stop:count')
    args = p.parse_args(argv)

    src = args.config or default_config_path()
    with open(src) as fh:
        base = yaml.safe_load(fh)
    try:
        original = get_path(base, args.param)
    except Exception as exc:                       # noqa: BLE001
        print(f'cannot read {args.param!r} from {src}: {exc}')
        return 2

    values = parse_values(args.values)
    print(f'config : {src}')
    print(f'sweep  : {args.param}   (currently {original})')
    print()
    header = (f'{"value":>10}{"reach mm":>11}{"mass kg":>10}{"pay@full":>10}'
              f'{"pay@700":>10}{"droop mm":>10}{"peak %":>9}{"tcp m/s":>10}'
              f'  limited by')
    print(header)
    print('-' * len(header))

    with tempfile.TemporaryDirectory(prefix='arm_lab_sweep_') as tmpdir:
        for i, value in enumerate(values):
            raw = copy.deepcopy(base)
            set_path(raw, args.param, float(value))
            try:
                r = evaluate(raw, tmpdir, i)
            except Exception as exc:               # noqa: BLE001
                print(f'{value:>10.4g}   invalid: {exc}')
                continue
            print(f"{value:>10.4g}{r['reach_mm']:>11.0f}{r['mass_kg']:>10.2f}"
                  f"{r['payload_full_kg']:>10.2f}{r['payload_700_kg']:>10.2f}"
                  f"{r['droop_mm']:>10.2f}{r['peak_util_pct']:>9.0f}"
                  f"{r['tcp_speed_ms']:>10.2f}  {r['limiting']}")
    print()
    print('pay@full / pay@700 are static payload capacity at the TCP, with the '
          'configured torque reserve applied.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
