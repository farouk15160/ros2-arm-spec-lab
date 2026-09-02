"""Command line entry points that turn the config into ROS artefacts."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .config import load_config
from .controllers_builder import dump_controllers
from .urdf_builder import build_urdf


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument('--config', default=None, help='path to arm_config.yaml')
    p.add_argument('--ee-mass', type=float, default=None,
                   help='override the end-effector mass, kg')
    p.add_argument('--gravity', type=float, default=None, help='override gravity')
    p.add_argument('--command-interface', default=None,
                   choices=['position', 'velocity', 'effort'])


def urdf_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog='urdf_gen', description='Generate the arm URDF from the YAML config.')
    _common(p)
    p.add_argument('--payload-mass', type=float, default=0.0,
                   help='rigid test mass attached at the TCP, kg')
    p.add_argument('--controllers', default=None,
                   help='path to the controller YAML to embed in the gz plugin')
    p.add_argument('--initial-pose', default='home',
                   help='named pose the joints start in')
    p.add_argument('--free-base', action='store_true',
                   help='do not weld base_link to the world')
    p.add_argument('-o', '--output', default='-', help='output file, or - for stdout')
    args = p.parse_args(argv)

    cfg = load_config(args.config, ee_mass=args.ee_mass, gravity=args.gravity)
    urdf = build_urdf(
        cfg,
        controllers_file=args.controllers,
        initial_pose=args.initial_pose,
        fixed_to_world=not args.free_base,
        command_interface=args.command_interface,
        payload_mass=args.payload_mass,
    )
    if args.output == '-':
        sys.stdout.write(urdf)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as fh:
            fh.write(urdf)
        print(args.output, file=sys.stderr)
    return 0


def controllers_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog='controllers_gen',
        description='Generate the ros2_control YAML from the arm config.')
    _common(p)
    p.add_argument('--no-sim-time', action='store_true')
    p.add_argument('-o', '--output', default='-')
    args = p.parse_args(argv)

    cfg = load_config(args.config, ee_mass=args.ee_mass, gravity=args.gravity)
    if args.output == '-':
        import yaml
        from .controllers_builder import build_controllers
        yaml.safe_dump(build_controllers(
            cfg, command_interface=args.command_interface,
            use_sim_time=not args.no_sim_time), sys.stdout, sort_keys=False)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        dump_controllers(cfg, args.output,
                         command_interface=args.command_interface,
                         use_sim_time=not args.no_sim_time)
        print(args.output, file=sys.stderr)
    return 0
