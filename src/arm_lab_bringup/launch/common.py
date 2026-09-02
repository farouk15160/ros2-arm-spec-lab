"""Shared launch helpers: turn the YAML config into on-disk ROS artefacts.

Everything is regenerated on every launch into a scratch directory, so the
generated URDF and controller YAML always match the config you just edited.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Tuple

from arm_lab_model.config import load_config
from arm_lab_model.controllers_builder import dump_controllers
from arm_lab_model.urdf_builder import build_urdf

GENERATED_DIR = os.path.join(tempfile.gettempdir(), 'arm_lab_generated')


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def as_float(value: Any, default=None):
    text = str(value).strip()
    if text in ('', 'none', 'None', 'default'):
        return default
    return float(text)


def generate(config_file: str,
             ee_mass=None,
             payload_mass: float = 0.0,
             gravity=None,
             command_interface=None,
             initial_pose: str = 'home',
             use_sim_time: bool = True,
             fixed_to_world: bool = True) -> Tuple[Any, Dict[str, str]]:
    """Build the URDF and controller YAML; return the config and their paths."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    cfg = load_config(config_file, ee_mass=ee_mass, gravity=gravity)

    controllers_path = os.path.join(GENERATED_DIR, 'controllers.yaml')
    dump_controllers(cfg, controllers_path,
                     command_interface=command_interface,
                     use_sim_time=use_sim_time)

    urdf = build_urdf(cfg,
                      controllers_file=controllers_path,
                      initial_pose=initial_pose,
                      fixed_to_world=fixed_to_world,
                      command_interface=command_interface,
                      payload_mass=payload_mass)
    urdf_path = os.path.join(GENERATED_DIR, 'arm.urdf')
    with open(urdf_path, 'w') as fh:
        fh.write(urdf)

    return cfg, {
        'urdf': urdf_path,
        'urdf_xml': urdf,
        'controllers': controllers_path,
        'dir': GENERATED_DIR,
    }


def patch_world_gravity(world_path: str, gravity: float) -> str:
    """Write a copy of the world with a different gravity magnitude."""
    with open(world_path) as fh:
        text = fh.read()
    import re
    patched = re.sub(r'<gravity>[^<]*</gravity>',
                     f'<gravity>0 0 -{gravity:.6g}</gravity>', text)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    out = os.path.join(GENERATED_DIR, os.path.basename(world_path))
    with open(out, 'w') as fh:
        fh.write(patched)
    return out
