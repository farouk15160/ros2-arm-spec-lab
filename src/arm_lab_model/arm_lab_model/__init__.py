"""Data-driven model of a rover-mounted serial arm.

The whole workspace is generated from one YAML file: see
``arm_lab_model/config/arm_config.yaml``.
"""

from .config import ArmConfig, load_config, default_config_path  # noqa: F401
from .kinematics import ArmModel  # noqa: F401

__all__ = ['ArmConfig', 'ArmModel', 'load_config', 'default_config_path']
