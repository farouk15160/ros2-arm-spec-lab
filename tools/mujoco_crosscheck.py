"""Compatibility entry point for the reproducible direct-MJCF cross-check.

Prefer: robot_test check --config path/to/arm.yaml --output check.json
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'arm_lab_model'))
from arm_lab_model.mujoco_backend import main

if __name__ == '__main__':
    sys.exit(main(['check', *sys.argv[1:]]))
