"""Kinematics test bench: IK, workspace, singularity, accuracy and motion.

Everything here is driven by the same YAML configuration that builds the robot,
so a change to the arm changes the analysis with it.
"""

from .ik import IKResult, IKSolver, orientation_error  # noqa: F401
from .singularity import Metrics, classify, metrics  # noqa: F401

__all__ = ['IKSolver', 'IKResult', 'Metrics', 'metrics', 'classify',
           'orientation_error']
