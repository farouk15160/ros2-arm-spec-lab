"""Declarative description of every editable parameter.

The editor is generated from this list rather than hand-built, so adding a
parameter means adding one line here and nothing else. `path` is a dotted route
into the raw YAML, with `{i}` standing in for a joint index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence


@dataclass
class Field:
    path: str
    label: str
    kind: str = 'float'          # float | int | str | bool | choice | vec3
    unit: str = ''
    tip: str = ''
    minimum: float = -1e9
    maximum: float = 1e9
    step: float = 0.01
    decimals: int = 4
    choices: Optional[str] = None    # 'materials' | 'actuators' | explicit list
    options: Sequence[str] = ()


@dataclass
class Group:
    title: str
    fields: List[Field] = field(default_factory=list)
    note: str = ''


def robot_groups() -> List[Group]:
    return [
        Group('Mounting', [
            Field('robot.name', 'Robot name', 'str',
                  tip='Used for the URDF and the Gazebo model name.'),
            Field('robot.mount_xyz', 'Mount position', 'vec3', 'm',
                  tip='Where the shoulder yaw axis sits on the rover deck.'),
            Field('robot.mount_rpy', 'Mount orientation', 'vec3', 'rad'),
            Field('environment.gravity', 'Gravity', 'float', 'm/s^2',
                  tip='9.81 Earth, 3.72 Mars, 1.62 Moon. Changes every load '
                      'figure and the Gazebo world.',
                  minimum=0.0, maximum=30.0, step=0.01, decimals=3),
        ]),
        Group('Pedestal (the part bolted to the rover)', [
            Field('robot.pedestal.length', 'Length', 'float', 'm',
                  minimum=0.0, maximum=2.0, step=0.005),
            Field('robot.pedestal.outer_diameter', 'Outer diameter', 'float', 'm',
                  minimum=0.005, maximum=1.0, step=0.002),
            Field('robot.pedestal.wall_thickness', 'Wall thickness', 'float', 'm',
                  tip='Must be greater than zero and less than the outer radius.',
                  minimum=0.0002, maximum=0.2, step=0.0005, decimals=5),
            Field('robot.pedestal.material', 'Material', 'choice',
                  choices='materials'),
            Field('robot.pedestal.extra_mass', 'Lumped extra mass', 'float', 'kg',
                  tip='Bolts, connector plate, harness entry.',
                  minimum=0.0, maximum=50.0, step=0.05, decimals=3),
        ]),
    ]


def joint_groups(index: int) -> List[Group]:
    i = index
    return [
        Group('Joint', [
            Field(f'joints.{i}.name', 'Name', 'str'),
            Field(f'joints.{i}.type', 'Type', 'choice',
                  options=('revolute', 'prismatic')),
            Field(f'joints.{i}.origin_xyz', 'Origin offset', 'vec3', 'm',
                  tip='Measured from the previous link\'s far end. Usually zero: '
                      'link lengths already place the joints.'),
            Field(f'joints.{i}.origin_rpy', 'Origin rotation', 'vec3', 'rad',
                  tip='Applied before the joint rotation. This is what turns a '
                      'yaw joint into a pitch joint.'),
            Field(f'joints.{i}.axis', 'Rotation axis', 'vec3',
                  tip='In the joint frame, after the origin rotation.'),
        ]),
        Group('Limits', [
            Field(f'joints.{i}.limits.lower', 'Lower limit', 'float', 'rad',
                  minimum=-12.0, maximum=12.0, step=0.05, decimals=3),
            Field(f'joints.{i}.limits.upper', 'Upper limit', 'float', 'rad',
                  minimum=-12.0, maximum=12.0, step=0.05, decimals=3),
            Field(f'joints.{i}.limits.velocity', 'Speed limit', 'float', 'rad/s',
                  tip='Capped by the actuator\'s own no-load speed.',
                  minimum=0.01, maximum=50.0, step=0.05, decimals=3),
        ]),
        Group('Actuator', [
            Field(f'joints.{i}.actuator', 'Actuator', 'choice',
                  choices='actuators',
                  tip='Torque limit is peak torque x gear ratio x efficiency.'),
        ]),
        Group('Link (hollow tube)', [
            Field(f'joints.{i}.link.name', 'Link name', 'str'),
            Field(f'joints.{i}.link.length', 'Length', 'float', 'm',
                  tip='Everything downstream moves with this. Reach, mass and '
                      'gravity torque all follow.',
                  minimum=0.001, maximum=5.0, step=0.005),
            Field(f'joints.{i}.link.outer_diameter', 'Outer diameter', 'float',
                  'm', minimum=0.004, maximum=1.0, step=0.002),
            Field(f'joints.{i}.link.wall_thickness', 'Wall thickness', 'float',
                  'm', tip='Drives bending stiffness far more than mass does.',
                  minimum=0.0002, maximum=0.2, step=0.0005, decimals=5),
            Field(f'joints.{i}.link.material', 'Material', 'choice',
                  choices='materials'),
            Field(f'joints.{i}.link.direction', 'Tube direction', 'vec3',
                  tip='Which way the tube runs out of the joint frame.'),
            Field(f'joints.{i}.link.extra_mass', 'Lumped extra mass', 'float',
                  'kg', tip='Fittings and brackets, carried at the joint.',
                  minimum=0.0, maximum=50.0, step=0.02, decimals=3),
        ]),
    ]


def end_effector_groups() -> List[Group]:
    return [
        Group('Tool', [
            Field('end_effector.name', 'Name', 'str'),
            Field('end_effector.mass', 'Total mass', 'float', 'kg',
                  tip='The WHOLE assembly including both jaws. Overridable at '
                      'launch with ee_mass:=<kg>.',
                  minimum=0.0, maximum=50.0, step=0.05, decimals=3),
            Field('end_effector.tcp_offset', 'Flange to TCP', 'float', 'm',
                  minimum=0.0, maximum=1.0, step=0.005),
            Field('end_effector.com_offset', 'Flange to centre of mass', 'float',
                  'm', minimum=0.0, maximum=1.0, step=0.005),
            Field('end_effector.body_length', 'Body length', 'float', 'm',
                  minimum=0.001, maximum=1.0, step=0.005),
            Field('end_effector.body_width', 'Body width', 'float', 'm',
                  minimum=0.001, maximum=1.0, step=0.005),
            Field('end_effector.body_height', 'Body height', 'float', 'm',
                  minimum=0.001, maximum=1.0, step=0.005),
            Field('end_effector.material', 'Material', 'choice',
                  choices='materials'),
        ]),
        Group('Jaws', [
            Field('end_effector.simulate_fingers', 'Simulate the jaws', 'bool',
                  tip='Off models the tool as one rigid lump: faster, fewer '
                      'contacts, no grasping.'),
            Field('end_effector.stroke', 'Total opening', 'float', 'm',
                  minimum=0.0, maximum=1.0, step=0.005),
            Field('end_effector.finger_length', 'Jaw length', 'float', 'm',
                  minimum=0.001, maximum=0.5, step=0.005),
            Field('end_effector.finger_thickness', 'Jaw thickness', 'float', 'm',
                  minimum=0.001, maximum=0.2, step=0.002),
            Field('end_effector.finger_width', 'Jaw width', 'float', 'm',
                  minimum=0.001, maximum=0.5, step=0.005),
            Field('end_effector.finger_mass', 'Jaw mass (each)', 'float', 'kg',
                  tip='Carved out of the total mass above, not added to it.',
                  minimum=0.0, maximum=10.0, step=0.01, decimals=4),
            Field('end_effector.grip_force_min', 'Minimum grip force', 'float',
                  'N', minimum=0.0, maximum=5000.0, step=1.0, decimals=1),
            Field('end_effector.grip_force_max', 'Maximum grip force', 'float',
                  'N', minimum=0.0, maximum=5000.0, step=1.0, decimals=1),
            Field('end_effector.grip_speed', 'Jaw speed', 'float', 'm/s',
                  minimum=0.001, maximum=2.0, step=0.005),
        ]),
    ]


def control_groups() -> List[Group]:
    return [
        Group('Control loop', [
            Field('control.update_rate', 'Update rate', 'int', 'Hz',
                  minimum=1, maximum=5000, step=10),
            Field('control.command_interface', 'Command interface', 'choice',
                  options=('velocity', 'position', 'effort'),
                  tip='velocity is stable and respects torque limits; effort is '
                      'the honest torque test; position is visualisation only.'),
            Field('control.dynamic_torque_reserve', 'Torque held in reserve',
                  'float', 'fraction',
                  tip='Held back for acceleration when quoting payload.',
                  minimum=0.0, maximum=0.9, step=0.05, decimals=3),
            Field('control.tcp_speed_limit', 'TCP speed limit', 'float', 'm/s',
                  minimum=0.001, maximum=10.0, step=0.01, decimals=3),
            Field('control.tcp_acceleration', 'TCP acceleration for the check',
                  'float', 'm/s^2', minimum=0.0, maximum=50.0, step=0.05,
                  decimals=3),
            Field('control.contact_force_limit', 'Contact force limit', 'float',
                  'N', minimum=0.0, maximum=5000.0, step=1.0, decimals=1),
        ]),
        Group('Trajectory controller gains', [
            Field('control.gains.p_scale', 'Proportional scale', 'float',
                  minimum=0.0, maximum=1000.0, step=0.5, decimals=3),
            Field('control.gains.i_scale', 'Integral scale', 'float',
                  minimum=0.0, maximum=1000.0, step=0.1, decimals=3),
            Field('control.gains.d_scale', 'Derivative scale', 'float',
                  minimum=0.0, maximum=1000.0, step=0.1, decimals=3),
            Field('control.gains.i_clamp', 'Integral clamp', 'float',
                  minimum=0.0, maximum=1000.0, step=0.5, decimals=3),
        ]),
        Group('Cartesian motion', [
            Field('motion.cartesian_speed', 'Cartesian speed', 'float', 'm/s',
                  minimum=0.001, maximum=10.0, step=0.01, decimals=3),
            Field('motion.cartesian_accel', 'Cartesian acceleration', 'float',
                  'm/s^2', minimum=0.001, maximum=50.0, step=0.05, decimals=3),
            Field('motion.joint_acceleration', 'Joint acceleration ceiling',
                  'float', 'rad/s^2',
                  tip='Often the real limit on cycle time, ahead of torque.',
                  minimum=0.01, maximum=200.0, step=0.5, decimals=3),
            Field('motion.path_resolution', 'Waypoint spacing', 'float', 'm',
                  minimum=0.0005, maximum=0.1, step=0.001, decimals=4),
        ]),
        Group('CAN bus', [
            Field('can_bus.bitrate', 'Bitrate', 'int', 'bit/s',
                  minimum=10000, maximum=10000000, step=125000),
            Field('can_bus.frames_per_joint_per_cycle', 'Frames per joint',
                  'int', minimum=1, maximum=20, step=1),
            Field('can_bus.bits_per_frame', 'Bits per frame', 'int',
                  minimum=20, maximum=200, step=1),
            Field('can_bus.max_utilisation', 'Maximum bus load', 'float',
                  'fraction', minimum=0.05, maximum=1.0, step=0.05, decimals=3),
        ]),
    ]


def error_groups() -> List[Group]:
    return [
        Group('Systematic (sets accuracy, correctable by calibration)', [
            Field('errors.backlash_arcmin', 'Gearbox backlash', 'float', 'arcmin',
                  minimum=0.0, maximum=120.0, step=0.5, decimals=3),
            Field('errors.joint_zero_offset_arcmin', 'Joint zero offset',
                  'float', 'arcmin (1 sigma)',
                  minimum=0.0, maximum=120.0, step=0.5, decimals=3),
            Field('errors.link_length_tolerance', 'Link length tolerance',
                  'float', 'm (1 sigma)',
                  minimum=0.0, maximum=0.02, step=0.0001, decimals=6),
            Field('errors.include_compliance', 'Include gravity droop', 'bool'),
            Field('errors.thermal_drift_arcmin', 'Thermal drift', 'float',
                  'arcmin (1 sigma)',
                  minimum=0.0, maximum=120.0, step=0.5, decimals=3),
        ]),
        Group('Random (sets repeatability, not correctable)', [
            Field('errors.encoder_bits', 'Encoder resolution', 'int', 'bits',
                  minimum=8, maximum=28, step=1),
            Field('errors.stiction_deadband_arcmin', 'Stiction deadband',
                  'float', 'arcmin (1 sigma)',
                  minimum=0.0, maximum=60.0, step=0.1, decimals=3),
        ]),
    ]


def spec_groups() -> List[Group]:
    def f(key, label, unit='', **kw):
        return Field(f'spec_targets.{key}', label, 'float', unit, **kw)
    return [
        Group('Geometry and mass', [
            Field('spec_targets.dof', 'Degrees of freedom', 'int',
                  minimum=1, maximum=20, step=1),
            f('reach_min', 'Minimum reach', 'm', step=0.005),
            f('reach_max', 'Maximum reach', 'm', step=0.005),
            f('arm_mass_target', 'Arm mass target', 'kg', step=0.1, decimals=3),
            f('arm_mass_max', 'Arm mass hard limit', 'kg', step=0.1, decimals=3),
            f('ee_mass_allowance', 'End-effector allowance', 'kg', step=0.05,
              decimals=3),
        ]),
        Group('Performance', [
            f('payload_at_full_reach', 'Payload at full reach', 'kg', step=0.1,
              decimals=3),
            f('payload_at_700mm', 'Payload at 700 mm', 'kg', step=0.1,
              decimals=3),
            f('tcp_speed', 'TCP speed', 'm/s', step=0.01, decimals=3),
            f('contact_force_limit', 'Contact force limit', 'N', step=1.0,
              decimals=1),
        ]),
        Group('Accuracy', [
            f('raw_positioning_accuracy', 'Raw positioning accuracy', 'm',
              step=0.001, decimals=5),
            f('vision_corrected_repeatability', 'Vision-corrected repeatability',
              'm', step=0.0005, decimals=5),
            f('orientation_accuracy_deg', 'Orientation accuracy', 'deg',
              step=0.1, decimals=3),
        ]),
        Group('Electrical', [
            f('main_voltage', 'Main bus', 'V', step=1.0, decimals=1),
            f('secondary_voltage', 'Secondary bus', 'V', step=1.0, decimals=1),
            Field('spec_targets.control_rate', 'Control rate', 'int', 'Hz',
                  minimum=1, maximum=5000, step=10),
            Field('spec_targets.can_bitrate', 'CAN bitrate', 'int', 'bit/s',
                  minimum=10000, maximum=10000000, step=125000),
        ]),
    ]


ACTUATOR_FIELDS = [
    ('peak_torque', 'Peak torque', 'N.m', 4),
    ('continuous_torque', 'Continuous torque', 'N.m', 4),
    ('gear_ratio', 'Gear ratio', '', 2),
    ('efficiency', 'Efficiency', '', 3),
    ('rotor_inertia', 'Rotor inertia', 'kg.m^2', 8),
    ('max_motor_speed', 'Max motor speed', 'rad/s', 2),
    ('mass', 'Mass', 'kg', 3),
    ('joint_stiffness', 'Output stiffness', 'N.m/rad', 1),
    ('friction', 'Friction', 'N.m', 3),
    ('quiescent_power', 'Quiescent power', 'W', 2),
    ('bus_voltage', 'Bus voltage', 'V', 1),
]

MATERIAL_FIELDS = [
    ('density', 'Density', 'kg/m^3', 1),
    ('youngs_modulus', "Young's modulus", 'Pa', 1),
    ('yield_strength', 'Yield strength', 'Pa', 1),
]
