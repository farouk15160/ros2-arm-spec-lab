"""Load and expand the arm configuration.

The YAML file holds design intent (lengths, wall thicknesses, material names,
actuator names). Everything physical -- masses, inertia tensors, torque limits,
second moments of area -- is derived here so that exactly one set of numbers
feeds the URDF, the controllers, the dashboard and the spec report.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import yaml
import numpy as np

DEFAULT_CONFIG_ENV = 'ARM_LAB_CONFIG'


def _vec(value: Sequence[float], n: int = 3) -> List[float]:
    out = [float(v) for v in value]
    if len(out) != n:
        raise ValueError(f'expected {n} components, got {out}')
    if not all(math.isfinite(v) for v in out):
        raise ValueError('vector components must be finite')
    return out


def _number(value, path: str, minimum: float = 0.0, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or value < minimum or (positive and value == minimum):
        relation = '>' if positive else '>='
        raise ValueError(f'{path} must be finite and {relation} {minimum}')
    return value


def _unit(value: Sequence[float]) -> List[float]:
    v = _vec(value)
    norm = math.hypot(*v)
    if norm < 1e-12:
        raise ValueError('direction/axis vector must be non-zero')
    return [c / norm for c in v]


@dataclass
class Material:
    name: str
    density: float
    youngs_modulus: float
    yield_strength: float
    color: List[float]

    @staticmethod
    def from_dict(name: str, d: Dict[str, Any]) -> 'Material':
        return Material(
            name=name,
            density=float(d['density']),
            youngs_modulus=float(d.get('youngs_modulus', 70.0e9)),
            yield_strength=float(d.get('yield_strength', 200.0e6)),
            color=[float(c) for c in d.get('color', [0.7, 0.7, 0.7, 1.0])],
        )


@dataclass
class Actuator:
    name: str
    peak_torque: float
    continuous_torque: float
    gear_ratio: float
    efficiency: float
    rotor_inertia: float
    max_motor_speed: float
    mass: float
    joint_stiffness: float
    friction: float
    quiescent_power: float
    bus_voltage: float
    # --- electrical, from the motor data sheet -------------------------------
    torque_constant: float = 0.0      # N.m/A motor side; 0 = not specified
    phase_resistance: float = 0.0     # ohm phase-to-phase at 20 C
    phase_inductance: float = 0.0
    max_phase_current: float = 0.0
    # --- thermal -------------------------------------------------------------
    thermal_resistance: float = 1.5   # K/W winding to ambient
    thermal_capacity: float = 150.0   # J/K
    max_winding_temp: float = 155.0
    ambient_temp: float = 25.0
    insulation_class: str = 'H'
    # --- feedback ------------------------------------------------------------
    encoder_bits: int = 17            # motor side
    output_encoder_bits: int = 0      # second encoder on the output; 0 = none
    # --- gearbox, for deriving stiffness from the catalogue -------------------
    gearbox_series: str = ''          # 'CSF' selects the Harmonic Drive table
    gearbox_size: int = 0
    bracket_stiffness: float = 0.0    # N.m/rad, structure in series
    bearing_stiffness: float = 0.0
    gearbox_stiffness: float = 0.0    # filled in from the catalogue
    viscous_damping: float = 0.0      # N.m.s/rad, output side

    @property
    def output_peak_torque(self) -> float:
        """Peak torque at the joint output, after gearing and losses."""
        return self.peak_torque * self.gear_ratio * self.efficiency

    @property
    def output_continuous_torque(self) -> float:
        return self.continuous_torque * self.gear_ratio * self.efficiency

    @property
    def output_max_speed(self) -> float:
        """No-load joint speed, rad/s."""
        return self.max_motor_speed / self.gear_ratio

    @property
    def reflected_inertia(self) -> float:
        """Rotor inertia seen at the joint output."""
        return self.rotor_inertia * self.gear_ratio ** 2

    @staticmethod
    def from_dict(name: str, d: Dict[str, Any]) -> 'Actuator':
        return Actuator(
            name=name,
            peak_torque=float(d['peak_torque']),
            continuous_torque=float(d.get('continuous_torque', 0.4 * float(d['peak_torque']))),
            gear_ratio=float(d.get('gear_ratio', 1.0)),
            efficiency=float(d.get('efficiency', 0.8)),
            rotor_inertia=float(d.get('rotor_inertia', 0.0)),
            max_motor_speed=float(d.get('max_motor_speed', 300.0)),
            mass=float(d.get('mass', 0.5)),
            joint_stiffness=float(d.get('joint_stiffness', 1.0e4)),
            friction=float(d.get('friction', 0.0)),
            quiescent_power=float(d.get('quiescent_power', 0.0)),
            bus_voltage=float(d.get('bus_voltage', 48.0)),
            torque_constant=float(d.get('torque_constant', 0.0)),
            phase_resistance=float(d.get('phase_resistance', 0.0)),
            phase_inductance=float(d.get('phase_inductance', 0.0)),
            max_phase_current=float(d.get('max_phase_current', 0.0)),
            thermal_resistance=float(d.get('thermal_resistance', 1.5)),
            thermal_capacity=float(d.get('thermal_capacity', 150.0)),
            max_winding_temp=float(d.get('max_winding_temp', 155.0)),
            ambient_temp=float(d.get('ambient_temp', 25.0)),
            insulation_class=str(d.get('insulation_class', 'H')),
            encoder_bits=int(d.get('encoder_bits', 17)),
            output_encoder_bits=int(d.get('output_encoder_bits', 0)),
            gearbox_series=str(d.get('gearbox_series', '')),
            gearbox_size=int(d.get('gearbox_size', 0)),
            bracket_stiffness=float(d.get('bracket_stiffness', 0.0)),
            bearing_stiffness=float(d.get('bearing_stiffness', 0.0)),
            viscous_damping=float(d.get('viscous_damping', 0.0)),
        )


@dataclass
class MeasuredInertial:
    """Complete assembly: CoM in link frame, tensor about CoM in link axes.

    Includes actuator and fittings already assigned to this moving body.
    Tensor order: ixx, iyy, izz, ixy, ixz, iyz (URDF tensor entries).
    """

    mass: float
    com: List[float]
    inertia: List[float]

    @property
    def matrix(self) -> np.ndarray:
        xx, yy, zz, xy, xz, yz = self.inertia
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

    @staticmethod
    def from_dict(d: Dict[str, Any], name: str) -> 'MeasuredInertial':
        path = f'{name}.inertial'
        if not isinstance(d, dict) or set(d) != {'mass', 'com', 'inertia'}:
            raise ValueError(f'{path} requires exactly mass, com and inertia')
        try:
            result = MeasuredInertial(_number(d['mass'], path + '.mass', positive=True),
                                      _vec(d['com']), _vec(d['inertia'], 6))
            moments = np.linalg.eigvalsh(result.matrix)
            if moments[0] <= 0 or moments[2] > moments[0] + moments[1] + 1e-12 * moments[2]:
                raise ValueError('tensor must be positive definite and obey principal-moment triangle inequalities')
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{path}: {exc}') from exc
        return result


@dataclass
class TubeLink:
    """A hollow cylinder, plus lumped extra mass at its proximal end."""

    name: str
    length: float
    outer_radius: float
    inner_radius: float
    material: Material
    direction: List[float]
    extra_mass: float           # brackets/fittings, at the proximal end
    actuator_mass: float        # motor sitting at the proximal end
    inertial: MeasuredInertial | None = None

    @property
    def com_xyz(self) -> np.ndarray:
        if self.inertial is not None:
            return np.array(self.inertial.com)
        return np.array(self.direction) * self.com_distance

    def inertia_in_link_frame(self) -> np.ndarray:
        if self.inertial is not None:
            return self.inertial.matrix
        transverse, _, axial = self.inertia_about_com()
        direction = np.array(self.direction)
        return transverse * np.eye(3) + (axial - transverse) * np.outer(direction, direction)

    @property
    def wall_thickness(self) -> float:
        return self.outer_radius - self.inner_radius

    @property
    def section_area(self) -> float:
        return math.pi * (self.outer_radius ** 2 - self.inner_radius ** 2)

    @property
    def tube_mass(self) -> float:
        return self.material.density * self.section_area * self.length

    @property
    def lumped_mass(self) -> float:
        """Mass concentrated at the joint (motor + fittings)."""
        return self.extra_mass + self.actuator_mass

    @property
    def mass(self) -> float:
        if self.inertial is not None:
            return self.inertial.mass
        return self.tube_mass + self.lumped_mass

    @property
    def second_moment_area(self) -> float:
        """I of the tube cross-section, m^4 -- drives bending deflection."""
        return math.pi / 4.0 * (self.outer_radius ** 4 - self.inner_radius ** 4)

    @property
    def section_modulus(self) -> float:
        if self.outer_radius <= 0.0:
            return 0.0
        return self.second_moment_area / self.outer_radius

    @property
    def com_distance(self) -> float:
        """Distance of the combined CoM from the joint, along `direction`.

        The tube's own CoM is at length/2; the lumped mass sits at 0.
        """
        total = self.mass
        if total <= 0.0:
            return 0.0
        return self.tube_mass * (self.length / 2.0) / total

    def inertia_about_com(self) -> List[float]:
        """[ixx, iyy, izz] in the tube frame (z along `direction`), about the
        combined CoM. Off-diagonal terms are zero by symmetry."""
        m_t = self.tube_mass
        ro2 = self.outer_radius ** 2
        ri2 = self.inner_radius ** 2
        # Hollow cylinder about its own centre.
        izz = 0.5 * m_t * (ro2 + ri2)
        ixx = (1.0 / 12.0) * m_t * (3.0 * (ro2 + ri2) + self.length ** 2)
        # Parallel-axis shift of the tube to the combined CoM.
        c = self.com_distance
        d_tube = self.length / 2.0 - c
        ixx += m_t * d_tube ** 2
        iyy = ixx
        # Lumped mass: a small solid puck at the joint (distance c away).
        m_l = self.lumped_mass
        r_l = self.outer_radius
        izz += 0.5 * m_l * r_l ** 2
        lump_ix = 0.25 * m_l * r_l ** 2 + m_l * c ** 2
        ixx += lump_ix
        iyy += lump_ix
        return [ixx, iyy, izz]


@dataclass
class JointSpec:
    name: str
    jtype: str
    origin_xyz: List[float]
    origin_rpy: List[float]
    axis: List[float]
    actuator: Actuator
    lower: float
    upper: float
    velocity_limit: float
    link: TubeLink

    @property
    def effort_limit(self) -> float:
        return self.actuator.output_peak_torque

    @property
    def usable_speed(self) -> float:
        """Commanded velocity limit, never above what the motor can spin."""
        return min(self.velocity_limit, self.actuator.output_max_speed)


@dataclass
class EndEffector:
    """The tool. `mass` is the whole assembly including the jaws."""

    name: str
    mass: float
    tcp_offset: float
    com_offset: float
    body_length: float
    body_width: float
    body_height: float
    material: Material
    simulate_fingers: bool
    stroke: float
    finger_length: float
    finger_thickness: float
    finger_width: float
    finger_mass: float
    grip_force_min: float
    grip_force_max: float
    grip_speed: float
    grasp_mode: str = 'force'

    @property
    def jaw_span(self) -> tuple:
        """Distance from the flange to the start and end of the jaws."""
        return (self.body_length, self.body_length + self.finger_length)

    @property
    def tcp_between_jaws(self) -> bool:
        """Is the tool centre point actually inside the grasp?

        If the TCP sits short of the jaws it is inside the gripper body, and
        every grasp will close on empty air while the object sits outside the
        jaws. That failure looks like a controller problem and is not one.
        """
        if not self.simulate_fingers:
            return True
        start, end = self.jaw_span
        return start - 1e-9 <= self.tcp_offset <= end + 1e-9

    @property
    def finger_joint_names(self) -> List[str]:
        if not self.simulate_fingers:
            return []
        return [f'{self.name}_left_joint', f'{self.name}_right_joint']


@dataclass
class ArmConfig:
    source_path: str
    raw: Dict[str, Any]
    name: str
    mount_xyz: List[float]
    mount_rpy: List[float]
    pedestal: TubeLink
    gravity: float
    reach_reference_joint: str
    materials: Dict[str, Material]
    actuators: Dict[str, Actuator]
    joints: List[JointSpec]
    end_effector: EndEffector
    control: Dict[str, Any]
    can_bus: Dict[str, Any]
    spec_targets: Dict[str, Any]
    test_poses: Dict[str, Any]
    payload_mass: float = 0.0

    @property
    def dof(self) -> int:
        return len(self.joints)

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.joints]

    @property
    def arm_mass(self) -> float:
        """Everything the rover has to carry, including the gripper.

        `end_effector.mass` is the complete assembly, jaws included, so the
        finger masses are carved out of it rather than added to it.
        """
        return (self.pedestal.mass + sum(j.link.mass for j in self.joints)
                + self.end_effector.mass)

    @property
    def structure_mass(self) -> float:
        return self.pedestal.mass + sum(j.link.mass for j in self.joints)

    def pose(self, name_or_list) -> Any:
        """Resolve a named pose to joint angles, or pass a list through."""
        if isinstance(name_or_list, (list, tuple)):
            return _pad(list(name_or_list), self.dof)
        entry = self.test_poses.get(name_or_list, name_or_list)
        return entry


def _derive_joint_stiffness(act: Actuator) -> None:
    """Replace a guessed joint stiffness with one derived from the catalogue.

    A named gearbox gives a real K3 from the manufacturer's torsional stiffness
    table. That is the *gearbox*, not the joint: brackets and bearings sit in
    series with it and are usually far softer, so quoting the catalogue figure
    as the joint stiffness overstates it by an order of magnitude. When those
    are given too, they are combined properly.
    """
    if not act.gearbox_series or not act.gearbox_size:
        return
    from .reference import gearbox_table, structural_series_stiffness
    table = gearbox_table(act.gear_ratio)
    entry = table.get(act.gearbox_size)
    if entry is None:
        raise KeyError(
            f'{act.name}: no catalogue entry for size {act.gearbox_size}; '
            f'available {sorted(table)}')
    act.gearbox_stiffness = entry.k3
    act.joint_stiffness = structural_series_stiffness(
        entry.k3, act.bracket_stiffness, act.bearing_stiffness)


def _pad(values: List[float], n: int) -> List[float]:
    values = [float(v) for v in values]
    if len(values) < n:
        values = values + [0.0] * (n - len(values))
    return values[:n]


def default_config_path() -> str:
    """The installed arm_config.yaml, unless ARM_LAB_CONFIG overrides it."""
    override = os.environ.get(DEFAULT_CONFIG_ENV)
    if override:
        return override
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory('arm_lab_model'), 'config', 'arm_config.yaml')
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        source = os.path.abspath(os.path.join(here, '..', 'config', 'arm_config.yaml'))
        if os.path.isfile(source):
            return source
        return os.path.join(sys.prefix, 'share', 'arm_lab_model', 'config', 'arm_config.yaml')


def load_config(path: str | None = None,
                ee_mass: float | None = None,
                payload_mass: float | None = None,
                gravity: float | None = None) -> ArmConfig:
    """Read the YAML and expand it into a fully derived model.

    ee_mass / payload_mass / gravity are the launch-time overrides.
    """
    path = path or default_config_path()
    with open(path, 'r') as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError('configuration must be a YAML mapping')
    for section in ('robot', 'materials', 'actuators', 'end_effector'):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f'{section} must be a mapping')
    if not isinstance(raw.get('joints'), list) or not raw['joints']:
        raise ValueError('joints must be a non-empty list')

    materials = {k: Material.from_dict(k, v) for k, v in raw['materials'].items()}
    actuators = {k: Actuator.from_dict(k, v) for k, v in raw['actuators'].items()}
    for act in actuators.values():
        for key in ('peak_torque', 'continuous_torque', 'gear_ratio', 'max_motor_speed',
                    'joint_stiffness', 'bus_voltage', 'thermal_resistance', 'thermal_capacity'):
            _number(getattr(act, key), f'actuators.{act.name}.{key}', positive=True)
        for key in ('mass', 'rotor_inertia', 'friction', 'viscous_damping', 'quiescent_power',
                    'torque_constant', 'phase_resistance', 'phase_inductance', 'max_phase_current',
                    'bracket_stiffness', 'bearing_stiffness'):
            _number(getattr(act, key), f'actuators.{act.name}.{key}')
        if not 0 < act.efficiency <= 1:
            raise ValueError(f'actuators.{act.name}.efficiency must be in (0, 1]')
        if act.continuous_torque > act.peak_torque:
            raise ValueError(f'actuators.{act.name}.continuous_torque exceeds peak_torque')
        if not all(math.isfinite(v) for v in (act.ambient_temp, act.max_winding_temp)) or act.max_winding_temp <= act.ambient_temp:
            raise ValueError(f'actuators.{act.name}: max_winding_temp must exceed ambient_temp')
        _derive_joint_stiffness(act)
    for mat in materials.values():
        for key in ('density', 'youngs_modulus', 'yield_strength'):
            _number(getattr(mat, key), f'materials.{mat.name}.{key}', positive=True)

    def _material(key: str) -> Material:
        if key not in materials:
            raise KeyError(f"unknown material '{key}'; have {sorted(materials)}")
        return materials[key]

    def _tube(d: Dict[str, Any], name: str, actuator_mass: float) -> TubeLink:
        outer_r = _number(d['outer_diameter'], name + '.outer_diameter', positive=True) / 2.0
        wall = _number(d['wall_thickness'], name + '.wall_thickness', positive=True)
        if wall <= 0.0 or wall >= outer_r:
            raise ValueError(
                f"{name}: wall_thickness {wall} must be >0 and < outer radius {outer_r}")
        return TubeLink(
            name=d.get('name', name),
            length=_number(d['length'], name + '.length', positive=True),
            outer_radius=outer_r,
            inner_radius=outer_r - wall,
            material=_material(d['material']),
            direction=_unit(d.get('direction', [0.0, 0.0, 1.0])),
            extra_mass=_number(d.get('extra_mass', 0.0), name + '.extra_mass'),
            actuator_mass=actuator_mass,
            inertial=MeasuredInertial.from_dict(d['inertial'], name) if 'inertial' in d else None,
        )

    robot = raw['robot']
    pedestal = _tube(robot['pedestal'], 'pedestal', 0.0)

    joints: List[JointSpec] = []
    for entry in raw['joints']:
        if entry.get('type', 'revolute') not in ('revolute', 'continuous'):
            raise ValueError(f"{entry['name']}: joint type must be revolute or continuous; linear transmissions and branched robots require a separate model")
        act_key = entry['actuator']
        if act_key not in actuators:
            raise KeyError(f"unknown actuator '{act_key}'; have {sorted(actuators)}")
        act = actuators[act_key]
        limits = entry.get('limits', {})
        joints.append(JointSpec(
            name=entry['name'],
            jtype=entry.get('type', 'revolute'),
            origin_xyz=_vec(entry.get('origin_xyz', [0.0, 0.0, 0.0])),
            origin_rpy=_vec(entry.get('origin_rpy', [0.0, 0.0, 0.0])),
            axis=_unit(entry.get('axis', [0.0, 0.0, 1.0])),
            actuator=act,
            lower=float(limits.get('lower', -3.14159)),
            upper=float(limits.get('upper', 3.14159)),
            velocity_limit=float(limits.get('velocity', 1.0)),
            link=_tube(entry['link'], entry['link'].get('name', entry['name'] + '_link'),
                       act.mass),
        ))

    names = [j.name for j in joints]
    links = ['world', 'base_link'] + [j.link.name for j in joints]
    if len(set(names)) != len(names) or len(set(links)) != len(links):
        raise ValueError('joint and link names must be unique within their namespace')
    for joint in joints:
        if not math.isfinite(joint.lower) or not math.isfinite(joint.upper) or joint.lower >= joint.upper:
            raise ValueError(f'{joint.name}: lower must be finite and less than upper')
        _number(joint.velocity_limit, joint.name + '.velocity', positive=True)

    ee_raw = raw['end_effector']
    ee = EndEffector(
        name=ee_raw.get('name', 'gripper'),
        mass=float(ee_mass if ee_mass is not None else ee_raw['mass']),
        tcp_offset=float(ee_raw.get('tcp_offset', 0.05)),
        com_offset=float(ee_raw.get('com_offset', 0.04)),
        body_length=float(ee_raw.get('body_length', 0.09)),
        body_width=float(ee_raw.get('body_width', 0.10)),
        body_height=float(ee_raw.get('body_height', 0.07)),
        material=_material(ee_raw.get('material', 'pa12_sls')),
        simulate_fingers=bool(ee_raw.get('simulate_fingers', True)),
        stroke=float(ee_raw.get('stroke', 0.18)),
        finger_length=float(ee_raw.get('finger_length', 0.07)),
        finger_thickness=float(ee_raw.get('finger_thickness', 0.012)),
        finger_width=float(ee_raw.get('finger_width', 0.03)),
        finger_mass=float(ee_raw.get('finger_mass', 0.06)),
        grip_force_min=float(ee_raw.get('grip_force_min', 10.0)),
        grip_force_max=float(ee_raw.get('grip_force_max', 100.0)),
        grip_speed=float(ee_raw.get('grip_speed', 0.05)),
        grasp_mode=str(ee_raw.get('grasp_mode', 'force')).lower(),
    )

    env = raw.get('environment', {})
    cfg = ArmConfig(
        source_path=os.path.abspath(path),
        raw=raw,
        name=robot.get('name', 'rover_arm'),
        mount_xyz=_vec(robot.get('mount_xyz', [0.0, 0.0, 0.0])),
        mount_rpy=_vec(robot.get('mount_rpy', [0.0, 0.0, 0.0])),
        pedestal=pedestal,
        gravity=float(gravity if gravity is not None else env.get('gravity', 9.81)),
        reach_reference_joint=env.get('reach_reference_joint', joints[1].name
                                      if len(joints) > 1 else joints[0].name),
        materials=materials,
        actuators=actuators,
        joints=joints,
        end_effector=ee,
        control=raw.get('control', {}),
        can_bus=raw.get('can_bus', {}),
        spec_targets=raw.get('spec_targets', {}),
        test_poses=raw.get('test_poses', {}),
        payload_mass=float(payload_mass or 0.0),
    )
    _number(cfg.gravity, 'environment.gravity')
    _number(cfg.payload_mass, 'payload_mass')
    _number(ee.mass, 'end_effector.mass')
    for key in ('body_length', 'body_width', 'body_height', 'finger_length', 'finger_width', 'finger_thickness'):
        _number(getattr(ee, key), f'end_effector.{key}', positive=True)
    for key in ('tcp_offset', 'com_offset'):
        if not math.isfinite(getattr(ee, key)):
            raise ValueError(f'end_effector.{key} must be finite')
    _number(ee.finger_mass, 'end_effector.finger_mass')
    if ee.simulate_fingers and ee.mass <= 2 * ee.finger_mass:
        raise ValueError('end_effector.mass must exceed the mass of both fingers')
    if cfg.reach_reference_joint not in cfg.joint_names:
        raise ValueError('environment.reach_reference_joint must name an existing joint')
    return cfg
