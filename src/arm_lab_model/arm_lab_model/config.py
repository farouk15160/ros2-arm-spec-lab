"""Load and expand the arm configuration.

The YAML file holds design intent (lengths, wall thicknesses, material names,
actuator names). Everything physical -- masses, inertia tensors, torque limits,
second moments of area -- is derived here so that exactly one set of numbers
feeds the URDF, the controllers, the dashboard and the spec report.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import yaml

DEFAULT_CONFIG_ENV = 'ARM_LAB_CONFIG'


def _vec(value: Sequence[float], n: int = 3) -> List[float]:
    out = [float(v) for v in value]
    if len(out) != n:
        raise ValueError(f'expected {n} components, got {out}')
    return out


def _unit(value: Sequence[float]) -> List[float]:
    v = _vec(value)
    norm = math.sqrt(sum(c * c for c in v))
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
        )


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
        return os.path.abspath(os.path.join(here, '..', 'config', 'arm_config.yaml'))


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

    materials = {k: Material.from_dict(k, v) for k, v in raw['materials'].items()}
    actuators = {k: Actuator.from_dict(k, v) for k, v in raw['actuators'].items()}

    def _material(key: str) -> Material:
        if key not in materials:
            raise KeyError(f"unknown material '{key}'; have {sorted(materials)}")
        return materials[key]

    def _tube(d: Dict[str, Any], name: str, actuator_mass: float) -> TubeLink:
        outer_r = float(d['outer_diameter']) / 2.0
        wall = float(d['wall_thickness'])
        if wall <= 0.0 or wall >= outer_r:
            raise ValueError(
                f"{name}: wall_thickness {wall} must be >0 and < outer radius {outer_r}")
        return TubeLink(
            name=d.get('name', name),
            length=float(d['length']),
            outer_radius=outer_r,
            inner_radius=outer_r - wall,
            material=_material(d['material']),
            direction=_unit(d.get('direction', [0.0, 0.0, 1.0])),
            extra_mass=float(d.get('extra_mass', 0.0)),
            actuator_mass=actuator_mass,
        )

    robot = raw['robot']
    pedestal = _tube(robot['pedestal'], 'pedestal', 0.0)

    joints: List[JointSpec] = []
    for entry in raw['joints']:
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
    return cfg
