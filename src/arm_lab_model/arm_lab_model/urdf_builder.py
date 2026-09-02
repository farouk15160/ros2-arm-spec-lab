"""Generate the URDF (with Gazebo + ros2_control tags) from the config.

A generator is used instead of a hand-written xacro because the geometry is
fully data-driven: link count, tube sizing, inertia tensors and joint limits all
fall out of the YAML, and xacro has no loops.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence
from xml.sax.saxutils import escape

import numpy as np

from .config import ArmConfig, TubeLink
from .kinematics import ArmModel, _frame_from_direction, matrix_to_rpy


def _fmt(values: Sequence[float]) -> str:
    return ' '.join(f'{float(v):.9g}' for v in values)


def _tube_local_frame(direction: Sequence[float]):
    """Rotation (and its rpy) taking the cylinder's z axis onto `direction`."""
    R = _frame_from_direction(np.asarray(direction, dtype=float))
    return R, matrix_to_rpy(R)


class UrdfBuilder:
    def __init__(self, cfg: ArmConfig,
                 controllers_file: Optional[str] = None,
                 initial_pose: Optional[Sequence[float]] = None,
                 fixed_to_world: bool = True,
                 command_interface: Optional[str] = None,
                 payload_mass: float = 0.0):
        self.cfg = cfg
        self.model = ArmModel(cfg)
        self.controllers_file = controllers_file
        self.fixed_to_world = fixed_to_world
        self.command_interface = command_interface or cfg.control.get(
            'command_interface', 'velocity')
        self.payload_mass = float(payload_mass)
        if initial_pose is None:
            initial_pose = cfg.test_poses.get('home', [0.0] * cfg.dof)
        self.q0 = self.model.resolve_pose(initial_pose)
        self.out: List[str] = []

    # ------------------------------------------------------------- helpers
    def _w(self, text: str = '') -> None:
        self.out.append(text)

    def _material_tags(self) -> None:
        for name, mat in self.cfg.materials.items():
            self._w(f'  <material name="{escape(name)}">')
            self._w(f'    <color rgba="{_fmt(mat.color)}"/>')
            self._w('  </material>')

    def _inertial(self, link: TubeLink, indent: str = '    ') -> None:
        R, rpy = _tube_local_frame(link.direction)
        com = np.asarray(link.direction, dtype=float) * link.com_distance
        ixx, iyy, izz = link.inertia_about_com()
        self._w(f'{indent}<inertial>')
        self._w(f'{indent}  <origin xyz="{_fmt(com)}" rpy="{_fmt(rpy)}"/>')
        self._w(f'{indent}  <mass value="{link.mass:.9g}"/>')
        self._w(f'{indent}  <inertia ixx="{ixx:.9g}" ixy="0" ixz="0" '
                f'iyy="{iyy:.9g}" iyz="0" izz="{izz:.9g}"/>')
        self._w(f'{indent}</inertial>')

    def _box_inertial(self, mass: float, dims: Sequence[float],
                      origin_xyz: Sequence[float], origin_rpy: Sequence[float],
                      indent: str = '    ') -> None:
        lx, ly, lz = (max(float(d), 1e-4) for d in dims)
        ixx = mass * (ly * ly + lz * lz) / 12.0
        iyy = mass * (lx * lx + lz * lz) / 12.0
        izz = mass * (lx * lx + ly * ly) / 12.0
        self._w(f'{indent}<inertial>')
        self._w(f'{indent}  <origin xyz="{_fmt(origin_xyz)}" rpy="{_fmt(origin_rpy)}"/>')
        self._w(f'{indent}  <mass value="{max(mass, 1e-6):.9g}"/>')
        self._w(f'{indent}  <inertia ixx="{ixx:.9g}" ixy="0" ixz="0" '
                f'iyy="{iyy:.9g}" iyz="0" izz="{izz:.9g}"/>')
        self._w(f'{indent}</inertial>')

    def _tube_geometry(self, link: TubeLink, tag: str) -> None:
        R, rpy = _tube_local_frame(link.direction)
        mid = np.asarray(link.direction, dtype=float) * (link.length / 2.0)
        self._w(f'    <{tag}>')
        self._w(f'      <origin xyz="{_fmt(mid)}" rpy="{_fmt(rpy)}"/>')
        self._w('      <geometry>')
        self._w(f'        <cylinder radius="{link.outer_radius:.9g}" '
                f'length="{link.length:.9g}"/>')
        self._w('      </geometry>')
        if tag == 'visual':
            self._w(f'      <material name="{escape(link.material.name)}"/>')
        self._w(f'    </{tag}>')

    def _actuator_housing(self, link: TubeLink, radius_scale: float = 1.15) -> None:
        """A little drum at the joint so the motor mass is visible in RViz."""
        if link.lumped_mass <= 0.0:
            return
        R, rpy = _tube_local_frame(link.direction)
        r = link.outer_radius * radius_scale
        h = min(0.9 * link.length, 2.0 * r) if link.length > 0 else 2.0 * r
        mid = np.asarray(link.direction, dtype=float) * (h / 2.0)
        self._w('    <visual>')
        self._w(f'      <origin xyz="{_fmt(mid)}" rpy="{_fmt(rpy)}"/>')
        self._w('      <geometry>')
        self._w(f'        <cylinder radius="{r:.9g}" length="{h:.9g}"/>')
        self._w('      </geometry>')
        self._w('      <material name="__actuator"/>')
        self._w('    </visual>')

    def _gazebo_link(self, name: str, mu: float = 0.9,
                     material: str = 'Gazebo/Grey') -> None:
        self._w(f'  <gazebo reference="{name}">')
        self._w(f'    <mu1>{mu}</mu1>')
        self._w(f'    <mu2>{mu}</mu2>')
        self._w('    <kp>1e7</kp>')
        self._w('    <kd>1e3</kd>')
        self._w('    <selfCollide>false</selfCollide>')
        self._w('  </gazebo>')

    # --------------------------------------------------------------- build
    def build(self) -> str:
        cfg = self.cfg
        self._w('<?xml version="1.0"?>')
        self._w(f'<!-- generated by arm_lab_model from {cfg.source_path} -->')
        self._w(f'<!-- DO NOT EDIT: change the YAML and re-launch instead -->')
        self._w(f'<robot name="{escape(cfg.name)}">')
        self._material_tags()
        self._w('  <material name="__actuator"><color rgba="0.25 0.28 0.32 1"/></material>')
        self._w('  <material name="__payload"><color rgba="0.85 0.45 0.10 1"/></material>')
        self._w('  <material name="__tcp"><color rgba="0.10 0.80 0.35 1"/></material>')

        if self.fixed_to_world:
            self._w('  <link name="world"/>')
            self._w('  <joint name="world_to_base" type="fixed">')
            self._w('    <parent link="world"/>')
            self._w('    <child link="base_link"/>')
            self._w(f'    <origin xyz="{_fmt(cfg.mount_xyz)}" rpy="{_fmt(cfg.mount_rpy)}"/>')
            self._w('  </joint>')

        # ------------------------------------------------------ base/pedestal
        ped = cfg.pedestal
        self._w('  <link name="base_link">')
        self._inertial(ped)
        self._tube_geometry(ped, 'visual')
        self._tube_geometry(ped, 'collision')
        self._w('  </link>')
        self._gazebo_link('base_link')

        parent = 'base_link'
        parent_link = ped
        for joint in cfg.joints:
            link = joint.link
            offset = (np.asarray(parent_link.direction, dtype=float) * parent_link.length
                      + np.asarray(joint.origin_xyz, dtype=float))
            damping = joint.actuator.friction * 0.1
            self._w(f'  <joint name="{escape(joint.name)}" type="{joint.jtype}">')
            self._w(f'    <parent link="{parent}"/>')
            self._w(f'    <child link="{escape(link.name)}"/>')
            self._w(f'    <origin xyz="{_fmt(offset)}" rpy="{_fmt(joint.origin_rpy)}"/>')
            self._w(f'    <axis xyz="{_fmt(joint.axis)}"/>')
            self._w(f'    <limit lower="{joint.lower:.9g}" upper="{joint.upper:.9g}" '
                    f'effort="{joint.effort_limit:.9g}" '
                    f'velocity="{joint.usable_speed:.9g}"/>')
            self._w(f'    <dynamics damping="{damping:.9g}" '
                    f'friction="{joint.actuator.friction:.9g}"/>')
            self._w('  </joint>')

            self._w(f'  <link name="{escape(link.name)}">')
            self._inertial(link)
            self._actuator_housing(link)
            self._tube_geometry(link, 'visual')
            self._tube_geometry(link, 'collision')
            self._w('  </link>')
            self._gazebo_link(link.name)

            parent = link.name
            parent_link = link

        self._build_end_effector(parent, parent_link)
        self._build_ros2_control()
        self._build_gazebo_plugins()
        self._w('</robot>')
        return '\n'.join(self.out) + '\n'

    # -------------------------------------------------------- end effector
    def _build_end_effector(self, parent: str, parent_link: TubeLink) -> None:
        ee = self.cfg.end_effector
        R_local, rpy_local = _tube_local_frame(parent_link.direction)
        flange = np.asarray(parent_link.direction, dtype=float)
        flange_offset = flange * parent_link.length
        # Local axes of the flange frame, expressed in the parent link frame.
        open_axis = R_local[:, 0]

        self._w(f'  <joint name="{ee.name}_mount" type="fixed">')
        self._w(f'    <parent link="{parent}"/>')
        self._w(f'    <child link="{ee.name}_base_link"/>')
        self._w(f'    <origin xyz="{_fmt(flange_offset)}" rpy="0 0 0"/>')
        self._w('  </joint>')

        body_mass = ee.mass
        if ee.simulate_fingers:
            body_mass = max(ee.mass - 2.0 * ee.finger_mass, 1e-3)
        body_com = flange * ee.com_offset
        body_centre = flange * (ee.body_length / 2.0)
        self._w(f'  <link name="{ee.name}_base_link">')
        self._box_inertial(body_mass,
                           [ee.body_width, ee.body_height, ee.body_length],
                           body_com, rpy_local)
        self._w('    <visual>')
        self._w(f'      <origin xyz="{_fmt(body_centre)}" rpy="{_fmt(rpy_local)}"/>')
        self._w('      <geometry>')
        self._w(f'        <box size="{ee.body_width:.9g} {ee.body_height:.9g} '
                f'{ee.body_length:.9g}"/>')
        self._w('      </geometry>')
        self._w(f'      <material name="{escape(ee.material.name)}"/>')
        self._w('    </visual>')
        self._w('    <collision>')
        self._w(f'      <origin xyz="{_fmt(body_centre)}" rpy="{_fmt(rpy_local)}"/>')
        self._w('      <geometry>')
        self._w(f'        <box size="{ee.body_width:.9g} {ee.body_height:.9g} '
                f'{ee.body_length:.9g}"/>')
        self._w('      </geometry>')
        self._w('    </collision>')
        self._w('  </link>')
        self._gazebo_link(f'{ee.name}_base_link', mu=1.2)

        if ee.simulate_fingers:
            half = ee.stroke / 2.0
            for sign, side in ((1.0, 'left'), (-1.0, 'right')):
                jname = f'{ee.name}_{side}_joint'
                lname = f'{ee.name}_{side}_finger'
                axis = open_axis * sign
                root = flange * ee.body_length
                self._w(f'  <joint name="{jname}" type="prismatic">')
                self._w(f'    <parent link="{ee.name}_base_link"/>')
                self._w(f'    <child link="{lname}"/>')
                self._w(f'    <origin xyz="{_fmt(root)}" rpy="0 0 0"/>')
                self._w(f'    <axis xyz="{_fmt(axis)}"/>')
                self._w(f'    <limit lower="0" upper="{half:.9g}" '
                        f'effort="{ee.grip_force_max:.9g}" '
                        f'velocity="{ee.grip_speed:.9g}"/>')
                self._w('    <dynamics damping="5.0" friction="1.0"/>')
                self._w('  </joint>')
                centre = (axis * (ee.finger_thickness / 2.0)
                          + flange * (ee.finger_length / 2.0))
                self._w(f'  <link name="{lname}">')
                self._box_inertial(
                    ee.finger_mass,
                    [ee.finger_thickness, ee.finger_width, ee.finger_length],
                    centre, rpy_local)
                for tag in ('visual', 'collision'):
                    self._w(f'    <{tag}>')
                    self._w(f'      <origin xyz="{_fmt(centre)}" '
                            f'rpy="{_fmt(rpy_local)}"/>')
                    self._w('      <geometry>')
                    self._w(f'        <box size="{ee.finger_thickness:.9g} '
                            f'{ee.finger_width:.9g} {ee.finger_length:.9g}"/>')
                    self._w('      </geometry>')
                    if tag == 'visual':
                        self._w('        <material name="__actuator"/>')
                    self._w(f'    </{tag}>')
                self._w('  </link>')
                self._gazebo_link(lname, mu=1.5)

        # A massless frame at the tool centre point, for RViz and for TF users.
        self._w(f'  <joint name="{ee.name}_tcp_joint" type="fixed">')
        self._w(f'    <parent link="{ee.name}_base_link"/>')
        self._w(f'    <child link="tcp_link"/>')
        self._w(f'    <origin xyz="{_fmt(flange * ee.tcp_offset)}" '
                f'rpy="{_fmt(rpy_local)}"/>')
        self._w('  </joint>')
        self._w('  <link name="tcp_link">')
        self._w('    <inertial><mass value="1e-6"/>'
                '<inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>'
                '</inertial>')
        self._w('    <visual><geometry><sphere radius="0.008"/></geometry>'
                '<material name="__tcp"/></visual>')
        self._w('  </link>')

        if self.payload_mass > 0.0:
            side = (self.payload_mass / 2000.0) ** (1.0 / 3.0) * 2.0  # ~ a dense block
            side = max(side, 0.04)
            self._w('  <joint name="payload_joint" type="fixed">')
            self._w('    <parent link="tcp_link"/>')
            self._w('    <child link="payload_link"/>')
            self._w('    <origin xyz="0 0 0" rpy="0 0 0"/>')
            self._w('  </joint>')
            self._w('  <link name="payload_link">')
            self._box_inertial(self.payload_mass, [side] * 3, [0, 0, 0], [0, 0, 0])
            self._w('    <visual><geometry>'
                    f'<box size="{side:.9g} {side:.9g} {side:.9g}"/>'
                    '</geometry><material name="__payload"/></visual>')
            self._w('    <collision><geometry>'
                    f'<box size="{side:.9g} {side:.9g} {side:.9g}"/>'
                    '</geometry></collision>')
            self._w('  </link>')
            self._gazebo_link('payload_link', mu=1.5)

    # ------------------------------------------------------- ros2_control
    def _build_ros2_control(self) -> None:
        cfg = self.cfg
        ci = self.command_interface
        self._w('  <ros2_control name="GazeboSimSystem" type="system">')
        self._w('    <hardware>')
        self._w('      <plugin>gz_ros2_control/GazeboSimSystem</plugin>')
        self._w('    </hardware>')
        for i, joint in enumerate(cfg.joints):
            self._w(f'    <joint name="{escape(joint.name)}">')
            if ci == 'position':
                self._w('      <command_interface name="position">')
                self._w(f'        <param name="min">{joint.lower:.9g}</param>')
                self._w(f'        <param name="max">{joint.upper:.9g}</param>')
                self._w('      </command_interface>')
            elif ci == 'effort':
                self._w('      <command_interface name="effort">')
                self._w(f'        <param name="min">{-joint.effort_limit:.9g}</param>')
                self._w(f'        <param name="max">{joint.effort_limit:.9g}</param>')
                self._w('      </command_interface>')
            else:
                self._w('      <command_interface name="velocity">')
                self._w(f'        <param name="min">{-joint.usable_speed:.9g}</param>')
                self._w(f'        <param name="max">{joint.usable_speed:.9g}</param>')
                self._w('      </command_interface>')
            self._w('      <state_interface name="position">')
            self._w(f'        <param name="initial_value">{self.q0[i]:.9g}</param>')
            self._w('      </state_interface>')
            self._w('      <state_interface name="velocity"/>')
            self._w('      <state_interface name="effort"/>')
            self._w('    </joint>')
        for jname in cfg.end_effector.finger_joint_names:
            self._w(f'    <joint name="{escape(jname)}">')
            self._w('      <command_interface name="position">')
            self._w('        <param name="min">0.0</param>')
            self._w(f'        <param name="max">{cfg.end_effector.stroke / 2.0:.9g}</param>')
            self._w('      </command_interface>')
            self._w('      <state_interface name="position">')
            self._w('        <param name="initial_value">0.0</param>')
            self._w('      </state_interface>')
            self._w('      <state_interface name="velocity"/>')
            self._w('      <state_interface name="effort"/>')
            self._w('    </joint>')
        self._w('  </ros2_control>')

    def _build_gazebo_plugins(self) -> None:
        self._w('  <gazebo>')
        self._w('    <plugin filename="gz_ros2_control-system" '
                'name="gz_ros2_control::GazeboSimROS2ControlPlugin">')
        if self.controllers_file:
            self._w(f'      <parameters>{escape(self.controllers_file)}</parameters>')
        self._w('      <controller_manager_name>controller_manager'
                '</controller_manager_name>')
        self._w('    </plugin>')
        self._w('  </gazebo>')
        self._w('  <gazebo>')
        self._w('    <self_collide>false</self_collide>')
        self._w('  </gazebo>')


def build_urdf(cfg: ArmConfig, **kwargs) -> str:
    return UrdfBuilder(cfg, **kwargs).build()
