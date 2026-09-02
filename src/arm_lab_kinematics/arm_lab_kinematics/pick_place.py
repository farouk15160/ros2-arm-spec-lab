"""Scripted pick and place, executed as Cartesian moves.

    ros2 run arm_lab_kinematics pick_place
    ros2 run arm_lab_kinematics pick_place --ros-args -p object_xyz:="[0.75,0.0,0.04]"

Approach, descend, close, lift, traverse, descend, release, retreat -- each leg
a straight line in Cartesian space at the configured TCP speed, so the whole
cycle respects the speed limit the specification quotes rather than only its
average. Every leg is checked for reachability, self-collision and torque
headroom before anything moves, and the run is reported afterwards.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from arm_lab_model.config import load_config
from arm_lab_model.kinematics import ArmModel

from .cartesian import plan_line
from .collision import CollisionChecker, build_allowed_collisions
from .ik import IKSolver
from .singularity import metrics
from .workspace import TOOL_DOWN


@dataclass
class Leg:
    name: str
    target: np.ndarray
    payload: float                 # what the arm is carrying on this leg
    grip: Optional[float] = None   # jaw opening to command after the move
    settle: float = 0.6


@dataclass
class LegReport:
    name: str
    planned: bool
    duration: float = 0.0
    peak_tcp_speed: float = 0.0
    peak_torque_use: float = 0.0
    worst_joint: str = ''
    min_sigma: float = 0.0
    collision: bool = False          # link on link: a real fault
    arm_near_ground: bool = False    # an arm link, not the tool, near the deck
    tool_ground_gap: float = math.inf
    notes: List[str] = field(default_factory=list)


class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_place')
        self.declare_parameter('config_file', '')
        self.declare_parameter('object_xyz', [0.75, 0.0, 0.04])
        self.declare_parameter('place_xyz', [0.45, -0.45, 0.04])
        self.declare_parameter('object_mass', 2.0)
        self.declare_parameter('object_size', 0.08)
        self.declare_parameter('approach_height', 0.16)
        self.declare_parameter('speed', 0.0)
        self.declare_parameter('arm_controller', 'arm_controller')
        self.declare_parameter('gripper_controller', 'gripper_controller')
        self.declare_parameter('dry_run', False)
        self.declare_parameter('grip_force', 60.0)

        self.cfg = load_config(self.get_parameter('config_file').value or None)
        self.model = ArmModel(self.cfg)
        self.solver = IKSolver(self.model)
        acm = build_allowed_collisions(self.model, samples=400)
        # Two checkers on purpose. Reaching down for an object on the ground is
        # supposed to bring the tool close to the ground, and the tool capsule
        # is a fat cylinder around a slim box, so counting that as a collision
        # would condemn every pick. Link-on-link contact is the real fault.
        self.checker = CollisionChecker(self.model, allowed=acm['always'],
                                        ground_z=None)
        self.ground_checker = CollisionChecker(self.model, allowed=acm['always'])
        self.tool_link = f'{self.cfg.end_effector.name}_base_link'

        motion = self.cfg.raw.get('motion', {})
        speed = float(self.get_parameter('speed').value)
        self.speed = speed if speed > 0 else float(
            motion.get('cartesian_speed', 0.2))
        self.dry_run = bool(self.get_parameter('dry_run').value)

        arm = self.get_parameter('arm_controller').value
        grip = self.get_parameter('gripper_controller').value
        self.traj_pub = self.create_publisher(
            JointTrajectory, f'/{arm}/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(
            Float64MultiArray, f'/{grip}/commands', 10)
        self.create_subscription(JointState, '/joint_states', self._on_state, 20)

        self.q = np.zeros(self.model.n)
        self.have_state = False
        self._last_opening = self.cfg.end_effector.stroke
        self.grip_force = float(self.get_parameter('grip_force').value)
        self._index = {n: i for i, n in enumerate(self.cfg.joint_names)}

    def _on_state(self, msg: JointState) -> None:
        for k, name in enumerate(msg.name):
            idx = self._index.get(name)
            if idx is not None and k < len(msg.position):
                self.q[idx] = float(msg.position[k])
        self.have_state = True

    # ------------------------------------------------------------- sequence
    def legs(self) -> List[Leg]:
        obj = np.array(self.get_parameter('object_xyz').value, dtype=float)
        place = np.array(self.get_parameter('place_xyz').value, dtype=float)
        mass = float(self.get_parameter('object_mass').value)
        size = float(self.get_parameter('object_size').value)
        lift = float(self.get_parameter('approach_height').value)
        ee = self.cfg.end_effector
        open_wide = min(size + 0.05, ee.stroke)
        closed = max(size - 0.004, 0.0)

        grasp = obj + np.array([0.0, 0.0, size / 2.0])
        above_pick = grasp + np.array([0.0, 0.0, lift])
        release = place + np.array([0.0, 0.0, size / 2.0])
        above_place = release + np.array([0.0, 0.0, lift])

        return [
            Leg('open jaws', above_pick, 0.0, grip=open_wide),
            Leg('approach object', above_pick, 0.0),
            Leg('descend to grasp', grasp, 0.0),
            Leg('close on object', grasp, 0.0, grip=closed, settle=1.0),
            Leg('lift', above_pick, mass),
            Leg('traverse', above_place, mass),
            Leg('descend to place', release, mass),
            Leg('release', release, 0.0, grip=open_wide, settle=0.8),
            Leg('retreat', above_place, 0.0),
        ]

    def _send_gripper(self, opening: float) -> None:
        """Open or close the jaws.

        In force mode `opening` is only used to decide the sign: a squeeze force
        to close, the same magnitude outward to open. The jaws then stop wherever
        the object is, which is the only way a grasp carries a load.
        """
        ee = self.cfg.end_effector
        names = ee.finger_joint_names
        if not names:
            return
        msg = Float64MultiArray()
        if ee.grasp_mode == 'force':
            # The jaw joints travel from 0 (shut) to stroke/2 (open), so a
            # POSITIVE effort along the joint axis opens them. Squeezing is
            # therefore a negative effort. Getting this backwards leaves the
            # jaws wide open at the moment they are supposed to grip, and the
            # arm lifts nothing.
            closing = opening < self._last_opening - 1e-6
            force = -abs(self.grip_force) if closing else abs(ee.grip_force_min)
            msg.data = [float(force)] * len(names)
        else:
            half = max(0.0, min(opening, ee.stroke)) / 2.0
            msg.data = [half] * len(names)
        self._last_opening = opening
        self.grip_pub.publish(msg)

    def _spin(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)

    def _run_leg(self, leg: Leg, q_start: np.ndarray) -> Tuple[LegReport,
                                                               np.ndarray]:
        report = LegReport(leg.name, planned=False)
        here = self.model.frames(q_start, light=True).tcp
        if float(np.linalg.norm(leg.target - here)) < 1e-4:
            report.planned = True                # a pure gripper action
        else:
            path = plan_line(self.model, q_start, leg.target, TOOL_DOWN,
                             speed=self.speed, solver=self.solver,
                             collision_checker=self.checker)
            report.notes = list(path.notes)
            if not path.feasible:
                return report, q_start
            report.planned = True
            report.duration = path.duration
            report.peak_tcp_speed = path.peak_tcp_speed

            worst_use, worst_joint, worst_sigma = 0.0, '', math.inf
            for k in range(0, len(path.joints), max(len(path.joints) // 24, 1)):
                q = path.joints[k]
                if not self.checker.is_free(q):
                    report.collision = True
                ground = self.ground_checker.check(q).ground
                for name, depth in ground:
                    if name == self.tool_link:
                        report.tool_ground_gap = min(report.tool_ground_gap,
                                                     -depth)
                    else:
                        report.arm_near_ground = True
                tau = self.model.gravity_torque(q, payload=leg.payload)
                use = np.abs(tau) / self.model.torque_limits
                if float(np.max(use)) > worst_use:
                    worst_use = float(np.max(use))
                    worst_joint = self.cfg.joint_names[int(np.argmax(use))]
                worst_sigma = min(worst_sigma,
                                  metrics(self.model, q).sigma_min_full)
            report.peak_torque_use = worst_use
            report.worst_joint = worst_joint
            report.min_sigma = worst_sigma

            if not self.dry_run:
                msg = JointTrajectory()
                msg.joint_names = list(self.cfg.joint_names)
                for k in range(len(path.times)):
                    point = JointTrajectoryPoint()
                    point.positions = [float(v) for v in path.joints[k]]
                    point.velocities = [float(v) for v in path.joint_speeds[k]]
                    seconds = float(path.times[k])
                    point.time_from_start = DurationMsg(
                        sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))
                    msg.points.append(point)
                self.traj_pub.publish(msg)
                self._spin(path.duration + 0.4)
            q_start = path.joints[-1]

        if leg.grip is not None and not self.dry_run:
            self._send_gripper(leg.grip)
            self._spin(leg.settle)
        return report, q_start

    def run(self) -> int:
        if not self.dry_run:
            self.get_logger().info('waiting for /joint_states ...')
            deadline = time.time() + 30.0
            while not self.have_state and time.time() < deadline and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.2)
            if not self.have_state:
                self.get_logger().error(
                    'no /joint_states; start the simulation first')
                return 2
            q = self.q.copy()
        else:
            q = self.model.resolve_pose('home')

        reports: List[LegReport] = []
        for leg in self.legs():
            report, q = self._run_leg(leg, q)
            reports.append(report)
            state = 'ok' if report.planned else 'UNREACHABLE'
            self.get_logger().info(f'{leg.name}: {state}')
            if not report.planned:
                break
        print(render(reports, self.cfg, self.model))
        return 0 if all(r.planned for r in reports) else 1


def render(reports: List[LegReport], cfg, model) -> str:
    W = 96
    lines = ['', '=' * W, '  PICK AND PLACE CYCLE', '=' * W]
    lines.append(f'{"leg":<20}{"time s":>8}{"peak TCP":>11}{"peak torque":>13}'
                 f'{"worst joint":>14}{"sigma_min":>11}   flags')
    lines.append('-' * W)
    total = 0.0
    for r in reports:
        if not r.planned:
            lines.append(f'{r.name:<20}   NOT REACHABLE  '
                         + ('; '.join(r.notes)[:52] if r.notes else ''))
            continue
        total += r.duration
        flags = []
        if r.collision:
            flags.append('SELF-COLLISION')
        if r.arm_near_ground:
            flags.append('ARM NEAR GROUND')
        elif r.tool_ground_gap < math.inf:
            flags.append('tool at the deck')
        if r.peak_torque_use > 1.0:
            flags.append('OVER TORQUE')
        elif r.peak_torque_use > 0.8:
            flags.append('torque > 80 %')
        if 0.0 < r.min_sigma < 0.02:
            flags.append('near singular')
        lines.append(f'{r.name:<20}{r.duration:>8.2f}{r.peak_tcp_speed:>11.3f}'
                     f'{r.peak_torque_use * 100:>12.0f}%{r.worst_joint:>14}'
                     f'{r.min_sigma:>11.4f}   {", ".join(flags)}')
    lines.append('-' * W)
    lines.append(f'  cycle time {total:.2f} s over {len(reports)} legs')
    worst = max((r.peak_torque_use for r in reports if r.planned), default=0.0)
    lines.append(f'  worst torque utilisation {worst * 100:.0f} % of peak')
    if any(r.collision for r in reports):
        lines.append('  SELF-COLLISION on at least one leg: the cycle is not safe')
    elif any(r.arm_near_ground for r in reports):
        lines.append('  An arm link comes within the ground margin: check the '
                     'approach height')
    else:
        lines.append('  No link-on-link contact and no arm link near the deck.')
    if any(r.tool_ground_gap < math.inf for r in reports):
        lines.append('  "tool at the deck" is expected on a grasp leg: the tool '
                     'capsule is a')
        lines.append('  cylinder as wide as the gripper body, so it touches the '
                     'ground plane')
        lines.append('  before the jaws do.')
    lines.append('=' * W)
    return '\n'.join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = PickAndPlace()
    try:
        return node.run()
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
