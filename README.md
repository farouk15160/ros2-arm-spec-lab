# Rover arm test bench (ROS 2 Jazzy + Gazebo Harmonic)

A workspace for answering one question: **is this arm specification physically
realistic?**

You describe the arm in a single YAML file — link lengths, tube diameters and
wall thicknesses, materials, actuators and gear ratios, end-effector mass — and
the workspace generates the URDF, the Gazebo model, the ros2_control setup, a
live dashboard and a written spec report from it. Change a number, relaunch,
see what it costs you.

```
src/
  arm_lab_model/     the model: config schema, kinematics, inverse dynamics,
                     deflection, URDF + controller generation, spec report, sweeps
  arm_lab_bringup/   launch files, Gazebo world, RViz config
  arm_lab_gui/       Qt capability dashboard + headless analysis nodes
```

---

## Build

```bash
cd ros2_robot_arm_calc_
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
# Full bench: Gazebo + controllers + RViz + dashboard
ros2 launch arm_lab_bringup sim.launch.py

# A heavier tool and a 2 kg load welded to the TCP
ros2 launch arm_lab_bringup sim.launch.py ee_mass:=1.4 payload_mass:=2.0

# A different arm entirely
ros2 launch arm_lab_bringup sim.launch.py config_file:=/path/to/my_variant.yaml

# On Mars, no Gazebo window
ros2 launch arm_lab_bringup sim.launch.py gravity:=3.72 gz_gui:=false

# Geometry check only, no physics: RViz + joint sliders
ros2 launch arm_lab_bringup view.launch.py
```

### Launch arguments

| argument | default | meaning |
|---|---|---|
| `config_file` | installed `arm_config.yaml` | the arm to build |
| `ee_mass` | from config | end-effector mass override, kg |
| `payload_mass` | `0.0` | rigid test mass welded at the TCP, kg |
| `gravity` | from config | m/s², also rewrites the world |
| `command_interface` | `velocity` | `position`, `velocity` or `effort` |
| `initial_pose` | `home` | any name from `test_poses` |
| `world` | test bench | world SDF |
| `gz_gui` / `rviz` / `dashboard` / `capability` | `true` | which windows and nodes to start |

---

## The config file

`src/arm_lab_model/config/arm_config.yaml` is the only file you edit. Copy it,
change it, pass it with `config_file:=`.

Each entry in `joints:` carries its own actuator and its own hollow-tube link:

```yaml
- name: joint_2                     # shoulder pitch
  type: revolute
  origin_xyz: [0.0, 0.0, 0.0]       # from the previous link's far end
  origin_rpy: [-1.5708, 0.0, 0.0]
  axis: [0.0, 0.0, 1.0]
  actuator: big_shoulder            # from the actuators: library
  limits: {lower: -2.10, upper: 2.10, velocity: 1.20}
  link:
    name: link_2
    length: 0.450                   # metres
    outer_diameter: 0.080
    wall_thickness: 0.0025          # hollow cylinder
    material: cfrp_tube             # from the materials: library
    direction: [1.0, 0.0, 0.0]      # which way the tube runs
    extra_mass: 0.18                # fittings, lumped at the joint
```

**Frame convention.** Start at the previous link's far end, translate by
`origin_xyz`, rotate by `origin_rpy` — that is the joint frame. The joint turns
about `axis`. The tube runs `length` metres along `direction`, and its far end
is where the next joint starts. So lengthening a link moves everything after it
automatically; reach, mass, inertia and gravity torque all follow.

**What is derived, not typed.** Link mass comes from
`density × π(r_o² − r_i²) × length`. The inertia tensor is the hollow-cylinder
tensor plus a parallel-axis term for the lumped actuator mass. The joint effort
limit is `motor peak torque × gear ratio × efficiency`. You never enter a mass
or an inertia by hand, so the model stays consistent with the geometry.

Other sections: `materials` (density, Young's modulus, yield strength),
`actuators` (torque, ratio, efficiency, rotor inertia, joint stiffness,
friction, bus voltage), `end_effector`, `control`, `can_bus`, `spec_targets`
(what the report grades against) and `test_poses`.

---

## The tools

### Spec report — the main event

```bash
ros2 run arm_lab_model spec_report
ros2 run arm_lab_model spec_report --ee-mass 1.4 --payload 3.0
ros2 run arm_lab_model spec_report --gravity 3.72 --json
```

Grades every row of the requirements table: DOF, reach, payload at full reach
and at 700 mm, arm mass, end-effector allowance, peak and continuous joint
torque, TCP speed, positioning accuracy, orientation accuracy, contact force,
gripper, power draw per bus, control rate and CAN bus load. Each line states the
requirement, what the configured arm achieves, and where the number came from.
Exit code is non-zero if anything fails, so it drops straight into CI.

### Parameter sweep

```bash
ros2 run arm_lab_model sweep --param joints.1.link.length --values 0.35:0.55:5
ros2 run arm_lab_model sweep --param joints.1.link.wall_thickness --values 0.0015,0.0025,0.004
ros2 run arm_lab_model sweep --param end_effector.mass --values 0.5,1.0,1.5,2.0
ros2 run arm_lab_model sweep --param environment.gravity --values 1.62,3.72,9.81
```

One parameter, a range of values, and the effect on reach, mass, payload
capacity, droop, peak torque utilisation and TCP speed side by side.

### Dashboard

Comes up with `sim.launch.py`, or attach it to a running sim:

```bash
ros2 launch arm_lab_bringup dashboard.launch.py
```

Shows, live: per-joint speed and torque against their limits, TCP position,
reach, TCP speed, **payload capacity at the current pose** with the joint that
limits it, droop under load split into tube bending and gearbox wind-up,
reachable contact force, power per bus, tube stress, and a scorecard of live
values against the spec. The buttons drive the trajectory controller so the
numbers move under real motion; `Full spec report` opens the written report.

The **payload** spinner is analysis only — it changes what the model assumes is
in the gripper. To make Gazebo physically carry a mass, relaunch with
`payload_mass:=`.

### Headless speed test

```bash
ros2 run arm_lab_gui speed_test --ros-args -p target_speed:=0.4 -p cycles:=6
```

Drives the arm back and forth at a commanded TCP speed and reports what it
actually achieved, plus the peak torque per joint. Answers "does the speed
figure survive the controller, or only the kinematics?"

### Capability topics

`capability_node` publishes `/arm_lab/tcp_speed`, `/arm_lab/payload_capacity`,
`/arm_lab/torque_utilisation`, `/arm_lab/tcp_droop`, `/arm_lab/reach`,
`/arm_lab/power`, `/arm_lab/joint_torque_model` and a `/diagnostics` status —
plot them with `rqt_plot` or record them with `ros2 bag`.

### Generators

```bash
ros2 run arm_lab_model urdf_gen -o /tmp/arm.urdf
ros2 run arm_lab_model controllers_gen
```

Launching also writes both to `/tmp/arm_lab_generated/` so you can inspect
exactly what was fed to Gazebo.

---

## How the numbers are computed

- **Forward kinematics / Jacobian** — world-frame recursion over the chain.
- **Joint torque** — recursive Newton-Euler with gravity, Coriolis and inertial
  terms, plus reflected rotor inertia (`J_rotor × ratio²`) and joint friction.
- **Payload capacity at the TCP** — gravity torque is affine in the payload
  mass, so `τᵢ(m) = Aᵢ + m·Bᵢ` is solved exactly against each joint's limit and
  the smallest bound wins. No search, and it names the limiting joint.
- **Max TCP speed** — the reachable velocity set is a zonotope, so the extreme
  is a vertex of the joint-velocity box; all 2ⁿ vertices are enumerated.
- **Droop** — unit-load (Castigliano) integration along each tube for bending
  and torsion, plus `τᵢ/kᵢ` elastic wind-up at each gearbox. On a geared arm the
  gearboxes usually dominate, which is exactly the point.
- **Contact force** — `τ = Jᵀ F` with the gravity-holding torque subtracted
  first, so it is the force actually available after the arm holds itself up.

### What the model does not cover

Backlash, encoder resolution, thermal derating, harness drag, control loop
jitter, and vision-loop error. The droop figure is a lower bound on positioning
error, not the whole budget. Repeatability in particular is not simulated — the
report says so rather than inventing a number.

### Torque shown in the dashboard

The bar is the model torque computed from the measured motion; the dashed line
is the effort the simulator reports. They differ because what `gz_ros2_control`
reports as effort depends on the command interface — in velocity and position
modes it is often the commanded value, not a constraint force. Trust the bar for
sizing decisions.

---

## Command interfaces

`velocity` (default) is a velocity motor limited by the joint effort limit:
stable, respects torque limits, good for speed and reach work. `effort` gives
the trajectory controller direct torque authority — the most honest test of a
torque budget, and the most likely to need gain tuning (`control.gains` in the
config). `position` is the least physical; use it for visualisation only.

---

## What the baseline configuration says about the spec

The shipped config is a plausible starting point, not a recommendation. Running
the report on it as-is gives **15 pass, 1 warn, 0 fail** — so the requirements
table is broadly self-consistent. Three results are worth arguing about.

**Accuracy is the binding constraint, and it is a gearbox problem.** The arm
droops **6.9 mm at full reach under 2 kg**, against a 10 mm raw accuracy target.
Only 0.1 mm of that is the tubes bending — **6.8 mm is elastic wind-up in the
gearboxes**. Thicker or stiffer tubes buy almost nothing here; stiffer output
stages, or a stiffness feed-forward correction, buy nearly everything. And
6.9 mm of a 10 mm budget is spent before backlash, encoder resolution or
calibration error are counted, so the raw figure is tight rather than
comfortable.

**Continuous torque, not peak torque, sizes the actuators.** Peak demand is only
55 % of the shoulder's peak rating, but it exceeds the *continuous* rating on
joints 1, 2 and 5 at the worst-case pose. That is fine for a burst duty cycle
and a problem for a hold-at-full-reach one — which is exactly the duty cycle a
sampling arm has. Thermal behaviour deserves a bench test; the simulator cannot
give you one.

**The 0.20 m/s TCP limit is a policy, not a hardware limit.** The joint speed
limits allow about 2.9 m/s at full reach, fourteen times the spec. `speed_test`
also shows that commanding a 0.20 m/s *average* through the trajectory
controller peaks near 0.38 m/s, because a joint-space spline peaks above its own
mean and the joint-space path is longer than the straight line between the
endpoints. If 0.20 m/s is a safety requirement it needs a Cartesian speed
monitor; trajectory timing will not deliver it.

Two more results that were never in doubt but are now quantified: payload
capacity is **3.94 kg at full reach** and **7.19 kg at 700 mm** against 2.0 and
3.0 kg targets, so there is real margin; and the CAN bus sits at **36 % load**
at 200 Hz, so the 1 Mbit/s classical bus can stay.

---

Copyright 2026 farouk15160. Licensed under the Apache License, Version 2.0.
