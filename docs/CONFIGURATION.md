# Configuration reference and design conventions

[Documentation index](README.md) · [Every shipped value](CONFIG_VALUES.md)

## Selection and precedence

The [project configuration guide](EXTENDED_PIPELINE.md) covers the general
pipeline, with component-specific examples in the [physical](PHYSICAL_ROBOT.md),
[planning](PIPELINE_PLANNING.md), [scene/perception](SCENE_PERCEPTION.md) and
[benchmark](TRAJECTORIES_BENCHMARKS.md) guides. Project files live under
`src/arm_lab_model/config/pipeline/` and reference one robot plus optional components.
Validate with `project_check PATH`; build/run with `robot_pipeline` or
`pipeline.launch.py project_file:=PATH`. `project_ur5e.yaml` is the integrated arm
example and `project_dog12_demo.yaml` supplies synthetic floating-base physics.
The original `project_quadruped_12dof.yaml` remains incomplete topology-only input.

The checker lists unknown tree inertials/limits and never establishes runtime
readiness. Resolution, export and execution apply their own required-input checks.
Explicit experiment acceleration limits are distinct from verified hardware limits.
Project component values affect pipeline commands, not legacy arm commands.
The remainder of this page documents that legacy format and its precedence.

Use `--config PATH` for offline commands or `config_file:=PATH` for ROS launch.
An explicit path is the clearest way to ensure all tools use the same design.
`load_config` without a path consults `ARM_LAB_CONFIG`, then ROS package lookup,
then a source-tree or Python-prefix fallback. Launch files resolve a blank
`config_file` to an installed package path themselves, so the environment
variable does not override every launch path.

`load_config` accepts `ee_mass`, `payload_mass` and `gravity` overrides.
These update the returned aggregate, not the source YAML. Callers still need to
propagate payload to the specific calculation/export that uses it. Report,
URDF, MuJoCo and ROS command-line spellings differ; see [commands](COMMANDS.md).

## Sections

| YAML section | Purpose and principal consumers |
|---|---|
| `robot` | Name, mount transform, pedestal geometry; config and exporters |
| `environment` | Gravity and joint used as reach origin; analytical model |
| `materials` | Named material library; geometry, mass and structural estimates |
| `actuators` | Named motor/transmission/electrical/thermal library |
| `joints` | Ordered serial rotary chain; one link and actuator reference per entry |
| `end_effector` | Whole-tool mass, shape, offsets, jaws and grip settings |
| `control` | ROS command interface, rate, gains and capability limits/reserves |
| `can_bus` | Bitrate and frame assumptions for communication estimates |
| `mass_model` | Allowances for estimated geometry; complete measured assemblies bypass those mass corrections |
| `structure` | Buckling/fatigue/bearing screening assumptions |
| `realtime` | Loop bandwidth and timing-lag budget |
| `sensing` | Current/force sensing and feedback assumptions |
| `holding` | Brake/encoder assumptions for power-loss analysis |
| `errors` | Synthetic backlash, calibration, dimensional and repeatability errors |
| `motion` | Cartesian planning and joint acceleration settings |
| `spec_targets` | Requirements that reports compare with achieved values |
| `test_poses` | Named joint lists, aliases or automatically generated reach poses |

The generated [field/type reference](API_REFERENCE.md) records dataclass fields
and declared defaults. The shipped YAML has illustrative values that override
many defaults; it is not a verified bill of materials.

## Link and joint inputs

Each joint specifies `name`, `type`, `origin_xyz`, `origin_rpy`, `axis`,
`actuator`, `limits` and `link`. Link fields include `name`, `length`,
`outer_diameter`, `wall_thickness`, `material`, `direction` and `extra_mass`.
Dimensions are metres, masses kilograms, angular limits radians and speed rad/s.
Wall thickness must be positive and smaller than the outer radius.

The arm loader supports `revolute` and `continuous`; prismatic transmissions
need force/linear-speed conventions and are rejected here. Continuous joints
still have a configured lower/upper sampling and benchmark interval. They are
exported as unlimited hinges in MuJoCo, while strict test poses must remain
inside that configured interval. Joint order defines parentage; adding `parent`
keys will not turn this schema into a branched robot.

Joint/link names must be unique within the checked namespaces. Keep identifiers
simple and avoid names reserved by generated world/base/tool elements.

## Mass, centre of mass and full inertia

Place this inside a joint's `link` mapping, or in `robot.pedestal` for a measured
pedestal. All three keys are required together:

```yaml
inertial:
  mass: 2.3
  com: [0.11, -0.02, 0.03]
  inertia: [0.02, 0.03, 0.04, 0.001, -0.002, 0.003]
```

This is the complete assigned moving assembly, including its motor and fittings.
The tensor is about its CoM in link axes, ordered `ixx, iyy, izz, ixy, ixz, iyz`.
For a CAD tensor about another origin, apply the parallel-axis theorem in the
correct direction; rotate into link axes. Some CAD products label products of
inertia using opposite signs to tensor off-diagonal entries. Check conventions.

The editor's **Mass / CoM / inertia…** dialog accepts moving-link values and can
restore the tube estimate. Explicit values remain fixed when dimensions change;
they do not automatically rescale. The collision primitive and stiffness still
come from the tube. There is no mesh/CAD importer in this path.

## Actuator conventions

| Field group | Units / interpretation |
|---|---|
| `peak_torque`, `continuous_torque` | Motor-side N·m before gearing |
| `gear_ratio`, `efficiency` | Positive reduction ratio; gearbox efficiency in `(0,1]` |
| `max_motor_speed` | Motor-side rad/s; divide by ratio for output speed |
| `rotor_inertia` | Motor-side kg·m²; multiply by ratio² for reflected inertia |
| `mass` | Assigned actuator mass in kg in the default moving-link model |
| `friction`, `viscous_damping` | Output Coulomb N·m and viscous N·m·s/rad; damping defaults to zero |
| `joint_stiffness` | Output torsional stiffness, N·m/rad |
| `gearbox_series`, `gearbox_size` | Optional catalog lookup; derived stiffness may replace the typed estimate |
| `bracket_stiffness`, `bearing_stiffness` | Additional output stiffness contributions, combined in series when supplied |
| `torque_constant`, `phase_resistance` | Motor N·m/A and phase-to-phase Ω; must use a consistent current convention |
| `phase_inductance`, `max_phase_current`, `bus_voltage` | H, A, V; not every report/simulator integrates all electrical dynamics |
| `quiescent_power` | Electronics/overhead W |
| Thermal fields | K/W, J/K and °C; lumped thermal assumptions |
| Encoder fields | Resolution bits, motor-side or output-side as named |

Do not enter an already geared actuator's output torque as motor torque and
multiply by its ratio again. Choose a consistent motor-side description or a
unit-ratio output-side equivalent, converting speed, inertia and electrical
parameters consistently. Efficiency is not a complete model of every loss;
avoid double-counting measured losses across efficiency and friction.

## Tool, payload and gripper

`end_effector.mass` includes both jaws. When Gazebo models separate fingers,
their masses are carved from that total and the body CoM is shifted to preserve
the configured assembly CoM. That does not establish identical dynamic inertia
between the rigid analytical tool and moving fingers.

`tcp_offset` and `com_offset` are distances along the final link's direction from
its distal flange. `simulate_fingers`, jaw geometry, `stroke` and `grasp_mode`
configure the Gazebo gripper. Force mode uses an effort controller; other grasp
modes take the position-controller branch. Use the documented intended values
`force` or `position`; the loader does not enforce that enum strictly.

A benchmark payload is extra mass at the TCP, separate from tool mass. Native
MuJoCo uses a tiny positive payload inertia to approximate the analytical point
mass. Gazebo's attached test object is a separate physical approximation.

## Target requirements and test poses

Changing `spec_targets` changes the question the report asks; it does not make
the robot stronger or faster. Reach targets are relative to the configured
`reach_reference_joint`, not necessarily the world origin. Keep desired payload,
speed, accuracy, mass, bus and gripper requirements consistent across experiments.

`test_poses` can contain numeric joint lists, named aliases, `auto_full_reach` or
`auto_reach_<metres>`. Automatic reach poses are solver results and can return
the closest achievable radius. Confirm actual FK rather than assuming the name
guarantees the requested location. MuJoCo simulation rejects invalid explicit
joint lists instead of clipping them.

Source: [loader and validation](../src/arm_lab_model/arm_lab_model/config.py),
[shipped YAML](../src/arm_lab_model/config/arm_config.yaml).
