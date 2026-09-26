# Unified model planning and interactive targets (Steps 5–6)

The versioned project selects one resolved physical robot tree. `pipeline_moveit.py`
derives SRDF chains, per-joint velocity/acceleration bounds, kinematics plugins,
OMPL planning configuration, and FollowJointTrajectory controller bindings from
that tree. `pipeline.launch.py` gives the **same URDF** to robot_state_publisher,
MoveIt, and RViz; the simulation bridge consumes the same resolved project.
When the project omits MoveIt or declares `enabled: false`, the launch starts
robot_state_publisher, MuJoCo, environment visualization and a generic RobotModel
RViz view. It does not load MoveIt configuration or start planning/target nodes.
The direct joint action remains `/arm_controller/follow_joint_trajectory`,
covering all configured actuated joints. Environment markers publish immediately
on `/pipeline/environment_markers` without waiting for a planning-scene service;
scene status explicitly reports `planning_scene_applied: false`.
This mode supports physically complete non-arm models such as a single-joint
dog-leg test rig. The original topology-only 12-DOF example still requires its missing
physical parameters before simulation; the explicitly synthetic `dog12_demo` has
a separate physically complete model. See the [robot run guide](RUN_ROBOTS.md)
for copyable commands and the distinction between these examples.

## Configure MoveIt

A project manifest references `moveit: moveit.yaml`. For example:

```yaml
schema_version: 1
enabled: true
default_group: arm
groups:
  arm:
    joints: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
             wrist_1_joint, wrist_2_joint, wrist_3_joint]
    base_link: base_link
    tip_link: tool0
    kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
    controller: arm_controller
velocity_scaling: 0.1
acceleration_scaling: 0.1
planning_time: 5.0
planner_id: RRTConnectkConfigDefault
position_tolerance: 0.005
orientation_tolerance: 0.01
# Required ONLY where the robot has no established acceleration limits:
# acceleration_limits:
#   shoulder_pan_joint: 0.5  # explicit experiment policy, not manufacturer rating
```

Every configured group must be one actual base-to-tip chain, with moving joints
in chain order. A 12-actuator quadruped can define four leg groups; no six-joint
assumption appears in generation. Group names and controller names are arbitrary
identifiers. Multiple groups may share one controller, in which case its joint
list is their ordered union. A joint cannot have conflicting controllers.

A floating-base robot gets an SRDF floating virtual joint (`world` to the body).
Fixed-base URDFs already contain their world mount. This enables a correct robot
model; it does **not** implement balance, walking, or floating-base control.

Acceleration is never silently invented. Unknown robot acceleration requires an
explicit per-joint `acceleration_limits` policy before generation. A policy may
reduce but never exceed a known robot rating. Velocity limits always come from
the robot. MoveIt time parameterization uses these limits and request scaling.
Only directly adjacent link collisions are disabled; other self-collisions remain
active. Mesh collision fidelity is inherited from the unified model.

## Launch and interact

After building and sourcing the workspace:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch arm_lab_bringup pipeline.launch.py \
  project_file:=/absolute/path/to/project.yaml
```

`rviz:=false` supports unattended execution. `group:=front_left` selects another
configured target group. `benchmark:=true benchmark_robot:=ur5e` enables the
simulation bridge's benchmark workflow; `benchmark_reference:=/path/to/data.yaml`
selects a reference when supported by the benchmark configuration. Reference robot,
joint ordering, scenario, frame, units, joint types and endpoint are validated
at startup before any motion, so an incompatible dataset cannot fail only after execution. Use
`output_dir:=/path/to/reports` for generated benchmark artifacts.

RViz's **Publish Point** tool emits `/clicked_point` and requests position-only
planning. The MotionPlanning panel also supports its native pose marker, group
selector, preview, and execution. To use this project's explicit preview,
execution-result tracking, and Save command, publish targets through the target
node:

```bash
# Position target: orientation remains unconstrained.
ros2 run arm_lab_kinematics pipeline_target target \
  --frame base_link --xyz 0.35 0.15 0.40

# Full pose: fixed-axis roll, pitch, yaw in radians.
ros2 run arm_lab_kinematics pipeline_target target \
  --frame base_link --xyz 0.35 0.15 0.40 --rpy 0 1.5707963268 0

# Observe ready/planning/preview/executing/succeeded/failed events.
ros2 topic echo /arm_lab/planning_status

# Execute the currently previewed plan. This response acknowledges the request;
# the status topic reports the final MoveIt action result.
ros2 service call /arm_lab/execute_plan std_srvs/srv/Trigger '{}'

# Save only after a confirmed successful execution.
ros2 service call /arm_lab/save_trajectory std_srvs/srv/Trigger '{}'
```

Full poses can also be sent as `geometry_msgs/msg/PoseStamped` on `/arm_lab/target`.
When an environment is enabled, this target node waits for confirmed scene application
on `/arm_lab/scene_status` before accepting planning or execution requests. In the
native RViz MotionPlanning panel, wait for scene readiness before planning.
Targets require `header.frame_id`; TF transforms them into the selected group's
base frame at the supplied timestamp (zero means latest). Invalid or unavailable
transforms produce a visible failure. Requests while planning/executing are
rejected as busy. A new target invalidates any old preview, including when the
new target is invalid; a stale preview cannot accidentally execute.

Planning uses MoveGroup with `plan_only=true`. The returned trajectory publishes
to `/display_planned_path`. Execution uses ExecuteTrajectory and checks both ROS
action status and MoveIt error code. No immediate service response is treated as
motion success. MoveIt validates the start state before sending the controller
trajectory; a changed robot start state may require replanning.

## Saved result contract

The Save command writes the shared validated trajectory format, including robot
model digest, actual planned positions/timing, target, planner settings, collision
check outcome and successful execution status. Files default to
`saved_trajectories/<robot>/` relative to the project manifest directory when no storage component is supplied.
Override with `moveit.saved_trajectories_dir` or the trajectory component directory;
relative overrides resolve against the component YAML file declaring them.
`scenario_id:=name` labels both the saved plan and controller benchmark observation.
Floating-base execution saving is explicitly rejected until synchronized base pose
and velocity can be captured; joint-only records cannot replay that initial state.
Absent planner derivatives are numerically differentiated and labeled in planner
metadata; they are never represented as measured derivatives. A benchmark result
is attached only when its exact trajectory digest matches the executed plan.
Otherwise the optional benchmark value is null. These files contain planned
samples and execution outcome; simulation telemetry/reference comparisons live
in the bridge benchmark artifacts.

The native RViz MotionPlanning panel is an independent client. Its executions
still pass through the benchmark-enabled simulation controller, but are not the
target node's last plan and cannot be saved with this target node's Save service.

## Test and verification

```bash
# Pure configuration/lifecycle tests; ROS-message tests skip without ROS.
PYTHONPATH=src/arm_lab_model:src/arm_lab_kinematics \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_kinematics/test/test_pipeline_planning.py

# Include real installed ROS action/message serialization tests.
source /opt/ros/humble/setup.bash
PYTHONPATH="src/arm_lab_model:src/arm_lab_kinematics:$PYTHONPATH" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_kinematics/test/test_pipeline_planning.py
```

Tests cover robot-driven limits and endpoints, refusal to invent acceleration,
invalid chains/options, floating-base semantics, four-group 12-joint generation,
unchanged input configuration, standalone file export, position-only versus pose
constraints using real MoveIt messages, execution/save lifecycle, RPY conversion,
and successful trajectory roundtrips with derivative provenance and benchmark
association. A launch-description import and action-message construction check
also ran against installed ROS 2 Humble. An actual headless ROS 2 Humble run also passed: an FK-derived pose target 15 mm
from the live TCP planned with OMPL, executed through MoveIt's FollowJointTrajectory
controller with final `SUCCEEDED`, and saved a validated trajectory. A second
position-only target planned successfully; an unreachable 100 m target failed and
could not execute the stale preview. This was simulation, not physical UR5e hardware.

To repeat the opt-in live check, launch the UR5e pipeline first, use the same
`ROS_DOMAIN_ID` in both terminals, and run:

```bash
ARM_LAB_LIVE_PIPELINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q src/arm_lab_kinematics/test/test_pipeline_planning.py -k live
```

The live test moves the simulated robot and writes a successful trajectory into
the configured storage directory. It is skipped during ordinary unit test runs.

## Current limits

- The runtime launch currently supports MuJoCo. The generated descriptions and
  standalone MoveIt files are reusable by other integrations.
- KDL groups must be chains. Whole-body non-chain planning requires another group
  representation/solver; quadruped balance and gait control are outside this layer.
- Only RRTConnect is currently exported. Additional planners need validated
  configuration rather than an arbitrary unverified planner string.
- Collision geometry quality, inertia accuracy, calibration and safe joint bounds
  depend on the supplied model. A generic reference UR model is not factory
  calibration of a particular physical robot.
- ROS 2 Humble is the verified parameter/action API. Later MoveIt distributions
  may use different request adapter plugin names.

Sources: [MoveIt Humble MoveGroup workflow](https://moveit.picknik.ai/humble/doc/examples/move_group_interface/move_group_interface_tutorial.html)
and the installed ROS 2 Humble `moveit_msgs` action/message definitions.

## Controller tolerances and success

The FollowJointTrajectory bridge honors per-joint path and goal position,
velocity and acceleration tolerances at the simulation timestep. A tolerance of
zero selects the controller default; -1 disables that component. Unknown joint
names, duplicate declarations and other negative/nonfinite values are rejected.
Defaults are path position `simulation.max_error` (0.05 rad/metres if absent),
goal position 0.02, goal velocity 0.01, and one second of settling. Goal defaults
can be configured as `pipeline_sim` ROS parameters `goal_position_tolerance`,
`goal_velocity_tolerance`, and `goal_time_tolerance`. A positive goal request's
`goal_time_tolerance` overrides settling time. Path velocity/acceleration and goal
acceleration have no default limit; explicitly supply them when needed.

Controller feedback includes desired/actual/error position, velocity, and
acceleration. Contact, actuator saturation, or any path tolerance violation aborts
execution. Success requires all enabled goal tolerances before the settling
deadline. Nonzero terminal velocity is rejected because the runtime holds the
endpoint position. A final acceleration can describe the incoming trajectory
segment (MoveIt time parameterization uses this); after the endpoint the hold
commands zero velocity and acceleration, without a jerk-continuity guarantee.
Effort-command trajectories and multi-DOF controller trajectories are unsupported.

## Generic dog1 runtime verification

The literal one-actuator `project_dog1_demo.yaml` was launched headlessly with
MoveIt absent. The live action test verified that no `move_group` or
`pipeline_target` existed, environment visualization became ready without a
planning service, and `/arm_controller/follow_joint_trajectory` returned SUCCESS
for `front_left_hip_pitch` at 0 → 0.25 → 0 radians over four seconds. Recorded peak
position was 0.249997 rad and final position was 0.00000229 rad. No GUI or physical
robot validation is implied.

With the dog1 launch running in the same ROS domain, repeat with:

```bash
ARM_LAB_LIVE_DOG1=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q src/arm_lab_gui/test/test_pipeline_dog1_live.py
```
