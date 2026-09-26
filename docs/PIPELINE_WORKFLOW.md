# Build, simulate, plan and benchmark a robot project

[Documentation index](README.md) · [Configuration contracts](EXTENDED_PIPELINE.md)

## Build the shared physical model

From the repository root, use the development environment described in
[getting started](GETTING_STARTED.md). The pipeline uses NumPy, PyYAML and
MuJoCo; report plots use Matplotlib, now included in `requirements-dev.txt`.
Source execution avoids needing to reinstall entry points after edits:

```bash
export PYTHONPATH="$PWD/src/arm_lab_model:$PWD/src/arm_lab_kinematics:$PWD/src/arm_lab_gui:$PYTHONPATH"
python3 -m arm_lab_model.project_config src/arm_lab_model/config/pipeline/project_ur5e.yaml
python3 -m arm_lab_model.pipeline_cli build \
  src/arm_lab_model/config/pipeline/project_ur5e.yaml --output /tmp/ur5e-build
python3 -m arm_lab_model.pipeline_cli analyze \
  src/arm_lab_model/config/pipeline/project_ur5e.yaml --output /tmp/ur5e-analysis
python3 -m arm_lab_model.pipeline_cli reference-check \
  src/arm_lab_model/config/pipeline/project_ur5e.yaml --output /tmp/ur5e-reference
```

Pip-installed equivalents are `project_check` and `robot_pipeline`. After a
colcon build, use `ros2 run arm_lab_model project_check` and
`ros2 run arm_lab_model robot_pipeline`. Build writes
`robot.resolved.json`, `robot.urdf`, `robot.xml`, `physics.json`, `manifest.json`
and, when enabled, generated `moveit/` configuration. Analyze reports link mass,
COM, inertia, weight, total mass/COM and static effort. The default analysis
pose is zero joint coordinates; use `physical_report` for a specified pose/payload.
Resolve errors identify missing physical inputs instead of guessing them.
`reference-check` writes `reference_check.json` with catalogue-linked nominal
kinematics checks and `hardware_validation: false`; it does not fabricate timed
measurements from static reference cases.

For your own robot, copy a tree and project manifest, reference STL links with
explicit units/materials or complete manual assembly inertials, and configure
all dynamic limits needed for the planned experiment. See
[STL and physical analysis](PHYSICAL_ROBOT.md). Additional robot types use the
same topology contract; the [twelve-actuator dog](DOG12_DEMO.md) is a runnable
synthetic example with a floating root.

## Execute, record and replay

```bash
python3 -m arm_lab_model.pipeline_cli simulate \
  src/arm_lab_model/config/pipeline/project_ur5e.yaml \
  --scenario src/arm_lab_model/config/pipeline/scenario_ur5e.yaml \
  --benchmark --benchmark-robot ur5e --save --output /tmp/ur5e-motion
```

The scenario names ordered joints and timestamped position/velocity/acceleration
points. It supplies explicit experiment acceleration settings, not manufacturer
ratings. The controller interpolates and follows the trajectory within configured
limits. Headless execution initializes a fresh simulation from the scenario start;
this is separate from ROS execution, which validates the current robot state.

`execution.json` reports simulation acceptance; `observation.json` stores sampled
channels. Successful execution with `--save` writes a uniquely named record under the
configured `trajectories.directory`, resolved relative to that component YAML.
The bundled UR5e project selects repository-root `saved_trajectories/ur5e/`.
Without a configured directory the CLI uses `<output>/saved_trajectories/ur5e/`.
Use the returned `saved_trajectory` path:

```bash
python3 -m arm_lab_model.pipeline_cli replay \
  src/arm_lab_model/config/pipeline/project_ur5e.yaml \
  --trajectory /absolute/path/to/saved_trajectory.yaml --output /tmp/ur5e-replay
```

Replay checks model identity and joint order. A saved success does not waive
current collision, state, interpolation and dynamic-limit validation. See
[trajectory records](TRAJECTORIES_BENCHMARKS.md) for their fields and semantics.

The bundled UR5e project enables benchmarking but has no measured trajectory
reference. Consequently, the command above normally exits **1** with simulation
`passed: true` and benchmark `status: incomplete`. This explicitly separates
successful simulated motion from missing hardware-comparison evidence.

## Supply independent reference observations

A compatible reference must match robot, scenario, joint order, frame, units and
end effector, with time expressed on the same experiment-relative clock. Describe
measurement provenance and acquisition conditions. Do not present simulated or
analytically generated samples as hardware recordings.

```bash
python3 -m arm_lab_model.pipeline_cli simulate \
  src/arm_lab_model/config/pipeline/project_ur5e.yaml \
  --scenario src/arm_lab_model/config/pipeline/scenario_ur5e.yaml \
  --reference /absolute/path/to/reference.yaml --output /tmp/ur5e-comparison

# Or compare independently recorded compatible files:
python3 -m arm_lab_model.pipeline_cli compare simulation.json reference.yaml \
  --tolerances tolerances.yaml --output /tmp/comparison
```

The standalone tolerance file is a metric-to-nonnegative-threshold YAML mapping
(zero requests exact agreement),
for example `{joint_position: 0.02, tcp_position: 0.005, duration: 0.05}`.
Those are illustrative acceptance settings, not UR5e accuracy specifications.
Channels without thresholds are unassessed; missing requested channels and
partial overlap yield incomplete results. Reports include `benchmark.json`,
`benchmark.md` and available comparison/error/TCP figures. Without Matplotlib,
numerical reports remain available and explain omitted plots.

CLI exits are **0** for accepted operations/comparisons, **1** for failed or
incomplete acceptance, and **2** for invalid inputs or unavailable dependencies.
Use a fresh report directory per run: report filenames are replaced in an
existing directory. Unique saved trajectory records are not overwritten.

## MoveIt, RViz and perception

Build/source the ROS workspace, then launch the same project. The integrated
path has been exercised with ROS 2 Humble and its installed MoveIt interfaces:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch arm_lab_bringup pipeline.launch.py \
  project_file:="$PWD/src/arm_lab_model/config/pipeline/project_ur5e.yaml" \
  benchmark:=true benchmark_robot:=ur5e output_dir:=/tmp/ur5e-ros-reports
```

RViz's Publish Point requests position-only planning. Use the MotionPlanning
panel for pose-marker interaction, or the explicit target/preview/save workflow:

```bash
ros2 run arm_lab_kinematics pipeline_target target --frame base_link --xyz 0.35 0.15 0.40
ros2 topic echo /arm_lab/planning_status
ros2 service call /arm_lab/execute_plan std_srvs/srv/Trigger '{}'
ros2 service call /arm_lab/save_trajectory std_srvs/srv/Trigger '{}'
```

An example target is not guaranteed reachable from every configuration. Add
`--rpy R P Y` for an orientation constraint. Wait for preview before execution
and confirmed successful execution before Save. Invalid/new targets clear old
previews. Native RViz panel executions are separate from the target node's last
plan and cannot be saved by its Save service. See [planning](PIPELINE_PLANNING.md)
for exact topics, status semantics and the opt-in live test.

`project_ur5e_perception.yaml` selects enabled environment/camera/perception
examples. Launch it instead to render sensor data and build filtered clouds;
headless rendering may require `MUJOCO_GL=egl`. Inspect `/sensors/<name>/...`,
`/perception/points` and `/arm_lab/scene_status`. The scene adapter applies objects
before this project's target node accepts motion. See [scene and perception](SCENE_PERCEPTION.md)
for camera calibration, OctoMap plugin requirements and external SLAM/recognition/
policy adapters. A configured algorithm interface is not a bundled trained model.

## Verification and evidence

For the subsequent multi-robot run profiles, simulation-only launch and workspace
agent guidance, see [the current run guide](RUN_ROBOTS.md). It records the latest
Python 3.12 pre-publish checks, ROS suite and separate live 1-DOF dog action test.
The counts and coverage below record the original twelve-stage implementation.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
python3 tools/build_docs.py --check
```

The final sourced-ROS suite passed **399 tests**, with one skipped opt-in live
check. That live MoveIt check passed separately, and `colcon` built all four
packages. A later immutable-policy regression was verified by rerunning the
34-test runtime/interpolation suite successfully; the 399 count predates that
one additional test. One local Matplotlib Axes3D warning did not prevent the generated 2D
comparison plots. These counts describe this implementation validation run.

Combined unit and live MoveIt execution covered **81.26% of statements in the
25 new pipeline modules** (2,714/3,340). This is not whole-repository coverage:
the broader repository's unit-only measurement was about 54%, including legacy
GUI code not exercised by that run. Camera/perception live behavior and EGL
near-clip regression checks passed separately; they are not additional coverage
claimed in the 81.26% figure.

ROS adapter tests require ROS sourced; rendering tests require a usable OpenGL
backend. Missing optional dependencies are explicit skips, not evidence of success.
For live MoveIt validation launch the UR5e pipeline first, then use the opt-in
command in [the planning guide](PIPELINE_PLANNING.md); it moves the simulated
robot and saves an execution record.

The integrated checks include five nominal UR5e FK cases agreeing to approximately
`1e-10` m, a small simulated trajectory with maximum joint tracking error about
`3.0505e-6` rad and no contacts/saturation, and actual ROS Humble pose planning,
preview, execution, saving, position-only planning and unreachable-target rejection.
An additional analytical motion-reference comparison passed with maximum joint
error `3.0505e-6` rad and TCP error `1.6877e-6` m, producing comparison plots.
These are configuration-specific numerical/simulation checks, not hardware tests.
The headless physics CI also runs UR5e `reference-check` and uploads its JSON
alongside the existing physics reports.
No measured UR5e motion data are bundled, and the known manufacturer/model mass
discrepancy remains explicitly reported. The dog fixture verifies contacts and
finite state but deliberately fails generic collision-free trajectory acceptance.

The current limits include approximate UR5e collision shapes, MuJoCo convex-hull
STL contacts, ideal pinhole cameras, rejection of dynamic planning-scene objects
until live pose updates exist,
external SLAM/learned algorithms, and no quadruped balance/gait controller.

## Implementation reports by stage

Paths below are within `src/arm_lab_model/arm_lab_model/` unless a package is
named. Each linked guide gives configuration examples and detailed test commands.
Test filenames are collected by the full pytest command above; ROS/live checks
need the environments explicitly described in their guides.

| Step | What was implemented | Main files added/modified | How it works | How to test | Remaining limits |
|---|---|---|---|---|---|
| 1 | Versioned project/config architecture | `project_config.py`, `config_contract.py`, `robot_topology.py`, `config/pipeline/` | Immutable strict composition and explicit trees | `test_project_config.py`, `project_check PATH` | Validation is not runtime readiness; [contracts](EXTENDED_PIPELINE.md) |
| 2 | UR5e benchmark robot/reference catalogue | `benchmark_reference.py`, `config/benchmarks/`, `robot_ur5e.yaml` | Pinned sources, unknown measurements, five independent nominal FK cases | `test_benchmark_reference.py` | No recorded hardware trajectory; [provenance](UR5E_REFERENCE.md) |
| 3 | STL/material physical analysis | `mesh_physics.py`, `physical_robot.py` | Closed-solid integration or complete manual override; tree statics | `test_mesh_physics.py`, `test_physical_robot.py` | Uniform material and validated solid required; [physics](PHYSICAL_ROBOT.md) |
| 4 | Unified description/export | `robot_export.py`, `robot_legacy.py` | One resolved model generates URDF/MJCF | `test_robot_export.py` | STL contacts use convex hulls; [exports](PHYSICAL_ROBOT.md) |
| 5 | MoveIt and controlled simulation | `arm_lab_kinematics/pipeline_moveit.py`, `arm_lab_bringup/launch/pipeline.launch.py`, `pipeline_runtime.py` | Generated groups/configs and MuJoCo trajectory action bridge | `test_pipeline_planning.py`, `test_pipeline_runtime.py`, live ROS launch | Chain solvers and MuJoCo runtime; [planning](PIPELINE_PLANNING.md) |
| 6 | RViz position/pose targeting | `arm_lab_kinematics/pipeline_target.py`, `arm_lab_bringup/rviz/pipeline.rviz` | TF-aware target, preview, ExecuteTrajectory and status lifecycle | Planning tests plus opt-in `-k live` | No hardware execution; [interaction](PIPELINE_PLANNING.md) |
| 7 | Save/replay trajectories | `trajectory_store.py`, `joint_trajectory.py`, `pipeline_cli.py` | Validated samples, model identity, provenance and successful execution | `test_trajectory_store.py`, CLI replay | Planned derivatives may be derived and labeled; [storage](TRAJECTORIES_BENCHMARKS.md) |
| 8 | Configurable environment | `scene_config.py`, `arm_lab_gui/pipeline_scene.py` | Shared geometry/poses become physics objects and MoveIt scene | `test_scene_perception.py`, ROS `test_pipeline_perception.py` | Dynamic planning objects rejected pending live updates; [scene](SCENE_PERCEPTION.md) |
| 9 | Mounted configurable cameras | `sensor_runtime.py`, exporter/simulation bridge | Calibrated RGB/depth rendering and optical frame publication | Camera projection test in `test_scene_perception.py` | Ideal pinhole sensors; [cameras](SCENE_PERCEPTION.md) |
| 10 | Perception interfaces | `perception.py`, `arm_lab_gui/pipeline_perception.py` | Cloud unprojection/filtering, OctoMap updater config, Python plugin callbacks | Scene/perception tests and enabled ROS example | External SLAM/learned algorithms; [perception](SCENE_PERCEPTION.md) |
| 11 | Benchmark engine | `benchmark_observations.py`, `benchmark_engine.py`, CLI/bridge hooks | Compatible channels/time overlap, rotation-aware errors and explicit thresholds | `test_benchmark_engine.py`, known-error fixtures, CLI compare | Missing reference remains incomplete; [comparison](TRAJECTORIES_BENCHMARKS.md) |
| 12 | Reports and documentation | `benchmark_reports.py`, `docs/`, `tools/build_docs.py` | Evidence-labeled JSON/Markdown/plots and generated API/help | `test_benchmark_reports.py`, `build_docs.py --check` | Plots optional without Matplotlib; [reports](TRAJECTORIES_BENCHMARKS.md) |
