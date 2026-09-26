# Run and test arms and dog configurations

[Documentation index](README.md) · [Instructions for coding agents](../AGENTS.md)

## Select a robot

| Robot | Configuration | Expected behavior |
|---|---|---|
| Original rover arm | `config/arm_config.yaml` with `mujoco_backend` | Existing arm equations, independent physics checks and controlled motion |
| UR5e simulation | `project_ur5e_sim.yaml` | Six-joint motion, save/replay, optional MoveIt/RViz; successful headless run exits 0 |
| UR5e reference benchmark | `project_ur5e.yaml` | Same robot; no measured trace bundled, so motion can pass while comparison is incomplete (exit 1) |
| Literal 1-DOF dog | `project_dog1_demo.yaml` | Fixed torso, one moving front-left leg; successful swing exits 0 |
| 12-DOF dog | `project_dog12_demo.yaml` | Floating torso/four legs; falls onto the floor; contacts intentionally fail collision-free trajectory acceptance (exit 1) |

Project and scenario files are under `src/arm_lab_model/config/pipeline/`.
Both dogs are synthetic examples, with no walking/balance policy or vendor claims.
The [1-DOF dog guide](DOG1_DEMO.md) explains its geometry, limits and physics checks;
[the 12-DOF guide](DOG12_DEMO.md) explains the expected contact-test rejection.

## Headless setup

From the repository root, use an environment with `requirements-dev.txt` installed.
For a fresh Python-only installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Then set source paths; this includes the kinematics package used by model export:

```bash
export PYTHONPATH="$PWD/src/arm_lab_model:$PWD/src/arm_lab_kinematics:$PWD/src/arm_lab_gui${PYTHONPATH:+:$PYTHONPATH}"
export CFG="$PWD/src/arm_lab_model/config/pipeline"
mkdir -p /tmp/arm-lab-runs
```

The commands below use source modules and require no editable package install.
Pip-installed console commands are also available; after a colcon build use
`ros2 run arm_lab_model robot_pipeline` instead of its Python module invocation.

## Original arm

```bash
python3 -m arm_lab_model.mujoco_backend check --samples 20 \
  --output /tmp/arm-lab-runs/arm-check.json
python3 -m arm_lab_model.mujoco_backend simulate --pose home --target stowed \
  --duration 5 --output /tmp/arm-lab-runs/arm-motion.json
```

Pass `--config /absolute/path/my_arm.yaml` to these commands for another legacy
arm design. Use legacy arm YAML here, not a pipeline project manifest.

## UR5e arm

```bash
python3 -m arm_lab_model.pipeline_cli build "$CFG/project_ur5e_sim.yaml" \
  --output /tmp/arm-lab-runs/ur5e-model
python3 -m arm_lab_model.pipeline_cli simulate "$CFG/project_ur5e_sim.yaml" \
  --scenario "$CFG/scenario_ur5e.yaml" --save --output /tmp/arm-lab-runs/ur5e
python3 -m arm_lab_model.pipeline_cli reference-check "$CFG/project_ur5e.yaml" \
  --output /tmp/arm-lab-runs/ur5e-reference
```

All three commands should return 0. `reference-check` compares five analytical
poses with published nominal UR5e kinematics, not physical motion measurements.
The simulation writes `execution.json` and `observation.json`. The configured
UR5e storage directory is `saved_trajectories/ur5e/` in this source checkout.
The execution report includes the exact unique `saved_trajectory` path.

For a measured comparison, use `project_ur5e.yaml` and
`--reference /absolute/path/reference.yaml` with matching scenario/frame/joints.
Without a trace, that profile intentionally reports `benchmark.status: incomplete`
and exits 1. See [reference formats and reports](TRAJECTORIES_BENCHMARKS.md).

## Literal 1-DOF dog: build, move, save and replay

```bash
python3 -m arm_lab_model.pipeline_cli build "$CFG/project_dog1_demo.yaml" \
  --output /tmp/arm-lab-runs/dog1-model
python3 -m arm_lab_model.pipeline_cli simulate "$CFG/project_dog1_demo.yaml" \
  --scenario "$CFG/scenario_dog1_demo.yaml" --save --output /tmp/arm-lab-runs/dog1
DOG1_TRAJECTORY=$(python3 -c 'import json; print(json.load(open("/tmp/arm-lab-runs/dog1/execution.json"))["saved_trajectory"])')
python3 -m arm_lab_model.pipeline_cli replay "$CFG/project_dog1_demo.yaml" \
  --trajectory "$DOG1_TRAJECTORY" --output /tmp/arm-lab-runs/dog1-replay
```

The front-left hip swings through `0 → 0.25 → -0.20 → 0` radians over three
seconds. All commands should return 0, with no contact or saturation. Because
this profile has no configured storage directory, the saved record is under
`/tmp/arm-lab-runs/dog1/saved_trajectories/dog1_demo/`.

## Existing 12-DOF dog: contact smoke test

```bash
python3 -m arm_lab_model.pipeline_cli build "$CFG/project_dog12_demo.yaml" \
  --output /tmp/arm-lab-runs/dog12-model
python3 -m arm_lab_model.pipeline_cli simulate "$CFG/project_dog12_demo.yaml" \
  --scenario "$CFG/scenario_dog12_demo.yaml" --output /tmp/arm-lab-runs/dog12
```

Build returns 0. Simulation returns **1 by design** once floor contacts occur;
inspect `max_contacts` and `saturated_steps` in `execution.json`. The tests below
assert that result. This is not a successful trajectory to save or a walking demo.

## ROS simulation and RViz

Use a fresh shell with system ROS Python, outside the isolated headless virtualenv.
The unified pipeline was tested on **ROS 2 Humble**; the legacy Gazebo instructions
in Getting Started describe the separate Jazzy/Harmonic workflow. Run Bash for
these examples, or use `setup.zsh` in Zsh.

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
# Install simulator/report dependencies in this ROS Python environment as needed.
python3 -m pip install --user 'mujoco>=3.2,<4' 'matplotlib>=3.6,<4'
python3 -c 'import rclpy, mujoco, matplotlib; print("ROS and simulator imports OK")'
colcon build --symlink-install --packages-select \
  arm_lab_model arm_lab_kinematics arm_lab_gui arm_lab_bringup
source install/setup.bash
export CFG="$PWD/src/arm_lab_model/config/pipeline"
export ROS_DOMAIN_ID=76 ROS_LOCALHOST_ONLY=1
```

ROS, pip and rosdep must already be installed and rosdep initialized. On this
workspace's tested machine the dependencies are already present; installation
commands are for a fresh setup. A colcon build alone does not install the pip
MuJoCo dependency. An isolated headless virtualenv's packages are not visible to
the system Python used by these ROS nodes.

Run one profile at a time; stop its launch with Ctrl-C before switching robots.
All participating terminals must source ROS/workspace and use the same domain.

```bash
# UR5e: MoveIt planning panel, robot and MuJoCo joint control
ros2 launch arm_lab_bringup pipeline.launch.py \
  project_file:="$CFG/project_ur5e_sim.yaml" output_dir:=/tmp/arm-lab-runs/ros-ur5e

# Or dog1: robot display and direct joint controller, without MoveIt
ros2 launch arm_lab_bringup pipeline.launch.py \
  project_file:="$CFG/project_dog1_demo.yaml" output_dir:=/tmp/arm-lab-runs/ros-dog1
```

Append `rviz:=false` for headless ROS execution. Use the dog12 project in the
second command to visualize its passive fall/contact behavior. Launching dog1
holds its initial pose; send a trajectory to move it.

In a second sourced terminal, while **dog1** is running:

```bash
export ROS_DOMAIN_ID=76 ROS_LOCALHOST_ONLY=1
ros2 action list
ros2 action send_goal /arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  '{trajectory: {joint_names: [front_left_hip_pitch], points: [{positions: [0.0], time_from_start: {sec: 0}}, {positions: [0.25], time_from_start: {sec: 2}}, {positions: [0.0], time_from_start: {sec: 4}}]}}' --feedback
```

Expect action `SUCCEEDED` and `error_code: 0`. This returns the leg to zero,
allowing the same command to be repeated. Observe `/joint_states` or RViz for
actual movement. UR5e position/pose targeting, preview, execute and Save commands
are in [the planning guide](PIPELINE_PLANNING.md). For RGB-D/OctoMap, select
`project_ur5e_perception.yaml` and follow [perception checks](SCENE_PERCEPTION.md).

## Tests and documentation checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_model/test/test_dog1_demo.py \
  src/arm_lab_model/test/test_dog12_demo.py \
  src/arm_lab_model/test/test_benchmark_reference.py
python3 tools/build_docs.py --check
git diff --check
```

Source ROS before running ROS adapter/launch tests. The full suite intentionally
skips the opt-in live MoveIt test unless enabled against a running UR5e pipeline:

```bash
ARM_LAB_LIVE_PIPELINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_kinematics/test/test_pipeline_planning.py -k live
```

With only dog1 running in the matching domain, its direct-action live check is:

```bash
ARM_LAB_LIVE_DOG1=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_gui/test/test_pipeline_dog1_live.py
```

Both live checks command simulated movement; the MoveIt check also saves a
trajectory. Keep matching domain
settings and stop unrelated launches first. Optional renderer tests require EGL;
unavailable dependencies are reported as skips, not successful execution.

## Future agent work

Pre-publish validation on **2026-09-26**:

- Fresh Python **3.12.14**, NumPy **2.5.3** and MuJoCo **3.14.0**: **393 passed,
  16 skipped** in the headless suite. Optional ROS/KDL and live checks require
  their separate environments; skips are not passes.
- The editable model package installed successfully. The CI commands for the
  150-sample legacy physics check, 0.5-second hold simulation and five-case UR5e
  nominal FK reference check all exited successfully.
- The sourced ROS/system-Python suite passed **410 tests**, with two opt-in live
  checks skipped, before the final NumPy compatibility regression was added.
  Benchmark/report/CLI tests then passed **50 checks**, including nonuniform-time
  effort/work integration without the removed `numpy.trapz` API; the fresh
  Python 3.12 full-suite count above includes this regression.
- Documentation generation/freshness and whitespace checks passed. `pip-audit`
  found no known vulnerabilities among **29 dependencies** resolved from
  `requirements-dev.txt` in the Python 3.10 audit environment. This audit does not
  cover ROS system packages or establish that every supported dependency version
  is vulnerability-free.

Earlier validation for the robot-profile addition: **409 full-suite tests passed**, with the two live
ROS checks excluded by their opt-in switches. The dog1 live action test then
passed separately: measured peak 0.249997 rad, final 0.00000229 rad, and action
SUCCESS. A subsequent scene-service regression and related boundary tests passed
all **12 checks**, with **81% statement coverage of `pipeline_scene.py`**. The
four ROS packages built successfully. Original-arm physics/motion commands and
the expected dog12 contact rejection were also exercised. Workspace skill
frontmatter, documentation freshness and local links were validated.

[AGENTS.md](../AGENTS.md) routes tasks to source files and documentation.
Reusable [modeling](../skills/arm-lab-modeling/SKILL.md) and
[validation](../skills/arm-lab-validation/SKILL.md) skills cover adding/editing
robots, physical invariants, test selection and evidence reporting. Ask an agent
to read either workspace skill by path; no global skill installation is needed.
