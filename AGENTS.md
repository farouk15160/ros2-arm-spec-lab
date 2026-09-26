# Working on Robot Design Lab

This workspace contains a legacy serial-arm design bench and a general tree-robot
pipeline. Preserve both interfaces when making shared changes.

## Start and route the task

Read [the documentation index](docs/README.md) and
[the robot run guide](docs/RUN_ROBOTS.md). Before changing a subsystem, follow its
row below. Inspect `git status --short` and preserve existing user edits. When
agents work concurrently, assign file ownership and coordinate shared changes.

| Task | Read first | Main edit locations |
|---|---|---|
| Add a robot, change joints, import CAD/STL, change materials or inertials | [Modeling skill](skills/arm-lab-modeling/SKILL.md) | `src/arm_lab_model/config/pipeline/`, `robot_topology.py`, `physical_robot.py`, `robot_export.py` |
| Run tests, diagnose physics differences, add reference data or benchmark metrics | [Validation skill](skills/arm-lab-validation/SKILL.md) | package `test/` directories, `pipeline_runtime.py`, `benchmark_*.py` |
| Plan/execute through ROS, alter launch or interactive targets | [Planning guide](docs/PIPELINE_PLANNING.md) and validation skill | `arm_lab_kinematics/pipeline_*.py`, `arm_lab_gui/pipeline_sim_node.py`, `arm_lab_bringup/launch/` |
| Change worlds, sensors, clouds or algorithm plugins | [Scene/perception guide](docs/SCENE_PERCEPTION.md) and validation skill | `scene_config.py`, `sensor_runtime.py`, `perception.py`, GUI `pipeline_scene.py`/`pipeline_perception.py` |
| Change legacy arm calculations or editor | [Development guide](docs/DEVELOPMENT.md), [configuration](docs/CONFIGURATION.md) | model `config.py`/`kinematics.py`, GUI schema/editor, legacy launch files |

Python module names in this table are relative to their package's inner Python
directory. These workspace skills live in canonical `skills/`; read the linked
`SKILL.md` when its task applies. They do not require changing global Codex config
or relying on a client-specific skill-menu installation.

## Model and evidence invariants

- `load_project` validates and freezes configuration; `resolve_robot` supplies the
  shared physical tree consumed by URDF and MuJoCo. Add physical behavior there
  before introducing a second definition in a runtime adapter.
- Component paths resolve against the file declaring them. Keep ordered joint
  names, SI units, frame names, COM-frame tensors and quaternion conventions
  explicit. See the schema guide before extending a format.
- Keep unknown manufacturer values unknown. Separate experiment acceleration
  policies from rated limits; an override may tighten a known limit, not raise it.
- Synthetic fixtures, nominal manufacturer kinematics and measured hardware
  traces are different evidence. Preserve that distinction in reports and tests.
- Dog1 is a fixed-base one-joint test rig. Dog12 is a floating contact fixture.
  Neither supplies locomotion or establishes a particular vendor robot model.
- Save/replay retains model identity, joint order, endpoint and applicable limits.
  Floating replay needs its initial base pose and velocity. Preserve rejection of
  invalid, colliding, saturated or stale execution requests.

## Completion and validation

For behavior changes, first add a failing regression through an existing public
loader, exporter, CLI, physics runtime or ROS boundary. Use independent physical
expectations where possible, rather than recalculating an expected result with
its own implementation. Keep configuration/value objects immutable; simulator
state and ROS resources necessarily have managed lifecycles.

Run focused tests, then the relevant integration checks in the validation skill.
Report skipped dependencies and untested hardware/UI paths separately. Target
at least 80% statement coverage for changed production modules; give the exact
scope and avoid treating legacy repository coverage as new-module coverage.

Update the relevant narrative guide with the implementation, file locations,
operation, test commands and limitations. Generated `docs/COMMANDS.md`,
`docs/API_REFERENCE.md` and `docs/CONFIG_VALUES.md` come from:

```bash
python3 tools/build_docs.py
python3 tools/build_docs.py --check
git diff --check
```

Use the latest user instruction to determine whether to stop after a phase or
continue. Work in this checkout is not permission to push, deploy or run a real
robot. Keep test outputs under a requested output directory or `/tmp` and stop
only the ROS processes started for the task.
