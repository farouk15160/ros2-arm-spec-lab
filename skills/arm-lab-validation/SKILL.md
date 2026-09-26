---
name: arm-lab-validation
description: Test Robot Design Lab changes across configuration, physical calculations, MuJoCo execution, ROS/MoveIt, perception and benchmark reports. Use when validating a robot profile, investigating numerical differences or reviewing pipeline changes.
---

# Validate the changed behavior

Read [run commands](../../docs/RUN_ROBOTS.md) for profiles/expected exits and
[development](../../docs/DEVELOPMENT.md) for the package change map. Commands run
from the repository root. Use public seams already exercised by package tests.

## Choose the necessary checks

| Change | Focused regression files under package `test/` |
|---|---|
| Configuration, topology, limits | `test_project_config.py`, `test_pipeline_cli.py` |
| STL/inertials/shared exports | `test_mesh_physics.py`, `test_physical_robot.py`, `test_robot_export.py` |
| Servo/interpolation | `test_pipeline_runtime.py`, GUI `test_trajectory_tolerance.py` |
| Robot examples | `test_dog1_demo.py`, `test_dog12_demo.py`, `test_benchmark_reference.py` |
| Reference alignment/storage/reports | `test_benchmark_engine.py`, `test_trajectory_store.py`, `test_benchmark_reports.py` |
| MoveIt/launch | kinematics `test_pipeline_planning.py`/`test_pipeline_launch.py`, GUI `test_pipeline_scene_mode.py`/`test_pipeline_scene_service.py` |
| Cameras/world/perception | model `test_scene_perception.py`, GUI `test_pipeline_perception.py` |

Run a targeted failing test before a behavior fix, then the implemented behavior.
For a combined change, run the full suite after shared interfaces settle:

```bash
export PYTHONPATH="$PWD/src/arm_lab_model:$PWD/src/arm_lab_kinematics:$PWD/src/arm_lab_gui${PYTHONPATH:+:$PYTHONPATH}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

The plugin override avoids incompatible ambient ROS pytest plugins. Use explicit
`-p pytest_cov --cov=<changed_module> --cov-report=term-missing` for coverage.
State the selected modules and denominator; a skipped simulator or ROS test is
not runtime evidence. The repository's old GUI code has substantially different
coverage from the newer pipeline modules.

## Physics and reference evidence

Use an independent solid formula, gravity calculation or published nominal FK
case before trusting a numerical result. Test the configured scenario against
actual MuJoCo, including a deliberate invalid/failed case when changing limits
or success semantics. Preserve nonfinite-state, saturation and collision checks.
Use `robot_pipeline reference-check` for UR5e nominal kinematics; use measured
observations with `compare` or `--reference` for real tracking comparisons.

Reference observations must match scenario, ordered joints, joint types, frame,
units and TCP. Preserve quaternion conventions: observation orientation is XYZW;
floating-base saved `qpos` uses position XYZ then quaternion WXYZ. Label synthetic
and analytical evidence honestly. Missing measured channels or a reference yield
incomplete results rather than hardware success. For benchmark changes consult
[the record/report contract](../../docs/TRAJECTORIES_BENCHMARKS.md).

## ROS and rendering evidence

Use the installed ROS distribution's system Python in a fresh shell; this pipeline
has live Humble evidence. Build/source all four workspace packages. Run one robot
per ROS domain because `/joint_states`, `/clock` and controller names are shared.
Stop earlier launches before switching profiles; source the workspace and use the
same domain in every participating terminal.

For launch/control changes, verify actual action completion and joint movement.
The opt-in MoveIt test in the run guide requires a running UR5e pipeline and
moves/saves the simulated robot. For camera changes, run actual EGL projection
regressions and, when ROS integration changes, `tools/check_perception_live.py`
against the perception profile. Preserve ROS sensor timestamps and optical frames.
A loaded RViz config is not evidence of manually tested desktop interaction.

Local ROS middleware requires sockets. If sandbox policy blocks them, use the
normal approval mechanism; report unavailable evidence if approval is absent.
Keep subprocess coverage results tied to the source version actually executed.
Use a fresh run after source line changes when combining coverage records.

## Finish

Update the relevant guide with commands, results, skipped checks and limits.
Regenerate docs and run `python3 tools/build_docs.py --check` plus
`git diff --check`. Check the diff for unintended files, keep user edits, and stop
test processes you started. Successful simulation does not authorize hardware
motion or publishing changes.
