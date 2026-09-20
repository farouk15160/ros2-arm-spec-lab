# Operating and troubleshooting the bench

[Documentation index](README.md) · [Install instructions](GETTING_STARTED.md)

## Before a run

1. Choose a workflow: offline analysis, MuJoCo benchmark or ROS/Gazebo integration.
2. Record `git rev-parse HEAD`, the complete input config and any external assets.
3. Record payload, gravity, initial/target poses and all command overrides.
4. For ROS, source the intended workspace and confirm its generated model paths.
5. Choose acceptance criteria before interpreting the outcome; a report's scope
   determines what its pass flag means.

## Operational checks

For headless tooling:

```bash
robot_test --help
robot_test check --samples 10
robot_test simulate --duration 0.5
```

For an already running ROS simulation:

```bash
ros2 node list
ros2 control list_controllers
ros2 topic hz /joint_states
ros2 topic echo /clock --once
ros2 topic echo /diagnostics --once
```

The broadcaster and requested arm/gripper controllers should be active; the
spare velocity controller is intentionally inactive initially. `/joint_states`
must contain the configured joint names. Match `config_file`, tool mass, gravity
and payload between simulator and analysis nodes. There is no HTTP health endpoint.

For a reproducible live session, record relevant topics:

```bash
ros2 bag record /joint_states /clock /diagnostics \
  /arm_lab/tcp_speed /arm_lab/joint_torque_model /arm_lab/power
```

Archive command trajectories as well when diagnosing tracking; state/metrics
alone may not identify what was commanded.

## Failure guide

| Symptom | Check and response |
|---|---|
| `robot_test: command not found` | Activate the environment and install `-e src/arm_lab_model` from a current checkout; older legacy editable installation placed scripts in ROS's lib directory |
| Missing `mujoco` | Install the simulation extra or `requirements-dev.txt` in the interpreter running the command |
| Missing `rclpy` / `python_qt_binding` | Source ROS; an isolated venv or replacement `PYTHONPATH` may hide ROS/system packages |
| Config cannot load | Inspect the named field, units, library names, axis and inertia validity; compare with the generated config reference |
| Strict pose rejection | Supply exactly one finite angle per joint, within the configured interval |
| `spec_report` exits 1 | Review failed requirements; the model may load correctly but miss a target |
| `engineering_report` exits 0 | This is not an all-clear verdict; its CLI currently returns 0 after rendering findings |
| MuJoCo motion fails | Inspect warning counters, saturation, tracking and speed ratio; change physical inputs or duration for a justified reason, not only the tolerance |
| KDL unavailable | Run verification in an environment with `PyKDL`; keep missing verification distinct from a passed check |
| RViz moves but loads look implausible | RViz/joint sliders are kinematic visualization; run a physics test and inspect inertials |
| Gazebo spawn/controller timeout | Start with the first launch error; inspect description, plugin availability and controller-manager state |
| Gravity or payload mismatch | Pass explicit simulation overrides and matching analysis parameters; editing the dashboard or YAML alone may not modify spawned physics |
| Cartesian target is displaced | Target node does not transform `PoseStamped.header.frame_id`; use model world coordinates |
| No capability messages | Check `/joint_states`; capability publisher waits until state has been ingested |
| Successful quadruped/humanoid smoke test | Check actual behaviour separately; a falling robot can still pass numerical health criteria |

## Temporary artifacts and concurrent sessions

Launch helpers use `tempfile.gettempdir()/arm_lab_generated` for `arm.urdf`,
`controllers.yaml` and patched world copies. The editor uses
`tempfile.gettempdir()/arm_lab_editor` for candidate/launch YAML. On a usual Linux
setup these are under `/tmp`. They are regenerated and are not authoritative
design storage. Copy artifacts into a run-specific directory before another run
overwrites them.

The current launch uses shared file names and global ROS topic/controller names.
Do not assume two simultaneous robot instances are isolated. Separate temporary
roots plus ROS domains/namespaces and remapping require deliberate setup; the
provided launch is designed around one bench instance.

## Stop, reproduce and recover

Stop a launch with Ctrl+C in its owning terminal. Save the original configuration
before editing; restore a known config and relaunch to recover model state.
Generated URDF/MJCF files should be regenerated rather than repaired by hand.

For a reproducible older software version, create a separate checkout:

```bash
git worktree add --detach ../arm-lab-reproduction <known-commit>
```

Build and source that checkout in a separate shell. This avoids rewriting the
current working tree. Do not reset or force-push shared history just to reproduce
a result. If a code regression needs reverting on the shared branch, review the
revert as a normal change and rerun the relevant checks.

## Reporting a problem

Include the smallest input that reproduces the issue, exact command, commit,
Python/MuJoCo/ROS versions, error output and expected result. For simulation
include timestep, seed/keyframe, controller mode and payload. For disagreement
between engines, distinguish model-derived torques, commanded torques and
simulator-reported effort, and attach the exact generated models.
