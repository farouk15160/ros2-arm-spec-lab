# Trajectory storage and benchmark evidence

This page records Steps 7, 11 and 12. All numerical test examples are explicitly
synthetic. A passing comparison against synthetic or nominal analytical data
does **not** mean that a physical UR5e has been validated.

## Step 7 — Save successful trajectories

`arm_lab_model.trajectory_store.save_trajectory(record, directory)` validates a
version 1 record and returns a unique path under `directory/<robot>/`. Publication
is atomic: readers never see a partially written YAML document. Existing records
are never overwritten. `load_trajectory(path, robot=None)` safely reads the same
format. Passing the replay robot checks the exact model digest, joint order and
declared position, velocity and acceleration limits before execution.

```yaml
schema_version: 1
robot: example_arm
model_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
scenario_id: small_joint_move
timestamp: '2026-09-25T12:00:00+00:00'
joint_names: [joint1]
start_state: [0.0]
target:
  joint_positions: [0.1]
trajectory:
  points:
    - {time_from_start: 0.0, positions: [0.0], velocities: [0.0], accelerations: [0.0]}
    - {time_from_start: 1.0, positions: [0.1], velocities: [0.0], accelerations: [0.0]}
planner: {name: example_planner, parameters: {}}
collision: {status: clear}
execution: {status: succeeded}
benchmark: null
```

The example hash is illustrative: use the generated model's actual SHA256.
For a Cartesian target, replace `target` with `frame`, `position: [x,y,z]` and
optional `orientation: [x,y,z,w]` unit quaternion. All angles and times use SI.
`collision` and `execution` may additionally carry `details`.

Floating robots additionally save `base_start_state: {qpos: [...], qvel: [...]}`.
The seven `qpos` entries are world position xyz followed by a normalized
quaternion **wxyz**, matching a MuJoCo free joint; six `qvel` entries contain
the generalized base velocity. These differ from TCP pose quaternion **xyzw**.
Replay restores the recorded floating base as well as joint state. Existing
fixed-base records need no additional field.

Only `collision.status: clear` and `execution.status: succeeded` records can be
saved. Timings must start at zero and increase strictly. Points require finite
positions, velocities and accelerations for every joint. Start state must equal
the first point. The planner must explicitly provide derivatives; storage does
not fill absent velocities or accelerations with invented zeroes.

```python
from arm_lab_model.trajectory_store import load_trajectory, save_trajectory

path = save_trajectory(record, 'saved_trajectories')
replay_record = load_trajectory(path, robot={
    'name': robot_name,
    'model_sha256': model_digest,
    'joint_names': joint_names,
    'limits': {'joint1': {'lower': -1.0, 'upper': 1.0,
                         'velocity': 0.5, 'acceleration': 1.0}},
})
```

Continuous joints omit both position bounds. Unknown dynamic bounds may be
omitted but must be supplied to the execution system when required. Storage
validates sampled values; execution must additionally validate interpolation,
current start state, collisions, controller state and dynamic limits.

The project CLI saves to `trajectories.directory`, resolved relative to the
trajectory component YAML. Without that setting it uses
`<output>/saved_trajectories`. `simulate --save` writes only successful execution.
`replay --trajectory <path>` restores saved acceleration policy and TCP selection,
falling back to project MoveIt acceleration policy for unspecified joints. An
explicit scenario `end_effector` can name a configured endpoint key or its link.
Headless replay starts a fresh experiment at the recorded joint and floating-base
state; ROS replay must instead verify the existing live start state.
When benchmarking is selected, the CLI validates the configured robot and loads
the reference before starting motion. Known scenario, frame, joint-type and TCP
identity mismatches are rejected at this point. ROS adapters can use the same
`prepare_benchmark(project, expected=...)` preflight API during startup and goal
acceptance; returned observations can be passed to `benchmark_execution` as
`reference_data` to preserve the exact validated snapshot.

```bash
PYTHONPATH=src/arm_lab_model:src/arm_lab_kinematics python3 -m arm_lab_model.pipeline_cli \
  simulate src/arm_lab_model/config/pipeline/project_ur5e.yaml \
  --scenario src/arm_lab_model/config/pipeline/scenario_ur5e.yaml \
  --output /tmp/ur5e-run --save
```

Step 7 test: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest
src/arm_lab_model/test/test_trajectory_store.py -q`.

## Step 11 — Compare simulation with external observations

`compare_runs(simulation, reference, tolerances)` consumes two observation
documents. They require matching robot, scenario, joint order, coordinate frame,
units and continuous-joint declarations. Times must share an experiment-relative
clock. The engine never guesses clock offsets, changes frames, reorders joints
or extrapolates outside recorded data. A partial-overlap comparison is marked
`incomplete`, even if the observed interval is within tolerance.

```yaml
schema_version: 1
robot: example_arm
scenario_id: small_joint_move
joint_names: [joint1]
frame: world
units: SI
evidence:
  kind: synthetic
  source: Documentation example; not hardware measurements
samples:
  time: [0.0, 0.5, 1.0]
  joint_position: [[0.0], [0.05], [0.1]]
  joint_velocity: [[0.0], [0.15], [0.0]]
  joint_acceleration: [[0.0], [0.0], [0.0]]
```

Optional channels are `joint_torque`, `tcp_position`, `tcp_orientation` (xyzw
unit quaternion) and `tcp_velocity`. Joint channels have N×J values and TCP
vectors have N×3; orientation has N×4. At least two time samples and one channel
are required. Optional metadata: `model_sha256`, `conditions`, and
`continuous_joints: [joint_name, ...]`, and `joint_types: [revolute, ...]`.
Types are `revolute`, `continuous`, or `prismatic`, defaulting to revolute for
older observations. Continuous types automatically enable angular wrap handling.
An optional `end_effector` identifies the actual TCP link/body. If supplied by
either observation, both must supply the same value; equal coordinate frames
alone do not establish that two observations describe the same point.
Evidence kind is `measured`,
`manufacturer`, `analytical`, or `synthetic`; `source` must describe the actual
origin. Labelling a file is not independent verification of its authenticity.
Declare the measured controller, calibration, payload, gravity and acquisition
conditions in the benchmark catalogue, and explicitly match them in simulation.

The packaged UR5e catalogue's untimed nominal FK cases are a separate evidence
type. They must not be assigned fabricated timestamps or presented as recorded
hardware trajectories. Missing real reference logs remain unavailable.

Run the independent nominal kinematics consistency check directly:

```bash
PYTHONPATH=src/arm_lab_model:src/arm_lab_kinematics python3 -m arm_lab_model.pipeline_cli \
  reference-check src/arm_lab_model/config/pipeline/project_ur5e.yaml \
  --output /tmp/ur5e-reference-check
```

This reads `benchmark.config` relative to its component YAML, verifies catalogue
file hashes and writes `reference_check.json`. Its five static FK cases check the
generated tree against manufacturer DH-derived analytical poses. The result
explicitly reports `hardware_validation: false`, and mass disagreement remains
visible separately from nominal kinematics acceptance.

```python
from arm_lab_model.benchmark_engine import compare_runs, load_observation
from arm_lab_model.benchmark_reports import write_benchmark_report

simulation = load_observation('simulation.yaml')
reference = load_observation('measured_run.yaml')
result = compare_runs(simulation, reference, {
    'joint_position': 0.01, 'tcp_position': 0.002,
    'tcp_orientation': 0.02, 'duration': 0.01,
})
paths = write_benchmark_report(result, 'reports/run_001')
```

External CSV ingestion requires explicit metadata and column declarations:

```python
reference = load_observation('measured.csv', metadata={
    'schema_version': 1, 'robot': 'example_arm', 'scenario_id': 'small_joint_move',
    'joint_names': ['joint1'], 'frame': 'world', 'units': 'SI',
    'evidence': {'kind': 'measured', 'source': 'Actual acquisition record identifier'},
}, columns={'time': 'seconds', 'joint_position': ['q1_rad'],
            'joint_torque': ['tau1_nm']})
```

Convert units and coordinate frames explicitly before import. Revolute joint
channels use rad, rad/s, rad/s² and N m. Declare prismatic joints explicitly:
their channels use m, m/s, m/s² and N (the channel name remains `joint_torque`
for actuator generalized effort). Mixed robots report errors per joint with
individual units; aggregating angular and linear RMSE is deliberately omitted.
The numerical channel tolerance is applied separately in each joint's SI unit.
The format supports arbitrary joint counts and tree topology: a quadruped with
12 revolute joints uses the same observation shape and storage API.

Alignment uses the union of timestamps in the overlapping interval. Scalar and
vector channels are linearly interpolated; orientation uses shortest-arc SLERP.
Optional continuous-joint position channels are unwrapped before interpolation
and compared modulo 2π. TCP position/velocity errors are Euclidean norms;
orientation error is the relative rotation angle. Joint errors are per-component.
Reported RMSE is sample-weighted; maximum and final errors are absolute maxima
over components. Tolerances apply to the maximum error, not RMSE.

Tolerance keys: `joint_position`, `joint_velocity`, `joint_acceleration`,
`joint_torque`, `tcp_position`, `tcp_orientation`, `tcp_velocity`, `duration`,
`energy`, `effort`. Missing requested channels produce `unavailable` and an
overall `incomplete` result. No tolerance means `unassessed`, never a pass.
Unrequested missing channels are listed but do not become implicit criteria.

Duration compares the full recorded intervals. Effort is the trapezoidal integral
of the sum of absolute joint torques over the overlap. Energy is the trapezoidal
integral of absolute per-joint torque × velocity, summed across joints. This is
absolute mechanical work, not electrical consumption, and opposing joints do
not cancel. Force × linear velocity similarly gives prismatic mechanical work.
Effort for mixed linear/angular joints is unavailable because N s and N m s
cannot be summed into a physically meaningful scalar. Sample rate and
interpolation affect integral accuracy. Clock endpoints within 1 ns are treated
as equivalent to tolerate floating-point stepping noise.

Step 11 test: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest
src/arm_lab_model/test/test_benchmark_engine.py -q`. Tests include independently
known constant-offset errors, quaternion double-cover and interpolation,
constant power (3 N m × 2 rad/s × 2 s = 12 J), mismatched metadata, unavailable
channels, time overlap, continuous-joint wrap and explicit CSV import.

## Step 12 — Reports and plots

`write_benchmark_report` writes `benchmark.json`, `benchmark.md`, individual
channel plots, `errors.png`, and TCP XY/XZ/YZ projections where available. The
report includes evidence classes/sources, tolerances, status, final TCP pose,
final torque, interval details and limitations. All plots show simulation and
reference series on the same axes. Plots use a headless Matplotlib canvas; no
display server is required. When Matplotlib is absent the numerical report
remains available and the Markdown explains why plots were not generated.

Choose a fresh output directory per experiment: reports in the same directory
replace reports from the preceding invocation. Trajectory storage always uses
unique paths instead. Report artifacts are intended for local trusted viewing.

Step 12 test: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest
src/arm_lab_model/test/test_benchmark_reports.py -q`. Report tests check
machine-readable status, explicit synthetic provenance, final-state summary,
comparison figures and incomplete-evidence reporting.

The public command workflow is exercised with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest
src/arm_lab_model/test/test_pipeline_cli.py -q`. Tests build and analyze the actual
UR5e project, check five manufacturer-derived FK poses, execute synthetic MuJoCo
holds, save and replay with preserved policy, restore floating-base state, compare
observations, and reject malformed scenarios or mismatched identities.
