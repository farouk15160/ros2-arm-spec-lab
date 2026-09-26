# UR5e benchmark reference

[Documentation index](README.md) · [Extended pipeline](EXTENDED_PIPELINE.md)

## Step 2 implementation

The UR5e is the first manufacturer-reference robot. Its tree is shared by later
physical analysis, global description, simulation and planning. The benchmark
catalogue links to that tree instead of duplicating link inertials.

Files:

- `src/arm_lab_model/config/benchmarks/ur5e.yaml`: robot identity, hardware position
  and speed limits, separate planning/model limits, payload, nominal DH parameters,
  source manifest, dataset catalogue and explicit limitations.
- `src/arm_lab_model/config/pipeline/robot_ur5e.yaml`: six-axis topology, nominal
  link frames, manual inertials, approximate visual/collision primitives, controller
  `base`, flange, force/torque frame and `tool0`.
- `src/arm_lab_model/config/pipeline/project_ur5e.yaml`: project entry point.
- `src/arm_lab_model/config/benchmarks/upstream/ur5e/`: verbatim pinned upstream
  parameters, Xacro frame/inertial definitions and license. These are provenance
  snapshots, not a standalone installation of `ur_description`.
- `src/arm_lab_model/config/benchmarks/datasets/ur5e_nominal_fk.yaml`: five untimed
  analytical forward-kinematics cases derived from the manufacturer's DH table.
- `src/arm_lab_model/arm_lab_model/benchmark_reference.py`: immutable strict loader
  that verifies local source/data SHA256 hashes and evidence classification.
- `src/arm_lab_model/test/test_benchmark_reference.py`: catalogue, integrity,
  malformed input and independent DH/tree consistency checks.

## Primary sources and interpretation

The [official ROS 2 description repository](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/b48aa88ac18a17466e767929f05f45b23e332a8e)
was pinned to commit `b48aa88ac18a17466e767929f05f45b23e332a8e` from its Humble
branch. Downloaded files retain the upstream BSD license and source headers.
The narrow `.gitattributes` whitespace exception preserves the original mixed
indentation in `ur_macro.xacro` so its recorded checksum remains verifiable.
The catalogue records a SHA256 hash for each file. The link inertial tensors are
rotated from upstream inertial-frame axes into link axes using `R I R^T`;
CoM vectors already use link axes. The base uses the upstream cylinder estimate.

The [UR5e SW5.20 technical specifications](https://www.universal-robots.com/manuals/EN/HTML/SW5_20/Content/prod-usr-man/complianceUR5e/H_g5_sections/appendix_g5/tech_spec_sheet.htm)
supply a 5 kg maximum payload, joint travel of ±360°, a maximum joint speed of
180°/s and a 20.7 kg robot mass. These are specifications, not measured trajectories.
Pose repeatability is not interpreted as absolute TCP accuracy or a tracking-error
tolerance. Payload CoM and inertia envelopes remain unknown in this catalogue.

The [manufacturer DH table](https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/)
is the independent kinematic reference. Its standard DH rows use metres and
radians. The untimed FK cases evaluate `Rz(theta) Tz(d) Tx(a) Rx(alpha)` in the
UR controller `base` frame, with the default `tool0` TCP. That `base` frame is
rotated π about Z from ROS `base_link`. The public table does not provide precise
UR5e link inertia tensors; the tree uses the ROS description's nonzero model
tensors instead of treating the table's approximation as literal zero inertia.

The model effort limits are 150 N·m for the first three joints and 28 N·m for the
wrists, taken from the ROS description. They are explicitly distinguished from
unknown continuous hardware torque ratings. Manufacturer acceleration limits
remain `null`. Applications must supply justified planning acceleration settings
instead of treating a chosen setting as a verified hardware limit. The ROS model
restricts the elbow to ±π for planning, while the manufacturer's joint-travel
specification remains ±2π in the benchmark catalogue.

**Known mass discrepancy:** the upstream model adds to 21.7 kg, whereas the manual
specifies 20.7 kg. Upstream explicitly warns that the 4 kg base mass may be
incorrect. This project preserves both values and reports the discrepancy; it
does not silently alter link masses to make the totals match. The base cylinder
and other primitive geometry are modeling approximations, not manufacturer CAD.
They cannot establish real collision clearance or contact behavior.

## Evidence and dataset contract

`load_benchmark_reference(path)` returns a frozen `BenchmarkReference` with
`path`, `robot_path`, ordered `joint_names` and recursively immutable `data`.
Loading performs local validation and hash checks only. It never starts a robot
or downloads data. Relative files resolve against the catalogue directory.

A dataset entry contains:

```yaml
kind: analytical_reference
source: ur_dh
file: datasets/ur5e_nominal_fk.yaml
sha256: "<64 lowercase hex characters for the actual file>"
format: static_fk_cases_v1
scenario: nominal_kinematics
joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
              wrist_1_joint, wrist_2_joint, wrist_3_joint]
channels:
  joint_position: {unit: rad, frame: null, uncertainty: null}
  tcp_position: {unit: m, frame: base, uncertainty: null}
  tcp_orientation: {unit: quaternion_xyzw, frame: base, uncertainty: null}
conditions:
  calibration: nominal
  controller: null
  payload: null
  gravity: null
  timestamp_convention: untimed static configurations
  acquisition: analytical evaluation
description: "Replace this illustrative entry with actual catalogue values."
```

The actual catalogue provides valid values and checksums. A static FK dataset
has `cases`, each with `name`, ordered `joint_position`, `tcp_position` and
`tcp_orientation` (quaternion XYZW); there are deliberately no invented times.
A later trajectory benchmark uses recorded elapsed timestamps and the observation
schema described in the benchmarking documentation.

Supported evidence labels are `hardware_measurement`, `manufacturer_reference`,
`analytical_reference` and `synthetic_fixture`. A hardware dataset requires a
hardware source and non-null calibration, controller, payload, gravity, timestamp
convention and acquisition conditions. Additional dataset-specific interpretation
belongs to its declared `format`; loading a catalogue does not establish that
channel samples are physically compatible with a selected simulation scenario.
Unknown uncertainties stay null. Never reclassify analytical or synthetic data
as hardware measurements.

**No real-robot motion recordings are bundled.** The included FK checks establish
agreement with published nominal kinematics, not hardware dynamics, tracking,
friction, energy or calibration accuracy. Adding a robot with documented public
motion measurements remains possible through a new catalogue and observation
files with acquisition and redistribution provenance.

## Testing and reproducibility

From the repository root:

```bash
PYTHONPATH=src/arm_lab_model python3 -m arm_lab_model.project_config \
  src/arm_lab_model/config/pipeline/project_ur5e.yaml
PYTHONPATH=src/arm_lab_model python3 -m arm_lab_model.pipeline_cli reference-check \
  src/arm_lab_model/config/pipeline/project_ur5e.yaml --output /tmp/ur5e-reference
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_model/test/test_benchmark_reference.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p pytest_cov \
  --cov=arm_lab_model.benchmark_reference --cov-report=term-missing \
  src/arm_lab_model/test/test_benchmark_reference.py
```

Stage verification: 17 tests passed. The catalogue loader and reference checker
achieved 85% statement coverage. All five independent
DH poses agree with the resolved ROS-frame tree to `5e-10` in position and rotation
matrix entries. The tolerance covers upstream rounded nominal transform angles;
it is a numerical consistency tolerance, not a hardware accuracy specification.
The project validator explicitly lists six missing joint acceleration limits.

`check_reference_model(resolved_robot, benchmark_path)` returns a JSON-ready report
with per-case position/orientation errors, an explicit analytical evidence label,
`hardware_validation: false`, and the 1 kg model/manufacturer mass discrepancy.
Its `passed` field describes nominal kinematics only.

The [integrated workflow](PIPELINE_WORKFLOW.md) now exports and executes this
model, including actual MoveIt simulation integration. Real-hardware execution,
calibrated geometry, measured motion traces and independently validated inertials
remain unavailable; successful simulation does not establish them.
