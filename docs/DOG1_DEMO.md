# Literal 1-DOF dog-shaped fixture

`dog1_demo` is a runnable synthetic robot with **exactly one actuated joint**:
`front_left_hip_pitch`. It has a fixed torso, four legs, a head and a tail. Only
the front-left leg swings; the other limbs are attached with fixed joints. This
is a bench fixture for model generation and trajectory testing, not a walking
robot, a vendor model or hardware benchmark. The separate floating
[12-DOF dog example](DOG12_DEMO.md) remains available.

## Files and physical inputs

All configurations are under `src/arm_lab_model/config/pipeline/`:

- `robot_dog1_demo.yaml`: complete tree, box geometry, inline material density,
  automatic mass/COM/inertia and explicit joint limits.
- `project_dog1_demo.yaml`: robot and shared MuJoCo simulation settings.
- `scenario_dog1_demo.yaml`: `synthetic_dog1_leg_swing`, a three-second joint path
  `0 → 0.25 → -0.20 → 0` radians. Endpoint `foot` names `front_left_foot`.

The homogeneous effective density is **500 kg/m³**, an illustrative input rather
than an asserted material or assembly measurement. The independently calculated
mass is **9.198 kg**: 7.2 kg torso, four 0.24 kg legs, 1.008 kg head and 0.03 kg
tail. The massless foot frame contributes no inertia. The fixed mount places the
torso at world Z=0.5 m. No floor or contact support is required for this fixture.

The moving leg is a 0.3 m uniform box with its COM 0.15 m below the hip.
At joint angle `q`, gravity compensation is
`0.24 × 9.81 × 0.15 × sin(q)` Nm. Tests compare that independent result with both
the generic static report and MuJoCo controller effort. Joint bounds are ±0.6 rad,
2 rad/s, 4 rad/s² and 5 Nm; all are synthetic demonstration limits.

## Build, execute, save and replay

Run from the repository root:

```bash
export PYTHONPATH="$PWD/src/arm_lab_model:$PWD/src/arm_lab_kinematics${PYTHONPATH:+:$PYTHONPATH}"

python3 -m arm_lab_model.pipeline_cli build \
  src/arm_lab_model/config/pipeline/project_dog1_demo.yaml \
  --output /tmp/dog1-build

python3 -m arm_lab_model.pipeline_cli simulate \
  src/arm_lab_model/config/pipeline/project_dog1_demo.yaml \
  --scenario src/arm_lab_model/config/pipeline/scenario_dog1_demo.yaml \
  --output /tmp/dog1-run --save

DOG1_TRAJECTORY=$(python3 -c \
  'import json; print(json.load(open("/tmp/dog1-run/execution.json"))["saved_trajectory"])')

python3 -m arm_lab_model.pipeline_cli replay \
  src/arm_lab_model/config/pipeline/project_dog1_demo.yaml \
  --trajectory "$DOG1_TRAJECTORY" --output /tmp/dog1-replay
```

Build outputs include URDF, MJCF, resolved model, physical report and manifest.
Execution outputs include simulation observations, an execution report and the
successful saved trajectory. All commands above return zero on successful runs.
The saved path is read from the execution report because filenames are unique.

## Validation and limitations

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_model/test/test_dog1_demo.py
```

Both integration tests pass. They check exactly one actuator/position/velocity
DOF, four legs, independent total mass and gravity torque, model export, actual
leg motion, successful CLI storage, and deterministic replay. The exercised
three-second trajectory reached a maximum tracking error of
**0.00007986 rad**, with **zero contacts and zero saturated steps**.

The robot is rigid and fixed to the world. Its inputs do not represent actual
motor, gearbox, friction, compliance or manufactured parts. Collision checking
uses the declared primitive boxes. These results establish numerical behavior
for this explicit synthetic fixture; they establish neither locomotion nor
agreement with a physical dog robot.
