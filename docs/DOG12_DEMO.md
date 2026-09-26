# Synthetic 12-DOF dog robot demo

The runnable `dog12_demo` example demonstrates a robot type beyond serial arms:
four branches with three actuated revolute joints per leg and a floating torso.
It is **synthetic**, with illustrative geometry, density and joint limits. It is
not vendor CAD, a specific ODG product, measured hardware or a gait controller.
The original `robot_quadruped_12dof.yaml` remains an unknown-parameter topology
example; this separate demo supplies complete explicit inputs for physics tests.

## Configuration

- `config/pipeline/robot_dog12_demo.yaml`: torso, twelve moving links, four
  massless foot frames, primitive box geometry and automatic homogeneous inertials.
- `config/pipeline/project_dog12_demo.yaml`: manifest selecting robot, simulation
  and environment.
- `config/pipeline/environment_dog12.yaml`: static floor with its top at world Z=0.
- `config/pipeline/scenario_dog12_demo.yaml`: 0.6-second zero-joint hold during free
  fall and floor contact; all twelve actuator names are explicit.

These paths are under `src/arm_lab_model/`. The effective material density is
500 kg/m³; it represents an illustrative uniform solid, not an actual composite
or assembled robot. Total mass is 9.56 kg. The torso starts at Z=0.5 m and the
straight legs reach Z=0.1 m. The free base adds six velocity coordinates (18 total)
and seven position coordinates (19 total), with twelve actuators. Joint motion,
velocity, acceleration and effort limits are explicitly synthetic policy values.

## Run and inspect

```bash
PYTHONPATH=src/arm_lab_model:src/arm_lab_kinematics python3 -m arm_lab_model.pipeline_cli build \
  src/arm_lab_model/config/pipeline/project_dog12_demo.yaml \
  --output /tmp/dog12-build

PYTHONPATH=src/arm_lab_model:src/arm_lab_kinematics python3 -m arm_lab_model.pipeline_cli simulate \
  src/arm_lab_model/config/pipeline/project_dog12_demo.yaml \
  --scenario src/arm_lab_model/config/pipeline/scenario_dog12_demo.yaml \
  --output /tmp/dog12-drop

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_model/test/test_dog12_demo.py
```

The `simulate` command deliberately returns exit status **1** once floor contacts
occur: the generic trajectory acceptance criterion requires collision-free
execution without actuator saturation. The drop can also saturate actuators at
impact (the tested run recorded 35 saturated steps and 16 simultaneous contacts). Inspect `execution.json` for `max_contacts > 0` and inspect
`observation.json` for the measured simulation channels. That rejection is correct
for the generic trajectory policy; this drop fixture is a numerical smoke test,
not a successful locomotion task or a saveable collision-free trajectory.

## Evidence and limits

Three tests check the independent analytic mass sum against MuJoCo body mass,
COM consistency, 12 actuators and 18 velocity DOF, finite warning-free simulation,
actual floor contacts involving lower legs, bounded actuator movement across
all branches, and honest rejection of the contact-rich hold scenario.

The joint controller supplies no floating-base balance, foothold selection,
contact-aware trajectory acceptance or locomotion policy. The robot may fall or
settle against the floor. Contact defaults are MuJoCo defaults and have not been
calibrated to materials. No real-robot benchmark agreement is claimed. A real
12-DOF robot needs its own measured model, actuator limits, calibrated contacts,
base state and contact-aware controller before motion results can be interpreted
as hardware predictions.
