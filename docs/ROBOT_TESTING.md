# Robot design and testing workflow

This project combines the original **fixed-base serial arm design bench**, a
**generic external-MJCF smoke tester**, and the [general project pipeline](PIPELINE_WORKFLOW.md).
The pipeline supports branched physical models, MoveIt, sensors and reference
comparisons; the old sizing tools, arm IK and dashboard remain arm-specific.

## Choose an engine

Use MuJoCo for fast, reproducible headless dynamics and contact experiments.
Keep Gazebo for the existing ROS 2 controllers, topics, sensors and integration
workflow. You do not need both for every task. MuJoCo is an optional Python
dependency; the core analytical tests do not require it or ROS.

The native MJCF exporter avoids the old URDF-import comparison discrepancy.
It exports complete link inertials, output-side torque motors, reflected rotor
inertia, Coulomb friction, viscous damping, joint limits and a rigid tool. It
does not export ROS controllers, moving gripper fingers or a ROS bridge.

References: [MuJoCo modeling and kinematic trees](https://mujoco.readthedocs.io/en/stable/modeling.html),
[Python interface](https://mujoco.readthedocs.io/en/stable/python.html).

## Enter a physical design

Copy `src/arm_lab_model/config/arm_config.yaml`; use SI units throughout.

| Input | Meaning |
|---|---|
| Link length, diameter, wall, material | Geometry, collision cylinder, tube mass and beam stiffness |
| Optional `link.inertial.mass` | Complete moving assembly mass, kg, including its assigned motor and fittings |
| Optional `link.inertial.com` | CoM `[x,y,z]`, metres from the proximal link origin, in link axes |
| Optional `link.inertial.inertia` | `[ixx,iyy,izz,ixy,ixz,iyz]`, kg·m², about CoM, oriented in link axes |
| Joint origin / axis / limits | Parent-frame translation, joint-frame rotation, rotation axis, range and speed |
| Actuator peak / continuous torque | **Motor-side** N·m; output rating is torque × ratio × efficiency |
| Gear ratio / efficiency | Positive reduction ratio; efficiency in `(0,1]` |
| Rotor inertia | Motor-side kg·m²; reflected value is inertia × ratio² |
| Friction / viscous damping | Output-side Coulomb N·m / viscous N·m·s/rad |
| Kt / resistance / current / voltage | Electrical model inputs with consistent datasheet current conventions |
| Thermal resistance / capacity / temperature | Lumped winding model, used by the engineering report |

For a catalog actuator already rated at its output, do not multiply its torque
by its internal gear ratio again. Convert catalog quantities to this convention
or represent it as a unit-ratio output actuator, consistently converting its
speed, inertia and electrical model as well. Do not mix the two conventions.

Example addition inside a joint's `link:` mapping:

```yaml
inertial:
  mass: 2.3
  com: [0.11, -0.02, 0.03]
  inertia: [0.02, 0.03, 0.04, 0.001, -0.002, 0.003]
```

All three fields are required together. They replace the tube-plus-motor
assembly estimate, so the motor is not added twice. An explicit inertia tensor
must be positive definite and its principal moments must satisfy the triangle
inequalities. CAD packages sometimes report *products of inertia* with opposite
signs; convert to the URDF tensor convention first. Rotate a CAD tensor into
link axes and shift it to the CoM before entering it.

The editor's **Mass / CoM / inertia…** button provides the same fields and a
**Use tube estimate** reset. Tube geometry still sets collision shape and
structural stiffness. A measured tensor does not turn a tube into an accurate
mesh or finite-element model. Geometry changes do not rescale a measured tensor;
update the mass properties from CAD or measurement yourself.

Validation rejects nonfinite vectors, nonpositive dimensions, invalid material
properties, impossible efficiencies, reversed limits and unsupported joint
types. The arm schema supports revolute and continuous joints. A linear actuator
needs force units and a linear transmission model; the old prismatic option
incorrectly reused rotary torque/gearing and is now rejected.

## Run before building hardware

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e src/arm_lab_model
python -m pytest -q

# Independent equations + FK check; seed, tolerances and model hash in JSON.
robot_test check --config my_arm.yaml --samples 150 --payload 1.0 -o check.json

# Hold a named pose with a torque-limited controller.
robot_test simulate --config my_arm.yaml --pose home --duration 2 -o hold.json

# Quintic motion between two named poses; use your config's pose names.
robot_test simulate --config my_arm.yaml --pose home --target stowed --duration 5 -o motion.json

# Export an inspectable model; optional interactive MuJoCo viewer.
robot_test export --config my_arm.yaml -o arm.xml
python -m mujoco.viewer --mjcf=arm.xml

# Existing ROS-free engineering tools, or their ros2 run equivalents.
spec_report --config my_arm.yaml --json
engineering_report --config my_arm.yaml
```

`robot_test` exits **0** for pass, **1** for failed acceptance criteria, **2**
for invalid inputs or missing dependencies. Reports contain the engine version
and generated-model hash; archive the source YAML alongside them. The generic
MJCF hash covers the root XML only; archive included models and assets too.

The arm motion test is feedforward plus PD with joint-output torque saturation.
When electrical data exist it also applies the voltage/current torque-speed
envelope. It reports tracking error, velocity-limit ratio, saturation fraction,
peak/RMS torque and whether RMS exceeds the continuous rating. A pass requires
finite states, no engine warnings, no saturation, tracking within tolerance,
and velocities within the configured limits. Continuous-rating exceedance is
reported separately; the test does not certify a sustained duty cycle.

Run trajectories at several payloads and durations. Repeat at half the timestep
and check that the conclusions do not change. This benchmark disables contacts
and uses a rigid tool; it neither proves collision-free motion nor validates
grasping. Its gains are benchmark gains, not the deployed ROS controller.

Power estimates now include copper loss using Kt and resistance when available,
plus mechanical power divided by efficiency and electronics overhead. They use
ambient winding temperature and give no regeneration credit. Missing electrical
data use an explicitly labeled heuristic. Validate hot-winding losses, driver
losses, supply limits and regeneration separately.

## Quadrupeds, humanoids and other robots

To validate a new 12-DOF dog-style **configuration** (no simulation):

```bash
PYTHONPATH=src/arm_lab_model python3 -m arm_lab_model.project_config \
  src/arm_lab_model/config/pipeline/project_quadruped_12dof.yaml
```

The example has four three-joint branches and a floating base. Its unknown
inertials and joint limits are listed as missing; the zero origin transforms
are explicit topology placeholders. `runtime_ready` remains false. See the
[extended pipeline guide](EXTENDED_PIPELINE.md) for current component contracts.
For an executable twelve-actuator floating model, use the separate
[synthetic dog demo](DOG12_DEMO.md). Its drop test checks contacts and finite state,
not successful standing or locomotion. The [UR5e workflow](PIPELINE_WORKFLOW.md)
checks nominal FK and controlled arm motion, and produces an explicitly incomplete
benchmark when no measured reference is configured. The eight-joint external MJCF
fixture below is a separate, existing numerical smoke test.

```bash
robot_test smoke examples/quadruped_drop.xml --duration 2 -o quadruped-drop.json
robot_test smoke /path/to/humanoid.xml --keyframe home --duration 2 -o humanoid-smoke.json
```

The supplied quadruped is an intentionally simple floating-base, four-branch
contact fixture with eight passive hinges. It drops onto the floor. A smoke
pass means that stepping completed without numerical warnings or nonfinite
state. Falling over can still pass. Controls are zero unless a keyframe supplies
constant controls. There is no gait, balance policy, task score or motor sizing
report for these external models yet.

The general tree, physical resolution, unified exporters, scenario observations,
trajectory storage and benchmark reports now provide reusable non-arm seams.
Remaining locomotion work includes contact-aware success policies, support/slip
metrics, calibrated contacts, standing/stepping controllers and whole-body balance.
A robot-specific controller remains essential for useful quadruped/humanoid tasks.

## What simulation can establish

Use this sequence: validate units and measured inputs → cross-check dynamics
→ simulate controlled loads and motion → sweep uncertain parameters → test one
real actuator/joint at low energy → compare measured current, temperature and
motion → update parameters → gradually test the complete robot.

Engine agreement demonstrates consistent equations for the chosen model.
It does not measure gearbox backlash, cable routing forces, real contact
friction, motor heating, structural fatigue or hardware safety. Record where
each design input came from and its uncertainty; prioritize those with the
largest effect on load margin.
