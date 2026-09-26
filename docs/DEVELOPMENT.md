# Development and extension guide

[Documentation index](README.md) · [Contributing](../CONTRIBUTING.md)

Coding agents start at [AGENTS.md](../AGENTS.md). Workspace
[modeling](../skills/arm-lab-modeling/SKILL.md) and
[validation](../skills/arm-lab-validation/SKILL.md) skills route robot additions,
physical edits, test selection and evidence reporting. Keep these pointers
current when moving public modules or changing the run workflows.

## Local development

Use the headless setup in [Getting started](GETTING_STARTED.md). The root pytest
configuration includes the model, kinematics and GUI-schema tests and excludes
runtime scripts named like tests. Schema tests use ROS-free path helpers; the
actual Qt application still needs ROS/Qt for interactive validation.

```bash
python -m pytest -q
python tools/build_docs.py --check
robot_test check --samples 150
```

MuJoCo tests skip when the optional engine is unavailable; a green suite with
skips is not evidence that its engine comparisons ran. CI installs MuJoCo and
runs the explicit cross-check and holding benchmark. KDL, desktop GUI, Gazebo
and real hardware validation remain separate from the headless CI workflow.

## Change map

| Change | Implementation locations | Validation |
|---|---|---|
| Add a physical input | `config.py`, example YAML, GUI `schema.py`/editor as appropriate | Loader boundaries, propagation through calculations and exports |
| Change frames or dynamics | `kinematics.py`, exporters | Nonzero offsets, rotated mounts, asymmetric inertials, energy/KDL/MuJoCo checks |
| Add an actuator behaviour | `actuator_model.py`, config, consumers | Units, corner cases, physical special case; state which simulator applies it |
| Add a benchmark | `mujoco_backend.py` or a separate module | Independent expected outcome, failed acceptance path, JSON and exit status |
| Change ROS control | Controller builder, URDF plugin tags, launch, node interfaces | Build, active controller state, recorded motion in the intended interface |
| Add an editor field | GUI schema/table or a custom dialog | Round-trip config, meaningful validation, offscreen/interactive Qt check |
| Add docs | Narrative Markdown or diagram `.mmd` | Regenerate references/embeds, verify local links, inspect diagrams |

Public implementation signatures are in [API reference](API_REFERENCE.md).
They are useful integration points but not a promised stable/versioned API.
Prefer building adapters at package boundaries to relying on private helpers.

## Small Python integration example

Run in the editable model environment:

```python
import numpy as np
from arm_lab_model.config import load_config
from arm_lab_model.kinematics import ArmModel
from arm_lab_model.mujoco_backend import crosscheck

cfg = load_config()  # pass a path for your own design
model = ArmModel(cfg)
q = model.resolve_pose('home', strict=True)
frames = model.frames(q)
torque = model.inverse_dynamics(q, np.zeros(cfg.dof), np.zeros(cfg.dof),
                                payload=1.0, fs=frames)
print('TCP in world metres:', frames.tcp)
print('Joint load torques in N.m:', torque)
assert crosscheck(cfg, samples=10, payload=1.0)['passed']
```

Rebuild `ArmModel` after modifying a config; it caches per-joint values. Do not
reuse a `FrameSet` for a different q or configuration. Explicit payload arguments
make the experiment unambiguous.

## Documentation maintenance

The generator owns `COMMANDS.md`, `CONFIG_VALUES.md` and `API_REFERENCE.md`.
Edit their source-of-truth code/YAML, not generated Markdown. It also replaces
only marked diagram blocks in narrative pages, leaving authored text intact.
Edit diagram files in `docs/diagrams/` and run:

```bash
python tools/build_docs.py
python tools/build_docs.py --check
```

The skill-guided documentation workflow uses source-derived commands, inputs,
types and launch arguments to reduce drift. CI runs the freshness/link check.
CI also executes the UR5e `robot_pipeline reference-check` command and uploads
its nominal-reference JSON with the physics artifacts. This checks analytical
kinematics, not physical robot measurements.
The checker validates relative file destinations, not URL availability or
Markdown anchors. Diagram sources are Mermaid; GitHub renders embedded copies.
When changing diagram syntax, render with a Mermaid-compatible preview as well.

No repository-wide formatter, lint or pre-commit enforcement is currently
configured. Follow the surrounding Python style and run `git diff --check`.
Do not claim an unconfigured quality gate has run.

## Extending beyond serial arms

The [project loader and topology contract](EXTENDED_PIPELINE.md) provide the
versioned declaration boundary. Extend a component's strict schema and tests
together. Preserve legacy YAML through `robot_legacy`; do not route a branched
body tree through serial `ArmModel` assumptions. Both unified exporters consume
`resolve_robot` results, so masses, COM, tensors and mesh transforms stay shared.

Useful test seams are `load_project`, `resolve_robot`, `physical_report`,
`build_robot_urdf`, `build_robot_mjcf`, `RobotSimulation`, `run_trajectory`,
`save_trajectory`, `compare_runs` and `write_benchmark_report`. Pure model and
configuration tests run without ROS; actual action-message/adapter checks need
ROS sourced. The live MoveIt check is opt-in and executes simulated motion:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
python3 tools/build_docs.py --check
# Launch the pipeline first in a matching ROS_DOMAIN_ID, then:
ARM_LAB_LIVE_PIPELINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_kinematics/test/test_pipeline_planning.py -k live
```

See [planning](PIPELINE_PLANNING.md) for the ROS environment and
[the workflow](PIPELINE_WORKFLOW.md) for actual evidence boundaries. A skipped
ROS/OpenGL test is unavailable evidence, not a passing integration check.

Further legged-robot work needs standing/stepping controllers, foot-contact
metrics, calibrated friction, slip/support evaluation and whole-body balance.
The [synthetic dog fixture](DOG12_DEMO.md) validates topology, physics conversion
and numerical contact behavior; its contact-rich drop correctly fails generic
collision-free trajectory acceptance. Add a contact-aware task policy before
using trajectory success as a locomotion score.
