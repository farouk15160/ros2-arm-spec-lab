# Contributing

Use Python 3.12 for the headless CI workflow. For live ROS tests, use the system
Python of the sourced ROS installation; the unified MoveIt/MuJoCo pipeline has
been exercised on Humble. See [robot run commands](docs/RUN_ROBOTS.md) for setup.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e src/arm_lab_model
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
robot_test check --samples 150
python tools/build_docs.py --check
```

Physics changes should include an independent check: an analytic special case,
energy identity, finite differences or comparison with an independent engine.
Exercise nonzero offsets, rotated frames and asymmetric tensors, not just the
default arm. Preserve existing YAML behaviour or document a migration.

Document SI units, reference frames, motor/output conventions, assumptions and
the exact scope of a test pass. Keep MuJoCo optional and pure model imports free
of ROS and GUI dependencies. Do not label simulation or historical runs as
hardware validation. Include the config, seed, engine version and timestep with
reported results. Run ROS/Gazebo checks separately when changing their code.

For bug reports include the smallest reproducing YAML/MJCF, command, Python and
engine versions, expected behaviour and actual output. For new robot types,
use the [shared tree model](docs/EXTENDED_PIPELINE.md) and
[workspace modeling skill](skills/arm-lab-modeling/SKILL.md); fixed, branched and
floating examples already exist. Use [agent instructions](AGENTS.md) and
[the validation skill](skills/arm-lab-validation/SKILL.md) for ownership and checks.
Keep legacy serial-arm interfaces distinct from general project manifests.

The default suite excludes live ROS tests unless their opt-in flags are enabled.
Report those skips separately from executed tests. Reproduce changes against the
declared dependency range when library compatibility changes; a passing local
NumPy 1.x environment does not establish compatibility with fresh CI dependencies.

Architecture, domain models, ROS interfaces and implementation guidance are in
the [documentation index](docs/README.md) and [developer guide](docs/DEVELOPMENT.md).
When changing command entry points, launch arguments, core APIs or the example
YAML, run `python tools/build_docs.py` and commit the regenerated references.
Edit diagram sources in `docs/diagrams/`; the same command updates their embeds.
Narrative guides must be reviewed alongside the source change.
