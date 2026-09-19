# Contributing

Use Python 3.12 for the ROS 2 Jazzy-compatible headless workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e src/arm_lab_model
python -m pytest -q
robot_test check --samples 150
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
discuss the [body-tree roadmap](docs/ROBOT_TESTING.md) before extending the
serial-arm assumptions throughout the codebase.
