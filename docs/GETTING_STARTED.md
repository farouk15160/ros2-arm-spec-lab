# Getting started

[Documentation index](README.md)

## 1. Choose your workflow

For UR5e or dog configurations, follow [Run different robots](RUN_ROBOTS.md),
including the tested Humble MuJoCo/RViz pipeline and literal 1-DOF dog commands.
The Jazzy/Gazebo instructions below describe the original serial-arm workflow.

For model calculations and MuJoCo tests, use Python 3.12 and the headless setup
below. For the editor, live dashboard, RViz and Gazebo, use the ROS 2 Jazzy
workspace. The shipped ROS configuration targets Gazebo Harmonic. The two
workflows share design data, but run different controllers and simulations.

All commands below start from the repository root. Examples use Bash; use
`setup.zsh` instead of `setup.bash` when sourcing ROS from Zsh.

## 2. Install the headless tools

```bash
git clone https://github.com/farouk15160/ros2-arm-spec-lab.git
cd ros2-arm-spec-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e src/arm_lab_model
```

This installs the development/test dependencies, including MuJoCo. A smaller
analytical-only installation can use `python -m pip install -e src/arm_lab_model`;
MuJoCo is optional for model loading and analysis. The package extra
`python -m pip install -e 'src/arm_lab_model[simulation]'` adds the engine without
the full development dependency list.

The kinematics package also contains pure Python analysis code, available from
the source checkout or a ROS build; the documented headless installation above
installs the model package's executables only.

## 3. Run the shipped arm

```bash
python -m pytest -q
robot_test check --samples 150 --output check.json
robot_test simulate --pose home --duration 2 --output hold.json
robot_test simulate --pose home --target stowed --duration 5 --output motion.json
robot_test export --output arm.xml
robot_test smoke examples/quadruped_drop.xml --duration 2 --output drop.json
```

Expect `passed: true` for these benchmark examples in the tested environment.
Read the report's `scope` field: a passive quadruped drop is a numerical/contact
fixture, not a demonstration of walking. `robot_test` returns 1 when acceptance
criteria fail and 2 for invalid input. Do not hide those exit statuses in scripts.

`robot_test export` only writes XML. To inspect it interactively on a desktop:

```bash
python -m mujoco.viewer --mjcf=arm.xml
```

The viewer is separate from the benchmark's feedforward/PD controller. Exporting
and opening an arm does not automatically run the holding or motion test.

## 4. Create your design

```bash
cp src/arm_lab_model/config/arm_config.yaml my_arm.yaml
```

Edit the copy using the [configuration guide](CONFIGURATION.md). Start with
dimensions, material, actuator selection, joint placement and tool mass. Enter
measured assembly mass, CoM and inertia when available. Keep `spec_targets`
aligned with your own requirements; they are not inferred from geometry.

```bash
robot_test check --config my_arm.yaml --payload 1 --output my-check.json
robot_test simulate --config my_arm.yaml --payload 1 --duration 2 --output my-hold.json
spec_report --config my_arm.yaml --json
engineering_report --config my_arm.yaml
```

A spec report failure means a requirement was missed, not necessarily a broken
installation. An engineering report currently prints findings and returns 0;
read its contents instead of treating exit status as engineering acceptance.

## 5. Build the ROS workspace

Use a separate shell with the ROS installation available. Avoid activating an
isolated Python virtualenv that hides system ROS/Qt packages.

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch arm_lab_bringup view.launch.py
```

`rosdep` requires ROS and rosdep to have been installed and initialized already.
Dependency declarations are in each package's `package.xml`. `view.launch.py`
opens RViz and joint sliders; it does not run dynamics.

```bash
ros2 run arm_lab_gui config_editor --config my_arm.yaml
ros2 launch arm_lab_bringup sim.launch.py config_file:="$PWD/my_arm.yaml"
```

For a simulation with no desktop windows:

```bash
ros2 launch arm_lab_bringup sim.launch.py \
  config_file:="$PWD/my_arm.yaml" gz_gui:=false rviz:=false dashboard:=false
```

To change physical payload or gravity, pass explicit launch arguments:

```bash
ros2 launch arm_lab_bringup sim.launch.py \
  config_file:="$PWD/my_arm.yaml" payload_mass:=1.0 gravity:=3.72
```

The gravity argument patches the world file as well as the analytical model.
Editing only `environment.gravity` does not patch an arbitrary Gazebo world.
See [ROS interfaces](ROS_INTERFACES.md) and the [runbook](RUNBOOK.md) before
comparing live estimates with simulator outputs.
