# Physical robot model and exports

## Step 3: physical inputs

`resolve_robot(project)` resolves the project robot to a copied, complete tree.
Unknown inertial values fail with the link name; zero is never substituted for
unknown data. Named materials come from the project's materials YAML, and inline
`material: {name: aluminum, density: 2700}` is also accepted.

```yaml
links:
  - name: upper_arm
    geometry:
      type: mesh
      path: meshes/upper_arm.stl
      units: mm
      scale: [1, 1, 1]
      origin: {xyz: [0, 0, 0], rpy: [0, 0, 0]}
    material: aluminum_6061
    inertial: {mode: auto}
```

Mesh paths are relative to the robot YAML. Units are mandatory (`m` or `mm`);
scale is positive in each axis. ASCII and binary STL are supported without an
extra mesh library. Signed tetrahedral integration computes volume, COM and full
inertia tensor. Edge pairing rejects open/nonmanifold meshes and inconsistent
winding. Surface area is also reported. The origin rotates the inertia into link
axes and translates COM. A homogeneous closed solid is assumed; STL cannot infer
hollow parts, motors, fasteners or density variation. Self-intersection detection
is not certified; use a validated CAD solid. Overlapping disconnected shells must
be repaired in CAD before automatic mass calculation.

Primitives use analytic formulas: `type: box` + `size: [x,y,z]`,
`type: cylinder` + `radius` and `length` (local Z axis), or `type: sphere` + `radius`.
All dimensions are metres. For complete CAD/assembly properties use:

```yaml
inertial:
  mode: manual
  mass: 3.2
  com: [0.1, 0, 0]
  inertia: [0.02, 0.03, 0.04, 0, 0, 0]
  source: Measured assembly/CAD report identifier
```

Tensor ordering is **Ixx, Iyy, Izz, Ixy, Ixz, Iyz**, about COM expressed in link
axes. Manual mode requires the complete tuple, preserves it exactly, and does not
estimate a replacement from material density. Manual volume is reported unknown.
Mass must be positive and the tensor positive definite with physical principal
moment triangle inequalities. Fixed coordinate frames may explicitly declare
zero mass, zero COM and zero tensor; movable bodies may not be massless.

`physical_report(robot, q=None, gravity=(0,0,-9.81), payload=None)` computes link
weight, robot mass and world COM, joint gravity compensation effort, and the base
support wrench. `q` is a joint-name mapping or an ordered vector (radians/metres).
Revolute and prismatic joints and branching trees are supported. Floating-base
pose can be passed as `q['__base__']={xyz: [...], rpy: [...]}`; the static report
assumes the base is externally supported, not free-falling. Payload has
`{link, mass, com}` and is a point mass for static loads only. Dynamic analysis
needs full inertia, velocity and acceleration and uses the MuJoCo runtime.

Validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_model/test/test_mesh_physics.py \
  src/arm_lab_model/test/test_physical_robot.py
```

Tests independently check solid cube volume/mass/COM/inertia, unit conversion,
rotated anisotropic tensors, rejection of open meshes/invalid densities,
manual overrides, unknown inputs, and analytic pendulum gravity plus payload.
These are analytical reference checks, not measured hardware benchmarking.

## Step 4: global robot exports

`robot_export.build_robot_urdf(robot, ros2_control=False, ...)` and
`robot_export.build_robot_mjcf(robot, simulation=None, environment=(), sensors=())`
consume the **same resolved tree**. The generated URDF is the unified description;
no manually maintained duplicate Xacro is needed. Both formats preserve the
COM and all six tensor components, including off-diagonal terms. Geometry origins
and explicit mesh SI scaling are shared. The legacy tube configuration is adapted
through `robot_legacy.py`, preserving the previous exporter's measured/equation
inertials, tool bodies and finger links. Its artificial tiny TCP mass is removed
and replaced with an explicit massless coordinate frame.

Fixed-base URDF includes a world mount using optional `base.origin: {xyz, rpy}`.
Floating robots retain their own root link with no world weld; MJCF adds a free
joint (seven position coordinates, six velocities). Branched limbs, continuous
and prismatic joints work through the same tree traversal. Fixed auxiliary frames
are retained in MuJoCo for tool/camera frame lookups. Missing joint effort or speed
prevents URDF control export; MJCF permits absent effort for inverse-dynamics-only
analysis, while controlled runtime must require effort bounds.

Optional URDF control tags use `mock_components/GenericSystem` by default and
position command/state interfaces. Select `gz_ros2_control/GazeboSimSystem` with
`controllers_file` only for a configured Gazebo deployment. These exports do not
connect to real hardware. Environment objects feed MJCF static collision bodies;
cameras feed URDF mount and optical frames and MJCF camera elements.

MuJoCo STL contacts use convex-hull collision approximation. Split concave shapes
into convex components when their concavities matter to contacts; visual geometry
alone is not evidence of exact contact geometry. URDF mesh collision behavior
depends on the downstream collision engine. Unspecified shapes remain unspecified,
and no full collision-checking claim is made for incomplete link geometry.

`robot_fingerprint(robot)` hashes canonical configuration plus referenced mesh
contents, for associating replay and benchmark records with the physical model.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_model/test/test_robot_export.py
```

Exporter tests compare mass and COM in URDF/MJCF, MuJoCo pendulum gravity against
`m*g*l`, floating-base DOF counts, prismatic loads, static obstacles, and legacy
arm gravity against the independent analytic Newton–Euler implementation.
