# Environment, cameras and perception (Steps 8–10)

The project manifest selects environment, sensors and perception files. These
components share the global robot tree and SI transforms. Set their `enabled`
flags to activate them. Examples are in
`src/arm_lab_model/config/pipeline/{environment,sensors,perception}_example.yaml`.
The camera example mounts to UR5e `wrist_3_link`; change this to an existing link
when using another robot. No camera model is assumed to be factory calibrated.

## Step 8: Environment

`scene_config.validate_environment()` checks the component contract and
`resolve_environment(project)` resolves mesh paths relative to its YAML file.
Each object requires `name`, `geometry`, `pose`, `frame: world`, `static`, and
`mass`. Poses contain `xyz` in metres and `rpy` in radians. Static objects may
have `mass: null`; moving bodies require positive mass. Primitive geometry is
`{type: box, size: [x,y,z]}`, `{type: sphere, radius: r}`, or
`{type: cylinder, radius: r, length: h}`. An STL example:

```yaml
schema_version: 1
enabled: true
objects:
  - name: machine
    geometry:
      type: mesh
      path: ../../robot_geometry/machine.stl
      units: mm
      scale: [1, 1, 1]
    pose: {xyz: [1.5, 0, 0], rpy: [0, 0, 1.57]}
    frame: world
    static: true
    mass: null
```

`robot_export.build_robot_mjcf(..., environment=objects)` uses these poses and
geometries in physics. `collision_objects(objects)` produces triangle meshes or
primitives for MoveIt. The `pipeline_scene` ROS node waits for
`/apply_planning_scene`, applies the world, checks the response, and publishes
latched `/pipeline/environment_markers` for RViz. A service error is reported
and retried. Missing STL paths, implicit STL units and invalid dimensions fail
validation.

MuJoCo uses a convex hull for mesh contacts. Supply separate convex objects for
concave machines and fixtures. MoveIt receives the actual triangle mesh. Dynamic
objects are rejected by the current global exporter and MoveIt scene adapter:
a live pose updater and validated dynamic-body support are required first.
The target node remains gated until a valid static scene is successfully applied.

## Step 9: Cameras

Camera mappings include type `rgb`, `depth` or `rgbd`, descriptive `model`,
`parent_link`, transform, `[width,height]` resolution, `frame_rate_hz`, calibrated
`intrinsics: {fx,fy,cx,cy}`, clipping `[near,far]` and Gaussian depth noise
`noise_stddev` in metres. Optional `fov_y_deg` must agree with resolution/fy.
The ROS mounting convention is x forward, y left, z up. Generated optical frames
use x right, y down, z forward. The exporter applies the additional OpenGL camera
rotation for MuJoCo. Asymmetric focal lengths and noncentral principal points
are preserved, including the half-pixel conversion to OpenGL.

`CameraRenderer(model, cameras, seed)` owns MuJoCo rendering contexts. Call
`render(data, name)` on the context's creating thread and `close()` at shutdown.
The exporter sizes the offscreen framebuffer for the largest configured camera,
including HD resolutions.
The frame has RGB uint8 and/or metric depth float32, intrinsics, simulation stamp
and optical frame ID. Invalid/clipped depth is NaN. MuJoCo shared RGB render clipping is set to the
minimum near and maximum far plane of all configured cameras, in metres;
per-camera depth clipping is applied afterward. Large scene extents therefore
do not silently hide nearby targets. Different per-camera RGB clipping planes
are not supported because MuJoCo stores them globally. Noise uses a reproducible RNG.
The simulation bridge schedules frames at each configured rate and publishes:

- `/sensors/<name>/rgb/image_raw`: `rgb8`
- `/sensors/<name>/depth/image_raw`: `32FC1` metres
- `/sensors/<name>/camera_info`: pinhole calibration

Camera optical/mount transforms are part of the global URDF. These are ideal
pinhole cameras: rolling shutter, lens distortion, stereo occlusion, RGB/depth
extrinsic mismatch and hardware-specific noise are not modeled. Camera `model`
is descriptive metadata; selecting a hardware name does not invent calibration.
Headless rendering requires an OpenGL backend, for example `MUJOCO_GL=egl`.

## Step 10: Perception

The `pipeline_perception` node validates camera/module dependencies before
starting. It subscribes to Image and CameraInfo, rejects calibration frame or
resolution mismatches, and transforms depth points using TF at the image stamp.
A bounded queue waits up to one second for matching TF, avoiding false future
extrapolation failures while preserving the measurement timestamp.
Supported image inputs include padded rows, big endian depth, `16UC1` millimetres,
`32FC1` metres and `rgb8`.

`perception.depth_to_pointcloud()` implements calibrated pinhole unprojection.
`filter_pointcloud()` removes invalid points, transforms, crops and computes
voxel centroids. `/perception/points` is published in the configured target frame.
Point-cloud range and voxel filtering are configurable; the lower-level API also
supports axis-aligned bounds.

When OctoMap is enabled, `moveit_octomap_config()` configures MoveIt's real
`occupancy_map_monitor/PointCloudOctomapUpdater`. A separate
`/perception/octomap_points` cloud retains its optical frame so the updater uses
the sensor origin for ray casting. MoveIt must have the occupancy updater plugin
installed and matching image-stamp TF. This project does not replace OctoMap with
a local occupancy approximation.

`slam`, `object_recognition` and `image_policy` have explicit plugin interfaces.
Each enabled module requires `camera` and `plugin: python_module:callable` with
optional `parameters`. The callable receives `(frame_bundle, parameters)` and
returns JSON-serializable output published on `/perception/results`. The bundle
contains `stamp`, `frame_id`, `intrinsics` and either `rgb` or `depth`. RGB modules
receive RGB frames; SLAM receives depth frames. Plugins own their state and may
publish additional ROS outputs. Missing imports/callables fail clearly. A SLAM
implementation, trained semantic recognizer or learned policy is not bundled;
install/configure an implementation explicitly. RGB-D synchronized SLAM requires
an adapter that handles synchronization; the basic callback is per stream.
Plugin configuration is executable local Python and must be trusted.

The optional `builtin:color_components` recognizer detects connected thresholded
RGB regions and returns pixel bounding boxes. It is a simple testable baseline,
not semantic object recognition or 3D pose estimation. Image-policy results are
observations on the results topic; they do not command the robot automatically.

## Tests and commands

From the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_model/test/test_scene_perception.py
source /opt/ros/humble/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  src/arm_lab_gui/test/test_pipeline_perception.py
```

The first suite checks known-plane unprojection, depth units, voxel centroids,
rigid transforms, malformed configuration, dependency failures, STL unit scaling,
and an actual EGL/MuJoCo projection against an analytical asymmetric camera.
EGL-only validation is skipped with an explicit reason when OpenGL is unavailable.
The ROS adapter suite checks endian/stride handling, collision/marker consistency
and quaternion transforms. It is skipped outside a sourced ROS environment.

After building/sourcing the workspace, adapters can also be started separately:

```bash
ros2 run arm_lab_gui pipeline_scene --ros-args -p project_file:=/absolute/project.yaml
ros2 run arm_lab_gui pipeline_perception --ros-args -p project_file:=/absolute/project.yaml
ros2 topic echo /perception/points --once
```

Use a manifest selecting enabled example components and the matching robot;
start simulation, robot_state_publisher and MoveIt with the same manifest first.
See the pipeline/MoveIt guide for launch commands.

References: [MuJoCo camera XML](https://mujoco.readthedocs.io/en/latest/XMLreference.html)
and [MoveIt perception pipeline](https://moveit.picknik.ai/humble/doc/examples/perception_pipeline/perception_pipeline_tutorial.html).

## Runnable example and live integration check

The manifest `config/pipeline/project_ur5e_perception.yaml` selects the UR5e,
static workbench/fixture, wrist RGB-D camera, point clouds and OctoMap. From the
repository root after building/sourcing the workspace:

```bash
ros2 launch arm_lab_bringup pipeline.launch.py \
  project_file:="$PWD/src/arm_lab_model/config/pipeline/project_ur5e_perception.yaml" \
  rviz:=false
# In a second sourced terminal on the same ROS domain:
python3 tools/check_perception_live.py
```

The live check subscribes to actual RGB, depth, CameraInfo and both cloud topics,
then queries MoveIt's actual planning scene. It requires nonempty OctoMap data,
the configured workbench/fixture and correctly framed, nonempty point clouds.
On 2026-09-25 with ROS Humble and MuJoCo 3.13 EGL it passed with 28,670 finite
depth pixels, 756 points in each cloud, 700 OctoMap bytes and both collision
objects. An additional actual EGL render passed at 1280x720. These are simulation
integration checks, not validation of real-camera calibration or hardware noise.
The occupancy updater currently limits updates to 5 Hz.
