---
name: arm-lab-modeling
description: Add or edit robot configurations, STL/material properties, inertials, topology or shared URDF/MuJoCo exports in Robot Design Lab. Use for serial arms, fixed test rigs and branched or floating robots.
---

# Add or change a robot model

Read [configuration contracts](../../docs/EXTENDED_PIPELINE.md) and
[physical modeling](../../docs/PHYSICAL_ROBOT.md) for the format being changed.
Use [run profiles](../../docs/RUN_ROBOTS.md) to select a comparable working model.

1. Identify the intended robot family, actuated joint count, fixed/floating base,
   reference frames and evidence source. Count a floating base's six velocity
   coordinates separately from actuator DOF. An unspecified vendor name is not
   enough evidence to invent its geometry or ratings.
2. Prefer a new `robot_<name>.yaml`, `project_<name>.yaml` and
   `scenario_<name>.yaml` under `src/arm_lab_model/config/pipeline/`. Reuse shared
   component files and the authoritative robot tree across run profiles. Use the
   legacy adapter when the source remains an `ArmConfig` serial tube arm.
3. Define geometry units/transforms and choose homogeneous density-derived
   inertials or complete manual assembly inertials. STL integration needs a closed,
   consistently oriented surface. Manual COM and tensor orientation must match
   the declared link frame; see the physical guide for tensor ordering. Missing
   or invalid inputs should produce an actionable error.
4. Exercise `load_project -> resolve_robot -> physical_report`, then both
   `build_robot_urdf` and `build_robot_mjcf`. Extend the strict loader and its public
   tests before adding a new key. Update `tools/build_docs.py` if introducing a new
   public module; configuration files in the pipeline directory are packaged by
   the model package's setup script.
5. Test an independent mass/COM/inertia or gravity-torque calculation and an actual
   MuJoCo motion/contact case. Check moving DOF/actuators, transforms, contacts,
   saturation and output evidence. If a fixture intentionally fails trajectory
   acceptance, assert the reason and explain its exit status.
6. For MoveIt, declare each planning chain, its controller and explicit acceleration
   policy. General simulation/RViz can run with MoveIt disabled. Floating-base
   balance and gait control are separate features, not consequences of exporting
   a legged topology.
7. Use [validation](../arm-lab-validation/SKILL.md), document commands and remaining
   assumptions, and link the example from the documentation index/run guide.

## Ownership map

- Declarations and topology: `project_config.py`, `config_contract.py`,
  `robot_topology.py`, `pipeline_options.py`.
- Physical properties and tree transforms: `mesh_physics.py`, `physical_robot.py`.
- Shared exports and legacy conversion: `robot_export.py`, `robot_legacy.py`.
- Execution: `joint_trajectory.py`, `pipeline_runtime.py`, `pipeline_cli.py`.
- Reference provenance: [UR5e guide](../../docs/UR5E_REFERENCE.md); preserve pinned
  snapshots/hashes instead of silently changing data to match a simulation.

Completion means a valid profile, independently checked physical quantities,
a runnable scenario with asserted acceptance semantics, packaged configuration,
and instructions another person can execute. Describe a synthetic approximation
as such; name unsupported dynamics explicitly.
