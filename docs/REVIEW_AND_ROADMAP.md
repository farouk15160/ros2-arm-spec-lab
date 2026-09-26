# Review of the robot design bench

Updated: 2026-09-26. The initial 2026-09-19 findings below cover serial-arm
source, headless tests, KDL and MuJoCo execution. The subsequent general pipeline
also has live Humble MoveIt, direct joint-action and perception evidence described
in the linked guides. No physical robot validation is claimed.

## General pipeline delivered

The workspace now provides a strict general robot tree, STL/material physical
properties, shared URDF/MJCF export, optional MoveIt planning, direct simulation
without MoveIt, trajectory storage/replay, static worlds, RGB-D/cloud/OctoMap
integration, and reference comparison reports. UR5e, fixed 1-DOF dog and floating
12-DOF dog profiles have runnable examples in [the run guide](RUN_ROBOTS.md).
Coding-agent instructions and workspace skills describe extension ownership and
test selection. [The stage reports](PIPELINE_WORKFLOW.md) document evidence scope.

Remaining general-pipeline work is specific:

- Add independently acquired, calibrated motion traces for hardware comparisons.
  The UR5e catalogue currently supplies manufacturer parameters and analytical FK.
- Replace approximate collision primitives with validated CAD when physical
  clearance matters, and identify actuator/friction/contact parameters.
- Add balance, gait and contact-aware task acceptance for legged robots. The dog1
  fixture is fixed; the dog12 contact test intentionally fails generic trajectory
  acceptance rather than claiming locomotion.
- Supply live object-pose tracking before enabling dynamic planning environments.
  Current world objects are static.
- Integrate selected SLAM and learned perception/control implementations through
  the existing plugin interfaces. Interfaces do not constitute trained algorithms.

## Findings addressed

| Finding | Effect | Change |
|---|---|---|
| Joint offset translated after applying its own frame rotation | Analytical geometry disagreed with URDF for nonzero offsets and rotated origins | Apply parent-frame translation first; regression test and asymmetric MuJoCo comparison |
| Tube-only mass properties; measured input silently ignored | CAD/mass measurements could not drive the equations | Complete measured assembly mass, off-axis CoM and full tensor, used by dynamics and both exports |
| Invalid physical inputs accepted | Negative/NaN parameters could propagate into plausible-looking reports | Loader checks physical ranges, vector finiteness, tensor realizability and supported joint types |
| Gazebo damping guessed from Coulomb friction | Exported and analytical loss models differed | Explicit output-side viscous damping, default zero, in dynamics and exports |
| Existing MuJoCo check failed but returned success | An automated job could report success for inconsistent dynamics | Direct native MJCF exporter, per-joint tolerances, deterministic sampling, nonzero failure exit |
| Holding power based on an overhead heuristic despite available motor data | Current and copper loss were not reflected in the main power estimate | Kt/resistance-based copper loss when specified, with labeled fallback |
| Test discovery imported ROS GUI modules | Headless CI required ROS/Qt | Explicit test discovery, independent path helpers, optional MuJoCo tests |
| Prismatic option reused torque and rotary transmission units | Force sizing could be dimensionally wrong | Reject in the arm schema; support external MJCF as a separate workflow |

Baseline before changes: 71 analytical/kinematics tests passed. The legacy
URDF-import MuJoCo script reported a maximum full-dynamics discrepancy of
approximately 0.126 N·m over 150 states while returning exit code 0. The native
MJCF route now agrees at floating-point precision for the shipped arm, and is
also tested with an asymmetric tensor, rotated mounting, nonzero joint offset
and payload. This establishes the new route; it is not a claim that every
third-party URDF importer is equivalent.

## Remaining limits, in priority order

1. **Hardware fidelity.** Default actuator/material data are design assumptions,
   not measurements of a completed robot. Enter measured complete assembly
   inertials and identify friction, thermal constants and backlash experimentally.
2. **Actuator fidelity across engines.** Native MuJoCo includes reflected rotor
   inertia. The existing Gazebo integration does not explicitly model the same
   transmission/rotor dynamics. Its controller and physics settings require a
   separate dynamic benchmark. Position control cannot establish torque capacity.
3. **Tool and contact models.** The analytical and native MuJoCo paths represent
   the complete tool as a rigid box. Gazebo can use separate moving fingers,
   whose full inertia/contact behaviour is not covered by that comparison.
   Successful grasping and friction estimates need dedicated validation.
4. **Structural and electrical estimates.** Beam stiffness remains tube-based
   even with measured mass properties. Lumped thermal/duty estimates, regeneration
   calculations and engineering margins are screening calculations, not FEA or
   a validated drive/battery simulation. Datasheet Kt/current conventions matter.
5. **Acceptance scope.** A short motion test can pass while its RMS torque exceeds
   a continuous rating; that condition is reported separately. Test duration,
   winding temperature, controller gains and task tolerances must match the job.
6. **General robot design UI.** The legacy editor, IK and dashboard remain serial
   arm tools. General body trees are configured in the separate pipeline YAML;
   a general tree editor, contact-aware task scoring and whole-body balance remain
   extensions to that working pipeline.

See [the workflow](ROBOT_TESTING.md) for commands, units, input conventions,
pass criteria and the staged extension plan.
