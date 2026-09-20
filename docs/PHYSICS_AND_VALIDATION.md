# Physics, experiments and result interpretation

[Documentation index](README.md) · [Practical test workflow](ROBOT_TESTING.md)

## Physical model

The analytical model is a fixed-base serial rigid-body arm. It computes forward
kinematics, a geometric Jacobian, recursive Newton–Euler inverse dynamics and
separate engineering estimates. Those estimates are not all coupled into a
single simulation of electronics, structure, contacts and controls.

| Calculation | Implementation and assumptions |
|---|---|
| Tube mass | Density × `pi * (outer_radius² - inner_radius²) * length`, plus proximal actuator/fitting mass |
| Composite inertia | Hollow-cylinder inertia plus lumped-body approximation and parallel-axis shifts, unless overridden by measured assembly properties |
| World inertia | Rotate the CoM tensor using the moving link orientation |
| Gravity torque | Inverse dynamics at zero velocity/acceleration, with gravity acceleration at the base |
| Full torque | Rigid-body inertial/Coriolis/gravity loads, reflected rotor inertia and optional friction terms |
| Reflected rotor inertia | `rotor_inertia * gear_ratio²`, added to joint acceleration demand |
| Analytical Coulomb friction | `friction * tanh(qd / 0.02)`; smoothed near zero speed |
| Viscous damping | `viscous_damping * qd`, output-side |
| Output peak/continuous torque | Motor rating × gear ratio × efficiency |
| Output no-load speed | Motor speed / gear ratio; actual torque-speed envelope can be more restrictive |
| Payload capacity | Static torque headroom at one pose, with configured reserve; not a thermal or dynamic payload certificate |
| Structural droop | Tube bending/torsion and joint compliance estimates; not deformable-body simulation |
| Power | Mechanical-power magnitude / efficiency + copper loss when specified + overhead; no regeneration credit |

The analytical tool is one rigid box with total tool mass. Measured link
inertials can be asymmetric; tube collision and stiffness approximations remain.
The payload is a point mass at the TCP in analytical dynamics. Native MuJoCo
uses negligible positive rotational inertia for its payload representation.

## What each verification establishes

| Check | Compared quantities | What it does not establish |
|---|---|---|
| Closed-form tube and pendulum tests | Mass/inertia and gravity torque against special cases | Accuracy of real material or motor inputs |
| Jacobian finite differences | FK perturbation versus analytical differential motion | Controller tracking |
| Energy balance | Mechanical joint power versus change in modeled energy | Contact, electronics or thermal losses outside the checked equations |
| KDL check | FK/dynamics from exported URDF versus the model, using a rigid tool | Moving-finger contact or all Gazebo plugins |
| MuJoCo `check` | FK, gravity and full inverse dynamics at seeded states | Contact/friction agreement; constraints and damping are disabled |
| MuJoCo `simulate` | Forward integration under benchmark torque control | ROS-controller behaviour, collision-free motion or sustained duty qualification |
| MuJoCo `smoke` | A supplied MJCF runs without warnings/nonfinite state | Successful standing/walking or a task-specific score |
| ROS speed test | Achieved motion and model-derived loads with ROS control active | Measured actuator torque or constant Cartesian speed along a joint-space spline |

Numerical agreement at machine precision is evidence of consistent equations
and export. Both engines can agree perfectly on incorrect masses supplied by a
user. Preserve measurement provenance and uncertainty alongside the configuration.

## MuJoCo inverse-dynamics check

```bash
robot_test check --config my_arm.yaml --samples 150 --seed 0 --payload 1 -o check.json
```

The random seed fixes sampled states. The API's default joint torque criterion
is `abs(error) <= 1e-7 + 1e-8 * abs(reference)` N·m; TCP error must be below
`1e-9` m. Per-joint tolerances prevent a large shoulder torque from hiding a
wrist discrepancy. Both paths retain reflected rotor inertia. Constraints and
damping are disabled to isolate rigid-body equations, and analytical friction
is excluded. The report records maximum errors, sample count, seed and tolerance.

## MuJoCo motion acceptance

```bash
robot_test simulate --config my_arm.yaml --pose home --target stowed \
  --duration 5 --timestep 0.001 --payload 1 --max-error 0.05 -o motion.json
```

The runner uses a quintic joint interpolation, inverse-dynamics feedforward and
PD feedback. Current benchmark gains are 80 N·m/rad and 15 N·m·s/rad, recorded
in the report. Motor commands are output torque. Clipping uses configured peak
torque, plus the electrical voltage/current envelope when Kt, resistance and
maximum phase current are available. The envelope currently uses its default
20 °C winding temperature; heating is not integrated into this short benchmark.

MuJoCo's Coulomb friction constraint and the analytical smoothed friction term
are different models near zero speed. Small tracking deviations should therefore
be expected under motion even when the friction-free equations agree exactly.

Pass requires all requested steps, finite states, no engine warnings, zero
saturated steps, tracking error within `--max-error`, and a maximum speed ratio
no greater than `1.001`. The step count is `ceil(duration / timestep)`, so actual
simulated duration can exceed the requested time by less than one step.

RMS torque above a continuous rating is reported separately and does **not**
currently force failure. Neither power, winding temperature nor collision
clearance is a motion acceptance criterion. Use the engineering report and a
dedicated task-specific acceptance policy for sustained operation.

## JSON report contracts

| Fields | Meaning |
|---|---|
| `test`, `scope`, `passed` | Which check ran, its limits, and its own verdict |
| `mujoco_version`, `model_sha256` | Engine version and exact generated/root XML identity |
| `samples`, `seed`, `atol_nm`, `rtol` | Inverse-dynamics experiment settings |
| `max_gravity_error_nm`, `max_full_dynamics_error_nm`, `max_tcp_error_m` | Numerical disagreement metrics |
| `requested_duration_s`, `duration_s`, `timestep_s`, `steps` | Requested and executed timing |
| `initial_q_rad`, `target_q_rad`, `final_q_rad`, `joint_names` | Arm trajectory state and array ordering |
| `controller`, `actuator_parameters_sha256` | Benchmark controller settings and actuator parameter identity |
| `max_tracking_error_rad`, `tracking_tolerance_rad` | Worst absolute joint position error and acceptance tolerance |
| `max_speed_limit_ratio`, `saturation_fraction` | Maximum speed/rating ratio and fraction of steps with any clipped joint |
| `peak_torque_nm`, `rms_torque_nm`, `continuous_torque_exceeded` | Per-joint applied torque statistics and continuous-rating flags |
| `warnings` | MuJoCo warning counters; any warning fails the numerical test |
| `nq`, `nv`, `nu`, `max_contacts`, `final_qpos`, `keyframe` | Generic MJCF dimensions/contact activity/final state and initial keyframe |

Fields differ by test. Preserve the source YAML/MJCF, assets, command and Git
commit: hashes identify inputs but do not reconstruct them. The generic MJCF
hash covers only its root XML, not included XML or mesh files. Reports do not
have a versioned external schema yet; consumers should ignore unfamiliar fields
and validate required ones for the selected `test`.

The report CLI uses strict JSON serialization for `robot_test`. Spec-report
serialization is a separate implementation and can include nonfinite numerical
values for unconstrained quantities; strict downstream JSON tools may need an
explicit normalization policy.

## Evidence and experiment design

The 2026-09-19 review of commit `ac73f68` recorded 108 passing tests, KDL/energy
checks, a 150-state MuJoCo comparison and a successful five-second `home` to
`stowed` move. Those are historical results, not a standing certification of
future revisions. CI re-executes its configured checks on each push.

For your own design, test a matrix of payload, pose, motion duration, gravity,
friction, efficiency and CoM uncertainty. Repeat representative trajectories at
half the timestep. Check outcomes and convergence, not just numerical survival.
Then compare model predictions with measured single-joint current, temperature
and motion before making claims about a complete physical robot.

Run the standalone KDL checks in an environment containing `PyKDL`:

```bash
python -m arm_lab_model.verification --samples 200 --config my_arm.yaml
```

Missing KDL is reported as a failed/unavailable verification, not silently
counted as a passed independent-engine check. Pure pytest execution and MuJoCo
tests do not establish that KDL was run.

Source: [dynamics](../src/arm_lab_model/arm_lab_model/kinematics.py),
[verification](../src/arm_lab_model/arm_lab_model/verification.py),
[MuJoCo runner](../src/arm_lab_model/arm_lab_model/mujoco_backend.py),
[actuator model](../src/arm_lab_model/arm_lab_model/actuator_model.py).
