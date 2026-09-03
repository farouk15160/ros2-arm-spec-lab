"""Fourth-engine cross-check: MuJoCo against the analytical model.

MuJoCo is not packaged for the ROS Python, so run it from its own environment:

    python3 -m venv ~/venvs/mujoco
    ~/venvs/mujoco/bin/pip install mujoco pyyaml numpy
    PYTHONPATH=src/arm_lab_model ~/venvs/mujoco/bin/python tools/mujoco_crosscheck.py

STATUS -- read this before trusting the output.

MuJoCo agrees with the model on gravity torque (1e-5 N.m) and on the Coriolis
terms (4e-6 N.m), and its imported body masses, centres of mass and inertia
tensors match the URDF to about 1e-6. It disagrees on the mass-matrix term by
roughly 2.5e-3 relative, and the disagreement grows toward the base.

That discrepancy is NOT resolved. What is known:

  - KDL, fed the same URDF, agrees with the model to 1.9e-8 on the full
    dynamics, and Gazebo agrees to 0.0 %, and the Lagrangian energy balance
    closes to 7e-9. Three independent checks say the model is right.
  - Comparing the inertia tensors in what this script assumes is the link frame
    shows MuJoCo placing a tube's axial inertia on a different axis. That
    comparison may itself be wrong, because MuJoCo does not necessarily use the
    URDF link frame as its body frame.

So the likeliest explanation is the URDF-to-MJCF frame conversion rather than
either dynamics implementation. The clean way to settle it is to generate MJCF
directly instead of importing URDF, which removes the conversion entirely.
Until then, treat this script as an open question, not as a verification.
"""

import re, sys
import numpy as np
sys.path.insert(0, 'src/arm_lab_model')
import mujoco
from arm_lab_model.config import load_config
from arm_lab_model.kinematics import ArmModel
from arm_lab_model.urdf_builder import build_urdf
from arm_lab_model.verification import _fingerless

cfg = _fingerless(load_config('src/arm_lab_model/config/arm_config.yaml'))
model = ArmModel(cfg)

urdf = build_urdf(cfg, fixed_to_world=False)
# MuJoCo's URDF reader does not know the ROS extension tags; strip them.
urdf = re.sub(r'<gazebo>.*?</gazebo>', '', urdf, flags=re.S)
urdf = re.sub(r'<gazebo [^>]*>.*?</gazebo>', '', urdf, flags=re.S)
urdf = re.sub(r'<ros2_control.*?</ros2_control>', '', urdf, flags=re.S)
# MuJoCo needs a compiler hint to treat the URDF inertials as authoritative.
urdf = urdf.replace('<robot name=', '<robot name=', 1)
urdf = urdf.replace('>', '>', 1)
insert = '  <mujoco><compiler discardvisual="true"/></mujoco>\n'
idx = urdf.index('>', urdf.index('<robot')) + 1
urdf = urdf[:idx] + '\n' + insert + urdf[idx:]
open('/tmp/arm_mj.urdf', 'w').write(urdf)

mj = mujoco.MjModel.from_xml_path('/tmp/arm_mj.urdf')
data = mujoco.MjData(mj)
mj.opt.gravity[:] = [0, 0, -cfg.gravity]
# Pure inverse dynamics: no contacts, no joint-limit constraints, no friction
# loss. Otherwise mj_inverse returns the constraint forces too, which are the
# solver's business and not a property of the rigid-body model.
mj.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT
                        | mujoco.mjtDisableBit.mjDSBL_LIMIT
                        | mujoco.mjtDisableBit.mjDSBL_EQUALITY
                        | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
mj.dof_damping[:] = 0.0
mj.dof_frictionloss[:] = 0.0
mj.dof_armature[:] = 0.0
print('MuJoCo model: %d dof, %d bodies' % (mj.nv, mj.nbody))

names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mj.njnt)]
print('joints:', names)
order = [names.index(n) for n in cfg.joint_names]

rng = np.random.default_rng(0)
worst_g = worst_f = 0.0
scale_g = scale_f = 0.0
for trial in range(150):
    q = rng.uniform(model.lower, model.upper)
    qd = rng.uniform(-1.5, 1.5, model.n)
    qdd = rng.uniform(-3.0, 3.0, model.n)

    for gravity_only in (True, False):
        v = np.zeros(model.n) if gravity_only else qd
        a = np.zeros(model.n) if gravity_only else qdd
        for k, j in enumerate(order):
            data.qpos[mj.jnt_qposadr[j]] = q[k]
            data.qvel[mj.jnt_dofadr[j]] = v[k]
            data.qacc[mj.jnt_dofadr[j]] = a[k]
        mujoco.mj_forward(mj, data)
        data.qacc[:] = 0.0
        for k, j in enumerate(order):
            data.qacc[mj.jnt_dofadr[j]] = a[k]
        mujoco.mj_inverse(mj, data)
        theirs = np.array([data.qfrc_inverse[mj.jnt_dofadr[j]] for j in order])

        mine = model.inverse_dynamics(q, v, a, include_friction=False)
        mine = mine - model.reflected_inertia * a
        diff = float(np.max(np.abs(theirs - mine)))
        if gravity_only:
            worst_g = max(worst_g, diff); scale_g = max(scale_g, np.abs(theirs).max())
        else:
            worst_f = max(worst_f, diff); scale_f = max(scale_f, np.abs(theirs).max())

print('\ngravity torque  vs MuJoCo: worst %.3e N.m  (largest torque %.1f N.m)' % (worst_g, scale_g))
print('full dynamics   vs MuJoCo: worst %.3e N.m  (largest torque %.1f N.m)' % (worst_f, scale_f))
tol = 1e-6 * max(scale_f, 1.0)
print('tolerance %.3e -> %s' % (tol, 'PASS' if max(worst_g, worst_f) < tol else 'FAIL'))
