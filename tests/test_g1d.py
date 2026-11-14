"""G1-D forward kinematics and failure rules."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from supermimic.g1d import constants as C
from supermimic.g1d.env_cfg import (
  failure_errors,
  heading_yaw,
  torso_offset_xz,
  yaw_link_height,
  yaw_quat_from_angle,
)
from supermimic.g1d.kinematics import G1DForwardKinematics
from supermimic.g1d.robot import load_model

BODIES = (*C.TRACKED_BODIES, "Yaw_Link")


def _random_state(model: mujoco.MjModel, rng: np.random.Generator):
  """A random G1-D state: planar root, column within its stroke, upper joints."""
  x, y, yaw = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-np.pi, np.pi)
  column = rng.uniform(*C.COLUMN_LIMIT)
  upper = np.array(
    [rng.uniform(*model.joint(n).range) for n in C.UPPER_JOINTS], dtype=np.float64
  )
  vel = rng.normal(size=3 + 1 + C.NUM_UPPER)
  return x, y, yaw, column, upper, vel


def test_fk_matches_mujoco():
  model = load_model()
  data = mujoco.MjData(model)
  fk = G1DForwardKinematics(model, BODIES, dtype=torch.float64)
  rng = np.random.default_rng(0)
  qa = {
    n: int(model.joint(n).qposadr[0])
    for n in (*C.BASE_JOINTS, *C.LIFT_JOINTS, *C.UPPER_JOINTS)
  }
  va = {n: int(model.joint(n).dofadr[0]) for n in qa}
  for _ in range(8):
    x, y, yaw, column, upper, vel = _random_state(model, rng)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[qa["base_x_joint"]], data.qpos[qa["base_y_joint"]] = x, y
    data.qpos[qa["base_yaw_joint"]] = yaw
    data.qvel[va["base_x_joint"]], data.qvel[va["base_y_joint"]] = vel[0], vel[1]
    data.qvel[va["base_yaw_joint"]] = vel[2]
    for n in C.LIFT_JOINTS:
      data.qpos[qa[n]] = 0.5 * column
      data.qvel[va[n]] = 0.5 * vel[3]
    for i, n in enumerate(C.UPPER_JOINTS):
      data.qpos[qa[n]] = upper[i]
      data.qvel[va[n]] = vel[4 + i]
    mujoco.mj_forward(model, data)

    agv = model.body(C.ROOT_BODY).id
    f64 = torch.float64
    root_pos = torch.tensor(data.xpos[agv], dtype=f64)
    root_quat = yaw_quat_from_angle(torch.tensor(yaw, dtype=f64))
    joint_pos = torch.tensor([column, *upper], dtype=f64)
    joint_vel = torch.tensor(vel[3:], dtype=f64)
    lin = torch.tensor([vel[0], vel[1], 0.0], dtype=f64)
    ang = torch.tensor([0.0, 0.0, vel[2]], dtype=f64)
    pos, quat, lin_w, ang_w = fk(root_pos, root_quat, joint_pos, lin, ang, joint_vel)
    for r, name in enumerate(BODIES):
      b = model.body(name).id
      res = np.zeros(6)
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
      np.testing.assert_allclose(pos[r].numpy(), data.xpos[b], atol=1e-9)
      q = data.xquat[b] * np.sign(data.xquat[b] @ quat[r].numpy())
      np.testing.assert_allclose(quat[r].numpy(), q, atol=1e-9)
      np.testing.assert_allclose(ang_w[r].numpy(), res[:3], atol=1e-9)
      np.testing.assert_allclose(lin_w[r].numpy(), res[3:], atol=1e-9)
    # The column-top height formula against the FK's Yaw_Link.
    offset = torso_offset_xz(model)
    t = BODIES.index("torso_link")
    z = yaw_link_height(pos[t], quat[t], offset)
    assert float(z) == pytest.approx(float(pos[BODIES.index("Yaw_Link"), 2]), abs=3e-4)


def test_heading_yaw_roundtrip():
  yaw = torch.linspace(-3.1, 3.1, 13)
  torch.testing.assert_close(heading_yaw(yaw_quat_from_angle(yaw)), yaw)


def test_failure_errors_on_reference_are_zero():
  model = load_model()
  fk = G1DForwardKinematics(model, C.TRACKED_BODIES)
  root_pos = torch.tensor([[1.0, 2.0, C.AGV_HEIGHT]])
  root_quat = yaw_quat_from_angle(torch.tensor([0.7]))
  joints = torch.zeros(1, C.NJ)
  joints[:, 0] = 0.2
  zero = torch.zeros(1, 3)
  pos, quat, _, _ = fk(root_pos, root_quat, joints, zero, zero, joints)
  offset = torso_offset_xz(model)
  err = failure_errors(root_pos, pos, quat, root_quat, pos, quat, offset)
  torch.testing.assert_close(err, torch.zeros(1, 3), atol=1e-5, rtol=0)
  # Column 5 cm lower: only the column rule moves (by 5 cm), the upper bodies by 5 cm.
  low = joints.clone()
  low[:, 0] -= 0.05
  pos2, quat2, _, _ = fk(root_pos, root_quat, low, zero, zero, low)
  err = failure_errors(root_pos, pos2, quat2, root_quat, pos, quat, offset)
  torch.testing.assert_close(err, torch.tensor([[0.0, 0.05, 0.05]]), atol=1e-5, rtol=0)
  # Chassis moved 20 cm: the re-anchored upper bodies do not move.
  moved = root_pos + torch.tensor([[0.2, 0.0, 0.0]])
  pos3, quat3, _, _ = fk(moved, root_quat, joints, zero, zero, joints)
  err = failure_errors(moved, pos3, quat3, root_quat, pos, quat, offset)
  torch.testing.assert_close(err, torch.tensor([[0.2, 0.0, 0.0]]), atol=1e-5, rtol=0)
