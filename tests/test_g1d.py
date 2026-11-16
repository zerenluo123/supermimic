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
from supermimic.g1d.robot import drive_params, load_model, stiff_drive_params

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


def test_stiff_chassis_gains_are_the_soft_ones_over_the_loop_time():
  soft, stiff = drive_params(), stiff_drive_params()
  for key in ("kv_xy", "kv_yaw"):
    assert stiff[key] == pytest.approx(soft[key] * C.TAU / C.STIFF_LOOP_TAU)


def test_checkpoint_plant_defaults_to_the_soft_chassis(tmp_path):
  import json

  from supermimic.g1d.benchmark import checkpoint_plant

  (tmp_path / "run.json").write_text(json.dumps({"step": 0, "runner_cfg": {}}))
  assert checkpoint_plant(tmp_path) == {"chassis": "soft"}
  run = {"step": 0, "runner_cfg": {}, "plant": {"chassis": "stiff"}}
  (tmp_path / "run.json").write_text(json.dumps(run))
  assert checkpoint_plant(tmp_path) == {"chassis": "stiff"}


def test_motion_shards_partition_a_packed_library(tmp_path):
  import json
  from types import SimpleNamespace

  from mjlab.tasks.tracking.mdp.commands import MotionLoader

  from supermimic.g1d.env_cfg import MOTION_FIELDS, load_motion_shard

  rng = np.random.default_rng(0)
  lengths = np.array([5, 3, 7, 2, 6, 4, 9])
  ends = np.cumsum(lengths)
  widths = {"root_pos_w": 3, "root_quat_w": 4, "root_lin_vel_w": 3}
  widths |= {"root_ang_vel_w": 3, "joint_pos": C.NJ, "joint_vel": C.NJ}
  full = {}
  for name in MOTION_FIELDS:
    dtype = np.float32 if name == "root_pos_w" else np.float16
    full[name] = rng.normal(size=(ends[-1], widths[name])).astype(dtype)
    np.save(tmp_path / f"{name}.npy", full[name])
  np.save(tmp_path / "motion_ends.npy", ends)
  (tmp_path / "joint_names.json").write_text(json.dumps(list(C.STATE_JOINTS)))
  names = [f"m{i}" for i in range(len(lengths))]
  (tmp_path / "motion_names.json").write_text(json.dumps(names))
  fk = SimpleNamespace(joint_names=list(C.STATE_JOINTS))
  starts = np.concatenate([[0], ends[:-1]])
  seen = []
  for rank in range(3):
    shard = load_motion_shard(str(tmp_path), fk, MotionLoader, rank, 3, True, "cpu")
    clips = list(range(rank, len(lengths), 3))
    assert shard.motion_names == [names[c] for c in clips]
    assert shard.motion_ends.tolist() == np.cumsum(lengths[clips]).tolist()
    assert shard.time_step_total == int(lengths[clips].sum())
    for name in MOTION_FIELDS:
      want = np.concatenate([full[name][starts[c] : ends[c]] for c in clips])
      got = getattr(shard, name)[torch.arange(shard.time_step_total)].numpy()
      np.testing.assert_allclose(got, want.astype(np.float32))
    seen += clips
  assert sorted(seen) == list(range(len(lengths)))


def test_fine_bins_cover_the_shard_timeline():
  from types import SimpleNamespace

  from mjlab.tasks.tracking.mdp.commands import MotionCommand

  from supermimic.g1d.env_cfg import set_bin_count

  total = 1003
  metrics = {k: torch.zeros(1) for k in ("sampling_entropy", "sampling_top1_prob")}
  metrics["sampling_top1_bin"] = torch.zeros(1)
  cmd = SimpleNamespace(
    bin_count=101,
    _external_probs=torch.full((101,), 1 / 101),
    motion=SimpleNamespace(time_step_total=total),
    metrics=metrics,
    device="cpu",
  )
  set_bin_count(cmd, 4 * cmd.bin_count)
  assert cmd.bin_count == 404
  assert cmd._external_probs.shape == (404,) and cmd.bin_failed_count.shape == (404,)
  for b in (0, 403):  # all mass on the first, then the last bin
    probs = torch.zeros(404)
    probs[b] = 1.0
    frames = MotionCommand._sample_bins(cmd, torch.arange(64), probs)
    assert int(frames.min()) >= 0 and int(frames.max()) <= total - 2
    lo, hi = b * (total - 2) // 404, (b + 1) * (total - 2) // 404
    assert int(frames.min()) >= lo and int(frames.max()) <= max(hi, lo + 1)
