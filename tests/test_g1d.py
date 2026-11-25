"""G1-D forward kinematics and failure rules."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.g1d import constants as C
from supermimic.g1d.env_cfg import (
  ACTION_RATE_WEIGHT,
  FAILURE_RULES,
  JOINT_LIMIT_WEIGHT,
  TRACKING_TERMS,
  WORLD_TRACKING_TERMS,
  _terms_from,
  apply_failure_rule,
  apply_reward_rule,
  effective_action,
  failure_errors,
  g1d_env_cfg,
  heading_yaw,
  torso_offset_xz,
  tracking_errors,
  world_tracking_errors,
  world_tracking_reward,
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


def test_fk_gradients_match_finite_differences():
  """Gradients flow through the pass recomputed in the backward."""
  fk = G1DForwardKinematics(load_model(), BODIES, dtype=torch.float64)
  torch.manual_seed(0)
  f64 = torch.float64
  quat = torch.randn(2, 4, dtype=f64)
  inputs = [
    torch.randn(2, 3, dtype=f64),
    quat / quat.norm(dim=-1, keepdim=True),
    0.3 * torch.randn(2, C.NJ, dtype=f64),
    torch.randn(2, 3, dtype=f64),
    torch.randn(2, 3, dtype=f64),
    torch.randn(2, C.NJ, dtype=f64),
  ]
  inputs = [x.requires_grad_() for x in inputs]
  assert torch.autograd.gradcheck(fk, inputs)


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
  # In the world frame the upper bodies moved with it.
  err = failure_errors(moved, pos3, quat3, root_quat, pos, quat, offset, True)
  torch.testing.assert_close(err, torch.tensor([[0.2, 0.2, 0.0]]), atol=1e-5, rtol=0)


def test_world_reward_scores_the_chassis_offset():
  """A chassis moved 20 cm with the body unchanged: the re-anchored body terms do
  not see it, the world-frame ones see 20 cm on every body."""
  fk = G1DForwardKinematics(load_model(), C.TRACKED_BODIES)
  root_pos = torch.tensor([[1.0, 2.0, C.AGV_HEIGHT]])
  root_quat = yaw_quat_from_angle(torch.tensor([0.7]))
  joints = torch.zeros(1, C.NJ)
  zero = torch.zeros(1, 3)
  ref = fk(root_pos, root_quat, joints, zero, zero, joints)
  moved = root_pos + torch.tensor([[0.2, 0.0, 0.0]])
  pos, quat, lin, ang = fk(moved, root_quat, joints, zero, zero, joints)
  rot = matrix_from_quat(quat)
  relative = tracking_errors(moved, root_quat, pos, rot, lin, ang, *ref)
  world = world_tracking_errors(pos, rot, lin, ang, *ref)
  torch.testing.assert_close(relative[:, 2], torch.zeros(1), atol=1e-6, rtol=0)
  torch.testing.assert_close(world[:, 0], torch.full((1,), 0.04), atol=1e-6, rtol=0)
  torch.testing.assert_close(world[:, 1:], torch.zeros(1, 3), atol=1e-6, rtol=0)


def test_world_reward_is_the_env_terms_summed():
  """The world model's ``world_reward`` step reward is what the env's reward
  manager sums from ``_terms_from``: the body terms, the penalties, times dt."""
  model = load_model()
  fk = G1DForwardKinematics(model, C.TRACKED_BODIES)
  torch.manual_seed(0)
  n, dt = 6, 0.02
  root_pos = torch.cat([torch.randn(n, 2), torch.full((n, 1), C.AGV_HEIGHT)], -1)
  root_quat = yaw_quat_from_angle(torch.randn(n))
  qj, vj = 0.3 * torch.randn(n, C.NJ), torch.randn(n, C.NJ)
  lin = torch.cat([torch.randn(n, 2), torch.zeros(n, 1)], -1)
  ang = torch.cat([torch.zeros(n, 2), torch.randn(n, 1)], -1)
  state = torch.cat([root_pos, root_quat, qj, lin, ang, vj], -1)
  ref = fk(
    root_pos + 0.1 * torch.randn(n, 3),
    yaw_quat_from_angle(torch.randn(n)),
    0.3 * torch.randn(n, C.NJ),
    torch.randn(n, 3),
    torch.randn(n, 3),
    torch.randn(n, C.NJ),
  )
  action, prev = torch.randn(n, C.NUM_UPPER + 4), torch.randn(n, C.NUM_UPPER + 4)
  limits = torch.stack([torch.full((C.NJ,), -0.2), torch.full((C.NJ,), 0.2)], -1)
  args = (*ref, action, prev, limits, torch.zeros(n, C.NJ), torso_offset_xz(model))
  terms = _terms_from(fk, state, *args, world_reward=True)
  assert not {name for name, _, _ in TRACKING_TERMS[:2]} & set(terms)
  summed = sum(weight * terms[name] for name, weight, _ in WORLD_TRACKING_TERMS)
  env_reward = dt * (
    summed
    + ACTION_RATE_WEIGHT * terms["action_rate_l2"]
    + JOINT_LIMIT_WEIGHT * terms["joint_limit"]
  )
  pos, quat, lin_w, ang_w = fk(root_pos, root_quat, qj, lin, ang, vj)
  wm_reward = world_tracking_reward(
    pos,
    matrix_from_quat(quat),
    lin_w,
    ang_w,
    torch.cat([qj, vj], -1),
    effective_action(action),
    effective_action(prev),
    *ref,
    limits,
    dt,
  )
  torch.testing.assert_close(wm_reward, env_reward)
  # The re-anchored rule keeps all six tracking terms; the failure errors and the
  # benchmark's error terms do not depend on the rule.
  relative = _terms_from(fk, state, *args)
  assert {name for name, _, _ in TRACKING_TERMS} <= set(relative)
  for key in ("failure", "upper_err", "upper_world_err", "wrist_world_err"):
    torch.testing.assert_close(terms[key], relative[key])


def test_world_reward_drops_the_chassis_terms():
  cfg = g1d_env_cfg()
  apply_reward_rule(cfg)
  assert {name for name, _, _ in TRACKING_TERMS} <= set(cfg.rewards)
  cfg.commands["motion"].world_reward = True
  apply_reward_rule(cfg)
  positive = {name for name, term in cfg.rewards.items() if term.weight > 0}
  assert positive == {name for name, _, _ in WORLD_TRACKING_TERMS}
  assert sum(cfg.rewards[name].weight for name in positive) == 4.0


def test_failure_rules_are_soft_unless_strict():
  def thresholds(cfg):
    terms = cfg.terminations
    return {r: terms[r].params["threshold"] for r in FAILURE_RULES if r in terms}

  cfg = g1d_env_cfg()
  assert thresholds(cfg) == {"chassis_xy": 0.25, "upper_body": C.UPPER_THRESHOLD}
  assert cfg.commands["motion"].world_failure
  cfg.commands["motion"].failure_rule = "strict"
  apply_failure_rule(cfg)
  assert thresholds(cfg) == {
    "chassis_xy": C.CHASSIS_THRESHOLD,
    "upper_body": C.UPPER_THRESHOLD,
    "column": C.COLUMN_THRESHOLD,
  }
  assert not cfg.commands["motion"].world_failure
  cfg.commands["motion"].failure_rule = "soft"
  apply_failure_rule(cfg)
  assert thresholds(cfg) == {"chassis_xy": 0.25, "upper_body": C.UPPER_THRESHOLD}


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


@pytest.mark.parametrize("planar", [False, True])
def test_planar_chassis_keeps_the_imagined_chassis_on_the_floor(planar):
  from supermimic.g1d.representation import G1DRepresentation

  rep = G1DRepresentation("cpu", planar_chassis=planar)
  rep.substep_coeff = 0.6
  B, N = 8, rep.num_joints
  yaw = torch.linspace(-3.0, 3.0, B)
  aux = torch.cat(
    [torch.randn(B, 2), torch.full((B, 1), 0.11), yaw_quat_from_angle(yaw)], -1
  )
  rep_state = torch.cat(
    [
      torch.full((B, 1), 0.11),
      torch.tensor([0.0, 0.0, -1.0]).expand(B, 3),
      torch.randn(B, 2),
      torch.zeros(B, 3),
      torch.randn(B, 1),  # v_xy, v_z w_x w_y, w_z
      torch.randn(B, 2 * N),
    ],
    -1,
  )
  for _ in range(5):
    aux, rep_state = rep.apply_prediction(
      aux, rep_state, torch.randn(B, rep.prediction_dim), 0.02
    )
  out_of_plane = (
    (rep_state[:, 0] - 0.11).abs().max()
    + (rep_state[:, 1:3]).abs().max()
    + rep_state[:, [6, 7, 8]].abs().max()
  )
  assert (out_of_plane < 1e-5) == planar
