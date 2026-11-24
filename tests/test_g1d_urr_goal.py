"""The G1-D URR keypoints and the URR actor goal / critic inputs built from them."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.g1d import constants as C
from supermimic.g1d.env_cfg import HalfTable, yaw_quat_from_angle
from supermimic.g1d.kinematics import G1DForwardKinematics
from supermimic.g1d.representation import G1DRepresentation
from supermimic.g1d.robot import load_model
from supermimic.g1d.urr import (
  motion_urr_rows,
  motion_urr_table,
  round_to_half,
  urr_features,
)
from supermimic.representations import (
  GOAL_REL_ORI_DIM,
  NUM_GOAL_FRAMES,
  create_representation,
  extract_actor_goal,
  goal_window_indices,
)

PALMS = ("left_hand_palm_link", "right_hand_palm_link")


def _random_qpos(model, data, rng):
  qa = {
    n: int(model.joint(n).qposadr[0])
    for n in (*C.BASE_JOINTS, *C.LIFT_JOINTS, *C.UPPER_JOINTS)
  }
  va = {n: int(model.joint(n).dofadr[0]) for n in qa}
  yaw, column = rng.uniform(-np.pi, np.pi), rng.uniform(*C.COLUMN_LIMIT)
  upper = [rng.uniform(*model.joint(n).range) for n in C.UPPER_JOINTS]
  vel = rng.normal(size=3 + 1 + C.NUM_UPPER)
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  data.qpos[qa["base_x_joint"]], data.qpos[qa["base_y_joint"]] = rng.uniform(-3, 3, 2)
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
  return yaw, np.array([column, *upper]), vel


def test_urr_features_match_mujoco():
  model = load_model()
  data = mujoco.MjData(model)
  fk = G1DForwardKinematics(model, C.TRACKED_BODIES, dtype=torch.float64)
  rng = np.random.default_rng(0)
  f64 = torch.float64
  for _ in range(6):
    yaw, joints, vel = _random_qpos(model, data, rng)
    agv = model.body(C.ROOT_BODY).id
    rot, origin = data.xmat[agv].reshape(3, 3), data.xpos[agv]
    root_quat = yaw_quat_from_angle(torch.tensor(yaw, dtype=f64))
    lin_b = quat_apply_inverse(
      root_quat, torch.tensor([vel[0], vel[1], 0.0], dtype=f64)
    )
    ang_b = torch.tensor([0.0, 0.0, vel[2]], dtype=f64)
    pose, twist = urr_features(
      fk,
      torch.tensor(joints, dtype=f64),
      torch.tensor(vel[3:], dtype=f64),
      lin_b,
      ang_b,
    )
    pos, ori = pose[:15].view(5, 3).numpy(), pose[15:].view(3, 6).numpy()
    lin, ang = twist[:15].view(5, 3).numpy(), twist[15:].view(3, 3).numpy()

    def local(body, rot=rot, origin=origin):
      b = model.body(body).id
      res = np.zeros(6)
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
      mat = rot.T @ data.xmat[b].reshape(3, 3)
      return rot.T @ (data.xpos[b] - origin), mat, rot.T @ res[3:], rot.T @ res[:3]

    # Hands: the palm bodies, pose and twist.
    for k, body in zip((1, 2), PALMS, strict=True):
      p, mat, v, w = local(body)
      np.testing.assert_allclose(pos[k], p, atol=1e-9)
      np.testing.assert_allclose(ori[k], mat[:, :2].reshape(-1), atol=1e-9)
      np.testing.assert_allclose(lin[k], v, atol=1e-9)
      np.testing.assert_allclose(ang[k], w, atol=1e-9)
    # Elbows: the elbow links.
    for k, body in zip((3, 4), ("left_elbow_link", "right_elbow_link"), strict=True):
      p, _, v, _ = local(body)
      np.testing.assert_allclose(pos[k], p, atol=1e-9)
      np.testing.assert_allclose(lin[k], v, atol=1e-9)
    # Chest: 0.2 m up the torso_link's z axis, moving with it.
    p, mat, v, w = local("torso_link")
    offset = mat @ np.array(C.URR_OFFSETS[0])
    np.testing.assert_allclose(pos[0], p + offset, atol=1e-9)
    np.testing.assert_allclose(ori[0], mat[:, :2].reshape(-1), atol=1e-9)
    np.testing.assert_allclose(lin[0], v + np.cross(w, offset), atol=1e-9)
    np.testing.assert_allclose(ang[0], w, atol=1e-9)


def _motion(lengths=(30, 70), seed=0):
  """A two-clip reference motion on the real kinematics, random but bounded."""
  g = torch.Generator().manual_seed(seed)
  total = sum(lengths)
  model = load_model()
  ranges = np.array([C.COLUMN_LIMIT, *(model.joint(n).range for n in C.UPPER_JOINTS)])
  low, high = torch.tensor(ranges.T, dtype=torch.float32)
  yaw = torch.rand(total, generator=g) * 6 - 3
  root_pos = torch.cat(
    [torch.rand(total, 2, generator=g) * 4 - 2, torch.full((total, 1), C.AGV_HEIGHT)],
    -1,
  )
  return SimpleNamespace(
    root_pos_w=root_pos,
    root_quat_w=yaw_quat_from_angle(yaw),
    root_lin_vel_w=torch.cat(
      [torch.randn(total, 2, generator=g), torch.zeros(total, 1)], -1
    ),
    root_ang_vel_w=torch.cat(
      [torch.zeros(total, 2), torch.randn(total, 1, generator=g)], -1
    ),
    joint_pos=low + (high - low) * torch.rand(total, C.NJ, generator=g),
    joint_vel=torch.randn(total, C.NJ, generator=g),
    fk=G1DForwardKinematics(model, C.TRACKED_BODIES),
    time_step_total=total,
    motion_ends=torch.tensor(np.cumsum(lengths)),
  )


def _robot_aux(motion, times):
  """Robot chassis 10 cm and 0.1 rad off the reference's."""
  pos = motion.root_pos_w[times] + torch.tensor([0.1, 0.0, 0.0])
  quat = yaw_quat_from_angle(torch.full((len(times),), 0.1))
  return torch.cat([pos, quat], -1)


def _rep(**kwargs):
  rep = G1DRepresentation("cpu", **kwargs)
  rep.actor_goal_noise = False
  return rep


def test_joint_goal_is_unchanged():
  motion = _motion()
  times = torch.tensor([0, 25, 29, 30, 64, 99])
  wm_goal, actor_goal = _rep().compute_wm_goal(motion, times, _robot_aux(motion, times))
  assert wm_goal.shape[-1] == 433
  torch.testing.assert_close(actor_goal, extract_actor_goal(wm_goal))
  assert _rep().actor_goal_dim(433) == 403 == actor_goal.shape[-1]
  assert _rep().critic_extra_dim == 1


def test_urr_goal_layout():
  motion = _motion()
  times = torch.tensor([0, 25, 29, 30, 64, 99])
  aux = _robot_aux(motion, times)
  joints_wm, joints_actor = _rep().compute_wm_goal(motion, times, aux)
  rep = _rep(actor_goal="urr")
  wm_goal, actor_goal = rep.compute_wm_goal(motion, times, aux)
  # The critic's goal keeps the joints; the actor's ends as before (chassis position
  # now, chassis orientation over the window).
  torch.testing.assert_close(wm_goal, joints_wm)
  assert actor_goal.shape[-1] == rep.actor_goal_dim(433) == 393
  tail = 3 + GOAL_REL_ORI_DIM
  torch.testing.assert_close(actor_goal[:, -tail:], joints_actor[:, -tail:])
  # Each window frame: that frame's keypoint pose; past a clip's end, its last frame.
  idx = goal_window_indices(motion, times)
  assert idx[2, 1] == 29 and idx[4, -1] == 99
  root_quat = motion.root_quat_w[idx]
  pose, _ = urr_features(
    motion.fk,
    motion.joint_pos[idx],
    motion.joint_vel[idx],
    quat_apply_inverse(root_quat, motion.root_lin_vel_w[idx]),
    quat_apply_inverse(root_quat, motion.root_ang_vel_w[idx]),
  )
  frames = actor_goal[:, : NUM_GOAL_FRAMES * C.URR_POSE_DIM]
  torch.testing.assert_close(frames, pose.reshape(len(times), -1), atol=1e-5, rtol=0)


def test_urr_critic_extra():
  motion = _motion()
  times = torch.tensor([3, 40, 99])
  rep = _rep(critic_urr=True)
  assert rep.critic_extra_dim == 1 + 11 * 57 == 628
  # A robot exactly on the reference frame: its own keypoints are the table's row.
  quat = motion.root_quat_w[times]
  rep_state = torch.cat(
    [
      torch.full((3, 1), C.AGV_HEIGHT),
      quat_apply_inverse(quat, torch.tensor([0.0, 0.0, -1.0]).expand(3, 3)),
      quat_apply_inverse(quat, motion.root_lin_vel_w[times]),
      quat_apply_inverse(quat, motion.root_ang_vel_w[times]),
      motion.joint_pos[times] + torch.tensor([0.05] + [0.0] * 16),
      motion.joint_vel[times],
    ],
    -1,
  )
  extra = rep.critic_extra(rep_state, motion, times)
  assert extra.shape == (3, 628)
  torch.testing.assert_close(extra[:, :1], torch.full((3, 1), 0.05))
  torch.testing.assert_close(
    extra[:, :1], _rep().critic_extra(rep_state, motion, times)
  )
  table = motion_urr_table(motion, True)
  idx = goal_window_indices(motion, times)
  torch.testing.assert_close(extra[:, 1:571], table[idx].reshape(3, -1))
  # The robot's column is 5 cm up: all five keypoints rise 5 cm, the rest of its
  # keypoints match the reference frame's.
  own = extra[:, 571:]
  ref = table[times]
  lift = torch.zeros(57)
  lift[[2, 5, 8, 11, 14]] = 0.05
  torch.testing.assert_close(own, ref + lift, atol=1e-5, rtol=0)


def test_urr_table_half_precision():
  motion = _motion()
  full = motion_urr_table(motion, True).clone()
  half = _motion()
  for name in (
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "joint_pos",
    "joint_vel",
  ):
    setattr(half, name, HalfTable(getattr(half, name), "cpu"))
  table = motion_urr_table(half, False)
  assert isinstance(table, HalfTable) and table.shape[1] == C.URR_POSE_DIM
  torch.testing.assert_close(table[:], full[:, : C.URR_POSE_DIM], atol=3e-3, rtol=0)
  # A later twist request re-tabulates the wider table.
  assert motion_urr_table(half, True).shape[1] == 57


def test_urr_options_need_g1d():
  with pytest.raises(ValueError, match="g1d"):
    create_representation(
      "gencoord", 0, 0, 0.02, 4, torch.zeros(3), "proprio", g1d_actor_goal="urr"
    )
  with pytest.raises(ValueError, match="actor_goal"):
    G1DRepresentation("cpu", actor_goal="points")


@pytest.mark.parametrize("actor_goal", ["joints", "urr"])
def test_chassis_window_goal_layout(actor_goal):
  motion = _motion()
  times = torch.tensor([0, 25, 29, 30, 64, 99])
  aux = _robot_aux(motion, times)
  base_wm, base_actor = _rep(actor_goal=actor_goal).compute_wm_goal(motion, times, aux)
  rep = _rep(actor_goal=actor_goal, chassis_window=True)
  wm_goal, actor_goal_ = rep.compute_wm_goal(motion, times, aux)
  # The critic's goal is untouched; the actor's content is too, and its chassis part
  # is the whole window's [rel_pos, rel_ori] in place of [anchor_pos, rel_ori].
  torch.testing.assert_close(wm_goal, base_wm)
  rel = 3 * NUM_GOAL_FRAMES + GOAL_REL_ORI_DIM
  assert actor_goal_.shape[-1] == rep.actor_goal_dim(433)
  assert rep.actor_goal_dim(433) == base_actor.shape[-1] - 3 + 3 * NUM_GOAL_FRAMES
  torch.testing.assert_close(actor_goal_[:, -rel:], wm_goal[:, -rel:])
  content = base_actor.shape[-1] - 3 - GOAL_REL_ORI_DIM
  torch.testing.assert_close(actor_goal_[:, :content], base_actor[:, :content])
  # Its first frame's position is the current-frame anchor_pos the base goal gives.
  torch.testing.assert_close(
    actor_goal_[:, content : content + 3], base_actor[:, content : content + 3]
  )


def test_chassis_window_noise_moves_the_whole_window_with_the_robot():
  motion = _motion()
  times = torch.tensor([0, 25, 29, 30, 64, 99])
  aux = _robot_aux(motion, times)
  rep = _rep(actor_goal="urr", chassis_window=True)
  _, clean = rep.compute_wm_goal(motion, times, aux)
  rep.actor_goal_noise = True
  torch.manual_seed(0)
  _, noisy = rep.compute_wm_goal(motion, times, aux)
  n, pos_dim = NUM_GOAL_FRAMES, 3 * NUM_GOAL_FRAMES
  content = n * C.URR_POSE_DIM
  torch.testing.assert_close(noisy[:, :content], clean[:, :content])
  pos = [g[:, content : content + pos_dim].reshape(-1, n, 3) for g in (clean, noisy)]
  # One planar pose error for all frames: the window keeps its shape and height,
  # and moves by at most the noise.
  torch.testing.assert_close(
    torch.cdist(pos[0], pos[0]), torch.cdist(pos[1], pos[1]), atol=1e-5, rtol=0
  )
  torch.testing.assert_close(pos[1][..., 2], pos[0][..., 2], atol=1e-6, rtol=0)
  reach = pos[0][..., :2].norm(dim=-1).amax()
  bound = C.ANCHOR_POS_NOISE * 2**0.5 + reach * C.ANCHOR_YAW_NOISE
  shift = (pos[1] - pos[0]).norm(dim=-1)
  assert 0 < float(shift.amax()) <= bound
  # Every frame's heading turns by the same yaw, within its bound.
  ori = [g[:, content + pos_dim :].reshape(-1, n, 6) for g in (clean, noisy)]
  yaw = [torch.atan2(o[..., 1], o[..., 0]) for o in ori]  # first column's heading
  turn = torch.remainder(yaw[1] - yaw[0] + torch.pi, 2 * torch.pi) - torch.pi
  torch.testing.assert_close(turn, turn[:, :1].expand_as(turn), atol=1e-5, rtol=0)
  assert float(turn.abs().amax()) <= C.ANCHOR_YAW_NOISE + 1e-6


def test_chassis_window_needs_g1d():
  with pytest.raises(ValueError, match="g1d"):
    create_representation(
      "gencoord", 0, 0, 0.02, 4, torch.zeros(3), "proprio", g1d_chassis_window=True
    )


def _half_motion():
  motion = _motion()
  for name in (
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "joint_pos",
    "joint_vel",
  ):
    setattr(motion, name, HalfTable(getattr(motion, name), "cpu"))
  return motion


def test_round_to_half_is_the_half_round_trip():
  g = torch.Generator().manual_seed(0)
  x = torch.randn(200_000, generator=g) * torch.exp2(
    torch.randint(-30, 12, (200_000,), generator=g).float()
  )
  x = torch.cat(
    [
      x,
      torch.tensor(
        [0.0, -0.0, 2.0**-24, 2.0**-25, 3 * 2.0**-26, 1 + 2.0**-11, 1 + 3 * 2.0**-11]
      ),
    ]
  )
  assert torch.equal(round_to_half(x), x.half().float())


@pytest.mark.parametrize("half", [False, True])
def test_urr_rows_on_use_are_the_table_rows(half):
  motion = _half_motion() if half else _motion()
  idx = goal_window_indices(motion, torch.tensor([0, 25, 29, 30, 64, 99]))
  table = motion_urr_table(_half_motion() if half else _motion(), True)[idx]
  torch.testing.assert_close(
    motion_urr_rows(motion, idx, True), table, atol=1e-6, rtol=0
  )
  pose = motion_urr_rows(motion, idx, False)
  torch.testing.assert_close(pose, table[..., : C.URR_POSE_DIM], atol=1e-6, rtol=0)
  assert not hasattr(motion, "_urr_table")  # nothing tabulated


@pytest.mark.parametrize("half", [False, True])
def test_goal_and_critic_without_the_table_are_unchanged(half):
  times = torch.tensor([0, 25, 29, 30, 64, 99])
  out = {}
  for urr_table in (True, False):
    motion = _half_motion() if half else _motion()
    aux = _robot_aux(motion, times)
    rep = _rep(actor_goal="urr", critic_urr=True, urr_table=urr_table)
    wm_goal, actor_goal = rep.compute_wm_goal(motion, times, aux)
    rep_state = torch.cat(
      [torch.zeros(len(times), 10), motion.joint_pos[times], motion.joint_vel[times]],
      -1,
    )
    out[urr_table] = (wm_goal, actor_goal, rep.critic_extra(rep_state, motion, times))
    assert hasattr(motion, "_urr_table") == urr_table
  for with_table, without in zip(out[True], out[False], strict=True):
    torch.testing.assert_close(without, with_table, atol=1e-6, rtol=0)
