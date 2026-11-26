"""The world-model reward's kernel-width scale (``wm_reward_std_scale``)."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.g1d import constants as C
from supermimic.g1d.env_cfg import (
  ACTION_RATE_WEIGHT,
  JOINT_LIMIT_WEIGHT,
  TRACKING_TERMS,
  WORLD_TRACKING_TERMS,
  effective_action,
  tracking_errors,
  world_tracking_errors,
  world_tracking_reward,
  yaw_quat_from_angle,
)
from supermimic.g1d.kinematics import G1DForwardKinematics
from supermimic.g1d.robot import load_model
from supermimic.utils.tracking_reward import tracking_reward

# The position and orientation kernels are scaled; the velocity kernels are not.
SCALED = (True, True, True, True, False, False)
SCALE, DT = 0.35, 0.02


def _batch(n: int = 6):
  fk = G1DForwardKinematics(load_model(), C.TRACKED_BODIES)
  torch.manual_seed(0)
  root_pos = torch.cat([torch.randn(n, 2), torch.full((n, 1), C.AGV_HEIGHT)], -1)
  root_quat = yaw_quat_from_angle(torch.randn(n))
  qj, vj = 0.3 * torch.randn(n, C.NJ), torch.randn(n, C.NJ)
  lin = torch.cat([torch.randn(n, 2), torch.zeros(n, 1)], -1)
  ang = torch.cat([torch.zeros(n, 2), torch.randn(n, 1)], -1)
  ref = fk(
    root_pos + 0.05 * torch.randn(n, 3),
    yaw_quat_from_angle(torch.randn(n)),
    qj + 0.1 * torch.randn(n, C.NJ),
    torch.randn(n, 3),
    torch.randn(n, 3),
    torch.randn(n, C.NJ),
  )
  pos, quat, lin_w, ang_w = fk(root_pos, root_quat, qj, lin, ang, vj)
  action, prev = torch.randn(n, C.NUM_UPPER + 4), torch.randn(n, C.NUM_UPPER + 4)
  limits = torch.stack([torch.full((C.NJ,), -0.2), torch.full((C.NJ,), 0.2)], -1)
  return dict(
    root=(root_pos, root_quat),
    bodies=(pos, matrix_from_quat(quat), lin_w, ang_w),
    joints=torch.cat([qj, vj], -1),
    actions=(effective_action(action), effective_action(prev)),
    ref=ref,
    limits=limits,
  )


def _penalties(b) -> torch.Tensor:
  action, prev = b["actions"]
  qj = b["joints"].chunk(2, -1)[0]
  lo, hi = b["limits"].unbind(-1)
  limit = ((lo - qj).clamp(min=0) + (qj - hi).clamp(min=0)).sum(-1)
  return (
    ACTION_RATE_WEIGHT * ((action - prev) ** 2).sum(-1) + JOINT_LIMIT_WEIGHT * limit
  )


def test_world_reward_scales_the_position_and_orientation_kernels():
  b = _batch()
  err = world_tracking_errors(*b["bodies"], *b["ref"])
  expected = sum(
    w * torch.exp(-err[:, i] / (std * (SCALE if SCALED[i + 2] else 1.0)) ** 2)
    for i, (_, w, std) in enumerate(WORLD_TRACKING_TERMS)
  )
  args = (*b["bodies"], b["joints"], *b["actions"], *b["ref"], b["limits"], DT)
  reward = world_tracking_reward(*args, pos_ori_std_scale=SCALE)
  torch.testing.assert_close(reward, DT * (expected + _penalties(b)))
  torch.testing.assert_close(
    world_tracking_reward(*args, pos_ori_std_scale=1.0), world_tracking_reward(*args)
  )


def test_relative_reward_scales_the_position_and_orientation_kernels():
  b = _batch()
  root_pos, root_quat = b["root"]
  err = tracking_errors(root_pos, root_quat, *b["bodies"], *b["ref"])
  expected = sum(
    w * torch.exp(-err[:, i] / (std * (SCALE if SCALED[i] else 1.0)) ** 2)
    for i, (_, w, std) in enumerate(TRACKING_TERMS)
  )
  aux = torch.cat([root_pos, root_quat], -1)
  args = (aux, *b["bodies"], b["joints"], *b["actions"], *b["ref"], b["limits"], DT)
  reward = tracking_reward(*args, pos_ori_std_scale=SCALE)
  torch.testing.assert_close(reward, DT * (expected + _penalties(b)))
