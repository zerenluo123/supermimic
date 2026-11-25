"""ScaleBFM's modes on the G1-D: each episode's mode gives the actor some parts of its
robot-frame goal, the others zeroed after the goal normalizer, then a bit per part;
imagination keeps the mode of the episode each start was collected in."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_apply

from supermimic.envs.world_model_env import WorldModelEnv, sample_start
from supermimic.g1d import constants as C
from supermimic.g1d.env_cfg import (
  G1DMotionCommand,
  _terms_from,
  torso_offset_xz,
  yaw_quat_from_angle,
)
from supermimic.g1d.kinematics import G1DForwardKinematics
from supermimic.g1d.robot import load_model
from supermimic.g1d.urr import urr_body_frames
from supermimic.models import MultiEncoderAE
from supermimic.representations import NUM_GOAL_FRAMES
from supermimic.tasks.task_spec import TaskSpec, build_actor_obs_td
from tests.conftest import (
  ACTION_DIM,
  MOTION_GOAL_DIM,
  make_mock_buffer,
  make_mock_command_env,
  make_mock_dynamics,
  make_mock_wm_representation,
)
from tests.test_g1d_urr_goal import _motion, _rep, _robot_state

FULL = [1.0] * 6
THREE_POINT = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]  # the body point and the palms


def _command(num_envs: int, modes: dict[str, float]) -> G1DMotionCommand:
  """A motion command holding only its modes."""
  cmd = G1DMotionCommand.__new__(G1DMotionCommand)
  cmd._env = SimpleNamespace(num_envs=num_envs, device="cpu")
  cmd.mode = torch.ones(num_envs, len(C.GOAL_PARTS))
  cmd.set_goal_modes(modes)
  return cmd


def test_each_reset_draws_the_mode_of_its_episode():
  assert C.GOAL_MODES["three_point"] == tuple(
    part for part, given in zip(C.GOAL_PARTS, THREE_POINT, strict=True) if given
  )
  torch.manual_seed(0)
  n = 20000
  cmd = _command(n, {"full": 0.6, "three_point": 0.4})
  cmd._draw_modes(torch.arange(n))
  rows, counts = cmd.mode.unique(dim=0, return_counts=True)
  assert rows.tolist() == [THREE_POINT, FULL]
  assert counts[1] / n == pytest.approx(0.6, abs=0.02)
  # Only the envs that reset draw; the others keep their episode's mode.
  before = cmd.mode.clone()
  cmd._draw_modes(torch.tensor([0, 1]))
  torch.testing.assert_close(cmd.mode[2:], before[2:])
  # Without modes every part is given and nothing is drawn.
  cmd = _command(4, {})
  cmd._draw_modes(torch.arange(4))
  assert (cmd.mode == 1).all()
  # The benchmark's single mode.
  cmd = _command(4, {"three_point": 1.0})
  cmd._draw_modes(torch.arange(4))
  assert cmd.mode.tolist() == [THREE_POINT] * 4
  with pytest.raises(ValueError, match="goal modes"):
    _command(4, {"two_point": 1.0})
  with pytest.raises(ValueError, match="sum to 1"):
    _command(4, {"full": 0.6, "three_point": 0.6})


@pytest.mark.parametrize(
  ("actor_goal", "width"), [("urr_robot_rel", 18), ("urr_robot", 9)]
)
def test_goal_parts_tag_each_dim_with_its_part(actor_goal, width):
  """The robot-frame goal holds each window frame's parts in turn."""
  motion = _motion()
  times = torch.tensor([0, 25, 64])
  aux, rep_state = _robot_state(motion, times)
  rep = _rep(actor_goal=actor_goal, chassis_window=True, goal_modes=True)
  goal = rep.compute_wm_goal(motion, times, aux, rep_state)[1]
  parts = rep.actor_goal_parts()
  assert parts.shape == goal.shape[-1:]
  by_part = goal.view(len(times), NUM_GOAL_FRAMES, len(C.GOAL_PARTS), width)
  for p in range(len(C.GOAL_PARTS)):
    torch.testing.assert_close(
      goal[:, parts == p], by_part[:, :, p].reshape(len(times), -1)
    )
  assert _rep(actor_goal=actor_goal, chassis_window=True).actor_goal_parts() is None
  with pytest.raises(ValueError, match="robot-frame"):
    _rep(actor_goal="urr", goal_modes=True)


def _actor(goal_parts: torch.Tensor) -> MultiEncoderAE:
  spec = TaskSpec(
    name="motion",
    goal_dim=goal_parts.numel(),
    extract_training_goal=lambda *a, **k: None,
    goal_parts=goal_parts,
  )
  return MultiEncoderAE(
    obs_dim=8,
    action_dim=3,
    task_specs=[spec],
    slot_mix={"motion": 1.0},
    latent_dim=4,
    hidden_dims=(16,),
    latent_bc_weight=0.0,
    history_length=1,
  )


def test_actor_reads_withheld_parts_as_zero():
  """ScaleBFM's masked goal: a withheld part enters the encoder as zero after the
  goal normalizer, whose statistics see every part, then come the mode's bits;
  without a mode every part is given."""
  torch.manual_seed(0)
  parts = torch.arange(6).repeat_interleave(3).repeat(2)  # 2 frames of 6 parts
  actor = _actor(parts)
  first = next(m for m in actor.encoders["motion"] if isinstance(m, torch.nn.Linear))
  assert first.in_features == parts.numel() + 6
  batch = 8
  goal = 3 * torch.randn(batch, parts.numel()) + 1
  history = torch.randn(batch, 1, 8)
  actor.update_normalization(build_actor_obs_td({"motion": goal}, history))
  actor.eval()
  mode = torch.tensor([THREE_POINT] * 4 + [FULL] * 4)

  def obs(goal, mode=mode):
    modes = None if mode is None else {"motion": mode}
    return build_actor_obs_td({"motion": goal}, history, None, modes)

  x = actor._encoder_input(obs(goal), "motion", 0, batch)
  withheld = mode[:, parts] == 0
  assert withheld[:4].any() and not withheld[4:].any()
  assert (x[:, : parts.numel()][withheld] == 0).all()
  normalized = actor.goal_normalizers["motion"](goal)
  torch.testing.assert_close(x[:, : parts.numel()][~withheld], normalized[~withheld])
  torch.testing.assert_close(x[:, parts.numel() :], mode)
  # A withheld part does not reach the action; a given one does.
  moved = goal.clone()
  moved[:, parts == 0] += 5.0  # the chassis: withheld in the first four rows
  torch.testing.assert_close(actor(obs(moved))[:4], actor(obs(goal))[:4])
  assert not torch.allclose(actor(obs(moved))[4:], actor(obs(goal))[4:])
  # No mode: every part is given.
  torch.testing.assert_close(
    actor(obs(goal, None)), actor(obs(goal, torch.ones(batch, 6)))
  )


def test_imagined_rollouts_keep_their_start_mode():
  """The actor and the critic read the mode stored with each start, all rollout."""
  batch = 8
  spec = TaskSpec(
    name="motion",
    goal_dim=MOTION_GOAL_DIM,
    extract_training_goal=lambda *a, **k: None,
    goal_parts=torch.zeros(MOTION_GOAL_DIM, dtype=torch.long),
  )
  env = WorldModelEnv(
    dynamics=make_mock_dynamics(),
    representation=make_mock_wm_representation(batch),
    env=make_mock_command_env(),
    task_specs=[spec],
    history_length=1,
    termination_softness=0.05,
  )
  start = sample_start(make_mock_buffer(), 1, batch)
  mode = torch.rand(batch, 6) > 0.5
  start["mode"] = mode
  for obs in (env.reset(start), env.step(torch.zeros(batch, ACTION_DIM))[0]):
    torch.testing.assert_close(obs["mode_motion"], mode.float())
    torch.testing.assert_close(obs["critic"][:, -6:], mode.float())


def test_palm_world_error_is_the_palm_keypoints_distance():
  model = load_model()
  fk = G1DForwardKinematics(model, C.TRACKED_BODIES)
  torch.manual_seed(0)
  n = 5
  root_pos = torch.cat([torch.randn(n, 2), torch.full((n, 1), C.AGV_HEIGHT)], -1)
  root_quat = yaw_quat_from_angle(torch.randn(n))
  qj, zero = 0.3 * torch.randn(n, C.NJ), torch.zeros(n, 3)
  state = torch.cat([root_pos, root_quat, qj, zero, zero, torch.zeros(n, C.NJ)], -1)
  ref_pos, ref_quat = root_pos + 0.05, yaw_quat_from_angle(torch.randn(n))
  ref_joints = qj + 0.2 * torch.randn(n, C.NJ)
  vj = torch.zeros(n, C.NJ)
  ref = fk(ref_pos, ref_quat, ref_joints, zero, zero, vj)
  action = torch.zeros(n, C.NUM_UPPER + 4)
  limits = torch.stack([torch.full((C.NJ,), -3.0), torch.full((C.NJ,), 3.0)], -1)
  offset = torso_offset_xz(model)
  terms = _terms_from(fk, state, *ref, action, action, limits, ref_joints, offset)

  def palms(root_pos, root_quat, joints):
    local = urr_body_frames(fk, joints)[0][:, 1:3]
    return root_pos[:, None] + quat_apply(root_quat[:, None].expand(-1, 2, -1), local)

  expected = palms(ref_pos, ref_quat, ref_joints) - palms(root_pos, root_quat, qj)
  torch.testing.assert_close(
    terms["palm_world_err"], expected.norm(dim=-1).mean(-1), atol=1e-5, rtol=0
  )
