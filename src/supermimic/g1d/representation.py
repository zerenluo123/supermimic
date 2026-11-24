"""GenCoord representation of the G1-D.

The full state is GenCoord's with ``AGV_link`` as the root and the 17 state joints
``[column, 16 upper]``, so the dynamics, integration and reward are GenCoord's.
The G1-D adds:

* the tanh-saturated chassis / column action wherever an action is scored or
  observed (the plant acts on it; ``Dynamics.proc`` applies it with ``tanh_mask``);
* the column's encoder bias and its own proprio noise (metres, m/s);
* its own world-model loss channels for the column (metres): at the joint nominal
  (0.1 rad) a 0.42 m stroke would barely count;
* the failure rules as the imagined termination (``normalized_failure_error``);
* the column height error as a critic-only input (``critic_extra``);
* the actor's goal carries the reference chassis position in the robot frame
  (``anchor_pos`` of the goal group), noised in training like odometry; or, with
  ``chassis_window``, every window frame's reference chassis position and heading
  in the robot frame, the odometry noise on the robot's own pose;
* optionally the URR keypoints (``urr.py``) in place of the reference joints in
  the actor's goal (``actor_goal="urr"``), and the window's and the robot's own
  keypoints as critic inputs (``critic_urr``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_mul

from supermimic.g1d import constants as C
from supermimic.g1d.env_cfg import (
  FAILURE_RULES,
  G1DJointMap,
  effective_action,
  failure_errors,
  state_soft_limits,
  torso_offset_xz,
  yaw_quat_from_angle,
)
from supermimic.g1d.robot import default_state_q
from supermimic.g1d.urr import motion_urr_rows, motion_urr_table, urr_features
from supermimic.representations import (
  GOAL_REL_ORI_DIM,
  GOAL_REL_POS_DIM,
  NUM_GOAL_FRAMES,
  GenCoordRepresentation,
  StateRepresentation,
  _compiled_assemble_goal,
  assemble_anchor_window,
  goal_window_indices,
  motion_anchor_window,
  motion_goal_window,
)
from supermimic.utils.gencoord_helpers import decompose_rep_state
from supermimic.utils.tracking_reward import tracking_reward

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

  from supermimic.sim_context import SimContext


class G1DRepresentation(GenCoordRepresentation):
  LOSS_WEIGHT_CONFIG = {
    **StateRepresentation.LOSS_WEIGHT_CONFIG,
    "joint_pos": "nominal_error_joint_pos",
    "joint_vel": "nominal_error_joint_vel",
    "column_pos": "nominal_error_column_pos",
    "column_vel": "nominal_error_column_vel",
  }

  critic_extra_dim = 1  # the column error; more with critic_urr (per instance)
  actor_goal_noise: bool = True
  """Noise the actor's reference chassis position (training); off to evaluate."""

  def __init__(
    self,
    device: torch.device | str,
    actor_goal: str = "joints",
    critic_urr: bool = False,
    chassis_window: bool = False,
    urr_table: bool = True,
  ):
    """``actor_goal``: per window frame the reference joints and their velocities
    (``"joints"``) or the reference URR keypoint pose in that frame's reference
    chassis frame (``"urr"``). ``critic_urr`` also gives the critic the window's
    keypoint poses and twists and the robot's own. ``chassis_window`` gives the
    actor every window frame's reference chassis position (not only the current
    one's) with its heading. ``urr_table``: tabulate the motion's keypoints once;
    without it they are computed from the reference on each use (no table, the
    same values)."""
    super().__init__(C.NJ, torch.tensor(default_state_q(), device=device))
    if actor_goal not in ("joints", "urr"):
      raise ValueError(f"actor_goal is 'joints' or 'urr', not {actor_goal!r}")
    self._encoder_bias: tuple[slice, SimContext] | None = None
    self.actor_goal = actor_goal
    self.critic_urr = critic_urr
    self.chassis_window = chassis_window
    self.urr_table = urr_table
    urr_width = C.URR_POSE_DIM + C.URR_VEL_DIM
    self.critic_extra_dim = 1 + ((NUM_GOAL_FRAMES + 1) * urr_width if critic_urr else 0)

  def bind(self, env: ManagerBasedRlEnv, sim_context: SimContext) -> None:
    super().bind(env, sim_context)
    robot = env.scene["robot"]
    self._soft_joint_pos_limits = state_soft_limits(robot)
    jm = G1DJointMap(robot.joint_names)
    self._lift_cols, self._upper_cols = jm.lift, jm.upper
    columns = sim_context.find("encoder_bias")
    self._encoder_bias = None if columns is None else (columns, sim_context)
    terms = env.cfg.terminations
    self._failure_thresholds = torch.tensor(
      [terms[rule].params["threshold"] for rule in FAILURE_RULES],
      device=self._default_joint_pos.device,
    )
    self._torso_offset = torso_offset_xz(env.sim.mj_model)
    if self.urr_table and (self.actor_goal == "urr" or self.critic_urr):
      motion_urr_table(env.command_manager.get_term("motion").motion, self.critic_urr)

  # ------------------------------------------------------------------
  # Proprio
  # ------------------------------------------------------------------

  def proprio_noise_bounds(
    self, num_actions: int, device: torch.device | str
  ) -> torch.Tensor:
    bounds = super().proprio_noise_bounds(num_actions, device)
    joint_pos, joint_vel = 6, 6 + self.num_joints  # [gravity, ang_vel, qj, vj, a]
    bounds[joint_pos] = C.COLUMN_POS_NOISE
    bounds[joint_vel] = C.COLUMN_VEL_NOISE
    return bounds

  def compute_wm_obs(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    joint_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> torch.Tensor:
    obs = super().compute_wm_obs(rep_state, aux_state, joint_state, sim_ctx)
    if self._encoder_bias is None:
      return obs
    columns, sim_context = self._encoder_bias
    bias = sim_context.denormalize(sim_ctx, columns)
    bias = torch.cat(
      [bias[..., self._lift_cols].sum(-1, keepdim=True), bias[..., self._upper_cols]],
      -1,
    )
    pad = obs.new_zeros(*obs.shape[:-1], 6)
    tail = obs.new_zeros(*obs.shape[:-1], self.num_joints)
    return obs + torch.cat([pad, bias, tail], -1)

  def compute_actor_sensor(
    self,
    rep_state: torch.Tensor,
    sim_ctx: torch.Tensor,
    aux_state: torch.Tensor | None = None,
    joint_state: torch.Tensor | None = None,
    last_action: torch.Tensor | None = None,
    sim_actor_obs: torch.Tensor | None = None,
    noisy: bool = False,
  ) -> torch.Tensor:
    if last_action is not None:
      last_action = effective_action(last_action)
    return super().compute_actor_sensor(
      rep_state,
      sim_ctx,
      aux_state=aux_state,
      joint_state=joint_state,
      last_action=last_action,
      sim_actor_obs=sim_actor_obs,
      noisy=noisy,
    )

  # ------------------------------------------------------------------
  # Goal
  # ------------------------------------------------------------------

  def actor_goal_dim(self, num_goals: int) -> int:
    rel = GOAL_REL_POS_DIM + GOAL_REL_ORI_DIM
    if self.actor_goal == "joints":
      content = num_goals - 3 - rel  # wm_goal = [joints, anchor_pos, rel_pos, rel_ori]
    else:
      content = NUM_GOAL_FRAMES * C.URR_POSE_DIM
    return content + (rel if self.chassis_window else 3 + GOAL_REL_ORI_DIM)

  def compute_wm_goal(
    self,
    motion: MotionLoader,
    motion_times: torch.Tensor,
    aux_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """GenCoord's goal with the reference chassis position at ``motion_times`` in
    the robot root frame appended to the content (the goal group's
    ``anchor_pos``), so the actor sees it; noised while ``actor_goal_noise``.
    With ``actor_goal="urr"`` the actor's content is the window's keypoint poses
    in place of the joints; ``wm_goal`` keeps the joints. With ``chassis_window``
    the actor's chassis part is the whole window's ``[rel_pos, rel_ori]`` from a
    noised robot pose in place of ``[anchor_pos, rel_ori]``, and ``wm_goal`` keeps
    a clean ``anchor_pos``."""
    window = motion_goal_window(motion, motion_times)
    target_pos, target_quat = motion_anchor_window(motion, motion_times)
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    wm_goal, _ = _compiled_assemble_goal(
      window, root_pos, root_quat, target_pos, target_quat
    )
    rel = GOAL_REL_POS_DIM + GOAL_REL_ORI_DIM
    anchor_pos = wm_goal[..., -rel : -rel + 3]
    rel_part = wm_goal[..., -rel:]
    if self.chassis_window:
      # Odometry errs on the robot's own pose, so every frame's target moves with it.
      if self.actor_goal_noise:
        root_pos, root_quat = _odometry_noise(root_pos, root_quat)
      chassis = torch.cat(
        _compiled_anchor_window(root_pos, root_quat, target_pos, target_quat), dim=-1
      )
    else:
      if self.actor_goal_noise:
        anchor_pos = (
          anchor_pos + (torch.rand_like(anchor_pos) * 2 - 1) * C.ANCHOR_POS_NOISE
        )
      chassis = torch.cat([anchor_pos, rel_part[..., GOAL_REL_POS_DIM:]], dim=-1)
    wm_goal = torch.cat([window, anchor_pos, rel_part], dim=-1)
    content = window
    if self.actor_goal == "urr":
      idx = goal_window_indices(motion, motion_times)
      urr = self._motion_urr(motion, idx, False)[..., : C.URR_POSE_DIM]
      content = urr.reshape(idx.shape[0], -1)
    actor_goal = torch.cat([content, chassis], dim=-1)
    return wm_goal, actor_goal

  # ------------------------------------------------------------------
  # Reward, termination, loss
  # ------------------------------------------------------------------

  def compute_wm_reward_from_reference(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    next_joint_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    target_body_pos_w: torch.Tensor,
    target_body_quat_w: torch.Tensor,
    target_body_lin_vel_w: torch.Tensor,
    target_body_ang_vel_w: torch.Tensor,
  ) -> torch.Tensor:
    return super().compute_wm_reward_from_reference(
      next_rep_state,
      next_aux_state,
      next_joint_state,
      effective_action(action),
      effective_action(last_action),
      target_body_pos_w,
      target_body_quat_w,
      target_body_lin_vel_w,
      target_body_ang_vel_w,
    )

  def normalized_failure_error(
    self,
    aux_state: torch.Tensor,
    rep_state: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
  ) -> torch.Tensor:
    """Largest failure-rule error over its threshold (> 1: failed), the
    simulator's rules on the imagined state."""
    pos, quat, _, _ = self._bodies(aux_state, rep_state)
    return self._failure_error(aux_state, pos, quat, ref_body_pos_w, ref_body_quat_w)

  def failure_error_and_reward(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    next_joint_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    target_body_pos_w: torch.Tensor,
    target_body_quat_w: torch.Tensor,
    target_body_lin_vel_w: torch.Tensor,
    target_body_ang_vel_w: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``normalized_failure_error`` and ``compute_wm_reward_from_reference`` of
    an imagined step, from one kinematics pass (the compiler does not merge two)."""
    pos, quat, lin_vel, ang_vel = self._bodies(next_aux_state, next_rep_state)
    error = self._failure_error(
      next_aux_state, pos, quat, target_body_pos_w, target_body_quat_w
    )
    reward = tracking_reward(
      next_aux_state,
      pos,
      matrix_from_quat(quat),
      lin_vel,
      ang_vel,
      next_joint_state,
      effective_action(action),
      effective_action(last_action),
      target_body_pos_w,
      target_body_quat_w,
      target_body_lin_vel_w,
      target_body_ang_vel_w,
      self._soft_joint_pos_limits,
      self.dt,
    )
    return error, reward

  def _bodies(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """World-frame ``(pos, quat, lin_vel, ang_vel)`` of the tracked bodies, as
    ``world_frame_bodies`` before turning the quaternions into matrices."""
    _, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    return self.fk(
      root_pos,
      root_quat,
      qj,
      quat_apply(root_quat, v_b),
      quat_apply(root_quat, w_b),
      vj,
    )

  def _failure_error(
    self,
    aux_state: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
  ) -> torch.Tensor:
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    errors = failure_errors(
      root_pos,
      pos,
      quat,
      root_quat,
      ref_body_pos_w,
      ref_body_quat_w,
      self._torso_offset,
    )
    return (errors / self._failure_thresholds).amax(-1)

  def critic_extra(
    self, rep_state: torch.Tensor, motion: MotionLoader, motion_times: torch.Tensor
  ) -> torch.Tensor:
    """Column height error (robot minus reference, m), as Reproduced's critic: the
    critic otherwise barely tells a sagging column from a healthy one. With
    ``critic_urr`` then the window's reference keypoints ``[pose, twist]`` and the
    robot's own, each in its own chassis frame (the reward's re-anchored frame)."""
    _, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
    column = qj[..., :1] - motion.joint_pos[motion_times, :1]
    if not self.critic_urr:
      return column
    idx = goal_window_indices(motion, motion_times)
    ref = self._motion_urr(motion, idx, True).reshape(idx.shape[0], -1)
    pose, twist = urr_features(motion.fk, qj, vj, v_b, w_b)
    return torch.cat([column, ref, pose, twist], dim=-1)

  def _motion_urr(
    self, motion: MotionLoader, idx: torch.Tensor, with_twist: bool
  ) -> torch.Tensor:
    """The reference keypoints at ``idx``, ``[pose, twist]`` (the pose at least)."""
    if self.urr_table:
      return motion_urr_table(motion, self.critic_urr or with_twist)[idx]
    return motion_urr_rows(motion, idx, with_twist)

  def loss_blocks(
    self, rep_state: torch.Tensor, aux_state: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    blocks = super().loss_blocks(rep_state, aux_state)
    joint_pos, joint_vel = blocks["joint_pos"], blocks["joint_vel"]
    blocks["joint_pos"], blocks["joint_vel"] = joint_pos[..., 1:], joint_vel[..., 1:]
    blocks["column_pos"], blocks["column_vel"] = joint_pos[..., :1], joint_vel[..., :1]
    return blocks


_compiled_anchor_window = torch.compile(assemble_anchor_window)


def _odometry_noise(
  root_pos: torch.Tensor, root_quat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """The root pose as odometry might read it: x, y and yaw off by up to
  ``ANCHOR_POS_NOISE`` and ``ANCHOR_YAW_NOISE``, uniformly."""
  u = torch.rand_like(root_pos) * 2 - 1
  offset = torch.cat(
    [u[..., :2] * C.ANCHOR_POS_NOISE, torch.zeros_like(u[..., 2:])], -1
  )
  turn = yaw_quat_from_angle(u[..., 2] * C.ANCHOR_YAW_NOISE)
  return root_pos + offset, quat_mul(turn, root_quat)
