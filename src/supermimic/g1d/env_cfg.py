"""G1-D motion-tracking environment, laid out like the G1's ``gencoord_env_cfg``.

G1-D specifics (from SuperMimic-Reproduced, g1d):
  * action: 19-d, 16 joint position offsets (G1 action scales) + (vx, wz, vz) =
    MAX * tanh(a) for the chassis (heading frame) and the column, on velocity
    drives. ``Dynamics.proc`` applies the same tanh (``tanh_mask``);
  * state: the GenCoord full state with ``AGV_link`` as the root and the joints
    ``[column, 16 upper]`` (the two lift slides summed), read from the joints so
    it is not a substep stale;
  * proprio: GenCoord's terms on the 17 state joints, the column with its bias,
    and the tanh-saturated previous action;
  * reward and failure rules on the forward kinematics of that state, the same
    functions the world model uses; failure = training termination: chassis world
    xy, upper bodies re-anchored at the chassis, and the column top height;
  * reset noise: AGV xy and yaw only, speed along the heading (differential
    drive), joints and column +-0.1.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.tasks.tracking.mdp import commands as commands_module
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_mul,
  sample_uniform,
)
from mjlab.viewer import ViewerConfig

from supermimic.g1d import constants as C
from supermimic.g1d.kinematics import G1DForwardKinematics
from supermimic.g1d.robot import (
  COLLISION_GEOMS,
  default_state_q,
  get_g1d_robot_cfg,
  upper_action_scale,
)
from supermimic.representations import (
  assemble_anchor_window,
  motion_anchor_window,
  motion_goal_window,
)
from supermimic.utils.helpers import rotation_error_magnitude
from supermimic.utils.tracking_reward import _reanchor_targets

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Shared math (simulator and world model)
# ---------------------------------------------------------------------------


def effective_action(action: torch.Tensor) -> torch.Tensor:
  """Policy output as the plant acts on it: the 16 joint offsets as given, the
  chassis / column speeds tanh-saturated (before their MAX)."""
  return torch.cat(
    [action[..., : C.NUM_UPPER], torch.tanh(action[..., C.NUM_UPPER :])], -1
  )


def yaw_quat_from_angle(yaw: torch.Tensor) -> torch.Tensor:
  half = 0.5 * yaw
  zero = torch.zeros_like(half)
  return torch.stack([half.cos(), zero, zero, half.sin()], dim=-1)


def heading_yaw(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_link_height(
  torso_pos: torch.Tensor, torso_quat: torch.Tensor, offset_xz: tuple[float, float]
) -> torch.Tensor:
  """World height of the Yaw_Link origin (the column top, on the waist-pitch axis)
  from the torso_link pose.

  The chassis only turns about z and the waist yaw turns the torso about its own
  z, so with the waist pitch p and the torso_link origin (o_x, ., o_z) in the
  Yaw_Link frame: z_yaw = z_torso + sin(p) o_x - cos(p) o_z, with cos(p) = R[2,2]
  and |sin(p)| = |R[2,0:2]|. The sign of sin(p) is taken positive (the pitch range
  is [-0.044, 2.36] rad), which costs at most 0.2 mm. Works for robot and
  reference alike (no joint angles needed).
  """
  w, x, y, z = torso_quat.unbind(-1)
  r20, r21, r22 = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
  sin_p = torch.sqrt(r20 * r20 + r21 * r21 + 1e-12)
  return torso_pos[..., 2] + sin_p * offset_xz[0] - r22 * offset_xz[1]


def torso_offset_xz(mj_model) -> tuple[float, float]:
  """The torso_link origin in the Yaw_Link frame (x, z)."""
  for b in range(mj_model.nbody):
    if mj_model.body(b).name.split("/")[-1] == "torso_link":
      pos = mj_model.body_pos[b]
      return float(pos[0]), float(pos[2])
  raise KeyError("torso_link")


def failure_errors(
  root_pos: torch.Tensor,
  body_pos: torch.Tensor,
  body_quat: torch.Tensor,
  root_quat: torch.Tensor,
  ref_pos: torch.Tensor,
  ref_quat: torch.Tensor,
  offset_xz: tuple[float, float],
) -> torch.Tensor:
  """``(..., 3)`` errors of the failure rules: chassis world xy distance, largest
  upper-body distance to its reference re-anchored at the chassis (xy + yaw),
  column top (Yaw_Link) height difference. Bodies are ``C.TRACKED_BODIES``."""
  chassis = (ref_pos[..., 0, :2] - root_pos[..., :2]).norm(dim=-1)
  tgt_pos, _ = _reanchor_targets(ref_pos, ref_quat, root_pos, root_quat)
  upper = (tgt_pos[..., 1:, :] - body_pos[..., 1:, :]).norm(dim=-1).amax(-1)
  t = C.TRACKED_BODIES.index("torso_link")
  column = (
    yaw_link_height(body_pos[..., t, :], body_quat[..., t, :], offset_xz)
    - yaw_link_height(ref_pos[..., t, :], ref_quat[..., t, :], offset_xz)
  ).abs()
  return torch.stack([chassis, upper, column], dim=-1)


FAILURE_RULES = ("chassis_xy", "upper_body", "column")


# ---------------------------------------------------------------------------
# Joint layout
# ---------------------------------------------------------------------------


class G1DJointMap:
  """Column indices of the G1-D joints in the entity's joint arrays."""

  def __init__(self, joint_names: tuple[str, ...] | list[str]):
    names = list(joint_names)
    self.base = [names.index(n) for n in C.BASE_JOINTS]
    self.lift = [names.index(n) for n in C.LIFT_JOINTS]
    self.upper = [names.index(n) for n in C.UPPER_JOINTS]
    self.num_joints = len(names)

  def state_joints(self, x: torch.Tensor) -> torch.Tensor:
    """``[column, 16 upper]`` from per-entity joint values (lift slides summed)."""
    return torch.cat([x[..., self.lift].sum(-1, keepdim=True), x[..., self.upper]], -1)


def state_soft_limits(robot, factor: float = 0.9) -> torch.Tensor:
  """(17, 2) soft limits of the state joints: the column range, then mjlab's soft
  limits of the upper joints."""
  jm = G1DJointMap(robot.joint_names)
  lo, hi = C.COLUMN_LIMIT
  mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
  limits = robot.data.soft_joint_pos_limits[0]
  column = limits.new_tensor([[mid - factor * half, mid + factor * half]])
  return torch.cat([column, limits[jm.upper]], 0).float()


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class G1DAction(ActionTerm):
  """19-d action ``[16 joint position offsets, vx, wz, vz]``.

  ``proc = f(a) * scale + offset`` with ``f`` = tanh on ``tanh_mask`` (the chassis
  and column speeds); ``scale``, ``_offset`` and ``tanh_mask`` are what
  ``Dynamics.proc`` reads. The joint targets subtract the encoder bias, as mjlab's
  ``JointPositionAction``.
  """

  cfg: G1DActionCfg

  def __init__(self, cfg: G1DActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.jm = G1DJointMap(self._entity.joint_names)
    dev = self.device
    self.upper_ids = torch.tensor(self.jm.upper, device=dev)
    self.base_ids = torch.tensor(self.jm.base, device=dev)
    self.lift_ids = torch.tensor(self.jm.lift, device=dev)
    self.scale = torch.tensor(
      [*upper_action_scale(), C.VX_MAX, C.WZ_MAX, C.VZ_MAX], device=dev
    )
    self._offset = torch.tensor([*default_state_q()[1:], 0.0, 0.0, 0.0], device=dev)
    self.tanh_mask = torch.arange(C.ACT_DIM, device=dev) >= C.NUM_UPPER
    self._raw = torch.zeros(self.num_envs, C.ACT_DIM, device=dev)
    self._upper_target = torch.zeros(self.num_envs, C.NUM_UPPER, device=dev)
    self._base_vel = torch.zeros(self.num_envs, 3, device=dev)
    self._lift_vel = torch.zeros(self.num_envs, 2, device=dev)

  @property
  def action_dim(self) -> int:
    return C.ACT_DIM

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions
    proc = effective_action(actions) * self.scale + self._offset
    robot = self._entity
    bias = robot.data.encoder_bias[:, self.upper_ids]
    self._upper_target = proc[:, : C.NUM_UPPER] - bias
    vx, wz, vz = proc[:, C.NUM_UPPER :].unbind(-1)
    # Differential drive: the forward speed is along the heading, never sideways.
    yaw = robot.data.joint_pos[:, self.jm.base[2]]
    self._base_vel = torch.stack([vx * yaw.cos(), vx * yaw.sin(), wz], -1)
    self._lift_vel = (0.5 * vz).unsqueeze(-1).expand(-1, 2)

  def apply_actions(self) -> None:
    robot = self._entity
    robot.set_joint_position_target(self._upper_target, joint_ids=self.upper_ids)
    robot.set_joint_velocity_target(self._base_vel, joint_ids=self.base_ids)
    # The lift's gravity feed-forward lives in its actuator.
    robot.set_joint_velocity_target(self._lift_vel, joint_ids=self.lift_ids)

  def reset(self, env_ids=None) -> None:
    self._raw[slice(None) if env_ids is None else env_ids] = 0.0


@dataclass(kw_only=True)
class G1DActionCfg(ActionTermCfg):
  def build(self, env: ManagerBasedRlEnv) -> G1DAction:
    return G1DAction(self, env)


# ---------------------------------------------------------------------------
# Motion command
# ---------------------------------------------------------------------------


class HalfTable:
  """A half-precision motion table read back as float32: indexing casts the rows it
  returns, so readers see the float32 table rounded to half."""

  def __init__(self, table: torch.Tensor, device: str):
    self.data = table.to(device=device, dtype=torch.float16)

  def __getitem__(self, index) -> torch.Tensor:
    return self.data[index].float()

  @property
  def shape(self) -> torch.Size:
    return self.data.shape

  @property
  def device(self) -> torch.device:
    return self.data.device

  @property
  def dtype(self) -> torch.dtype:
    return torch.float32


# Stored at half precision by ``half_precision_motions``; the root positions keep
# float32 (clips run 20 m from the origin, where half precision is ~1 cm coarse).
HALF_FIELDS = (
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
  "joint_pos",
  "joint_vel",
)


@contextmanager
def _g1d_motion_loading(body_names: tuple[str, ...], half_precision: bool):
  """Have ``MotionCommand.__init__`` build the G1-D kinematics (mjlab's needs a
  free-joint root) and, with ``half_precision``, a motion table that keeps
  ``HALF_FIELDS`` at half precision (loaded on the CPU first, so the float32
  copy never reaches the GPU)."""
  original_fk = commands_module.ForwardKinematics
  original_loader = commands_module.MotionLoader

  def build_fk(mj_model, body_ids, joint_q_adr, body_rows, device="cpu", **_):
    del body_ids, joint_q_adr, body_rows
    return G1DForwardKinematics(mj_model, body_names, device=device)

  def build_loader(motion_file, fk, device="cpu"):
    if not half_precision:
      return original_loader(motion_file, fk, device=device)
    motion = original_loader(motion_file, fk, device="cpu")
    motion.root_pos_w = motion.root_pos_w.to(device)
    for name in HALF_FIELDS:
      setattr(motion, name, HalfTable(getattr(motion, name), device))
    motion.motion_ends = motion.motion_ends.to(device)
    return motion

  commands_module.ForwardKinematics = build_fk
  commands_module.MotionLoader = build_loader
  try:
    yield
  finally:
    commands_module.ForwardKinematics = original_fk
    commands_module.MotionLoader = original_loader


class G1DMotionCommand(MotionCommand):
  """``MotionCommand`` on the G1-D: the reference joints are the 17 state joints
  and the root is ``AGV_link``, written to the planar and lift joints."""

  cfg: G1DMotionCommandCfg
  forced_frames: torch.Tensor | None = None
  """Per-env reset frames overriding the sampling mode (the benchmark plays each
  clip from its first frame)."""

  def __init__(self, cfg: G1DMotionCommandCfg, env: ManagerBasedRlEnv):
    with _g1d_motion_loading(tuple(cfg.body_names), cfg.half_precision_motions):
      super().__init__(cfg, env)
    self.jm = G1DJointMap(self.robot.joint_names)
    self.state_limits = state_soft_limits(self.robot)
    if float(env.scene.env_origins.abs().max()) != 0.0:
      raise ValueError("the G1-D runs with every env origin at 0 (env_spacing=0)")

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.jm.state_joints(self.robot.data.joint_pos)

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.jm.state_joints(self.robot.data.joint_vel)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.forced_frames is not None:
      frames = self.forced_frames[env_ids]
    elif self.cfg.sampling_mode == "start":
      frames = torch.zeros_like(env_ids)
    elif self.cfg.sampling_mode == "uniform":
      frames = self._uniform_sampling(env_ids)
    elif self.cfg.sampling_mode == "external":
      frames = self._external_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      frames = self._adaptive_sampling(env_ids)

    on_last = torch.isin(frames, self.motion.motion_ends - 1)
    frames = torch.where(on_last, frames - 1, frames)
    self.set_time_steps(env_ids, frames)

    n = len(env_ids)

    def u(ranges: dict[str, tuple[float, float]], key: str) -> torch.Tensor:
      lo, hi = ranges.get(key, (0.0, 0.0))
      return sample_uniform(lo, hi, (n,), device=self.device)

    root_pos = self.motion.root_pos_w[frames] + self._env.scene.env_origins[env_ids]
    root_pos[:, 0] += u(self.cfg.pose_range, "x")
    root_pos[:, 1] += u(self.cfg.pose_range, "y")
    root_ori = quat_mul(
      yaw_quat_from_angle(u(self.cfg.pose_range, "yaw")),
      self.motion.root_quat_w[frames],
    )
    # Differential drive: speed perturbed along the heading only, never sideways.
    yaw = heading_yaw(root_ori)
    dv = u(self.cfg.velocity_range, "x")
    root_lin_vel = self.motion.root_lin_vel_w[frames] + torch.stack(
      [dv * yaw.cos(), dv * yaw.sin(), torch.zeros_like(dv)], -1
    )
    root_ang_vel = self.motion.root_ang_vel_w[frames].clone()
    root_ang_vel[:, 2] += u(self.cfg.velocity_range, "yaw")
    joint_pos = self.joint_pos[env_ids] + sample_uniform(
      *self.cfg.joint_position_range, (n, C.NJ), device=self.device
    )
    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      self.joint_vel[env_ids],
    )
    self._pending_forward = True

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip the state joints and write the AGV pose / twist to the planar joints,
    half the column to each lift slide."""
    joint_pos = torch.clip(joint_pos, self.state_limits[:, 0], self.state_limits[:, 1])
    jm = self.jm
    qpos = torch.zeros(len(env_ids), jm.num_joints, device=self.device)
    qvel = torch.zeros_like(qpos)
    root_pos = root_pos - self._env.scene.env_origins[env_ids]
    qpos[:, jm.base[0]], qpos[:, jm.base[1]] = root_pos[:, 0], root_pos[:, 1]
    qpos[:, jm.base[2]] = heading_yaw(root_ori)
    qvel[:, jm.base[0]], qvel[:, jm.base[1]] = root_lin_vel[:, 0], root_lin_vel[:, 1]
    qvel[:, jm.base[2]] = root_ang_vel[:, 2]
    for i in jm.lift:
      qpos[:, i] = 0.5 * joint_pos[:, 0]
      qvel[:, i] = 0.5 * joint_vel[:, 0]
    qpos[:, jm.upper] = joint_pos[:, 1:]
    qvel[:, jm.upper] = joint_vel[:, 1:]
    self.robot.write_joint_state_to_sim(qpos, qvel, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _debug_vis_impl(self, visualizer) -> None:
    del visualizer  # the ghost assumes a free-joint robot


@dataclass(kw_only=True)
class G1DMotionCommandCfg(MotionCommandCfg):
  half_precision_motions: bool = False
  """Keep the motion table at half precision (``HALF_FIELDS``) to fit a large
  library next to training; float32 otherwise."""

  def build(self, env: ManagerBasedRlEnv) -> G1DMotionCommand:
    return G1DMotionCommand(self, env)


def _cmd(env: ManagerBasedRlEnv) -> G1DMotionCommand:
  return cast(G1DMotionCommand, env.command_manager.get_term("motion"))


# ---------------------------------------------------------------------------
# State and proprio
# ---------------------------------------------------------------------------


def g1d_full_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """GenCoord full state ``[root_pos, root_quat, qj, root_lin_vel_w, root_ang_vel_w,
  vj]`` with the AGV as root, read from the joints (fresh after a physics step)."""
  cmd = _cmd(env)
  jm = cmd.jm
  q, qd = cmd.robot.data.joint_pos, cmd.robot.data.joint_vel
  x, y, yaw = q[:, jm.base[0]], q[:, jm.base[1]], q[:, jm.base[2]]
  zero = torch.zeros_like(x)
  root_pos = torch.stack([x, y, torch.full_like(x, C.AGV_HEIGHT)], -1)
  lin = torch.stack([qd[:, jm.base[0]], qd[:, jm.base[1]], zero], -1)
  ang = torch.stack([zero, zero, qd[:, jm.base[2]]], -1)
  return torch.cat(
    [
      root_pos,
      yaw_quat_from_angle(yaw),
      jm.state_joints(q),
      lin,
      ang,
      jm.state_joints(qd),
    ],
    -1,
  )


def _split_state(state: torch.Tensor):
  nj = C.NJ
  return (
    state[:, 0:3],
    state[:, 3:7],
    state[:, 7 : 7 + nj],
    state[:, 7 + nj : 10 + nj],
    state[:, 10 + nj : 13 + nj],
    state[:, 13 + nj :],
  )


def g1d_projected_gravity(env: ManagerBasedRlEnv) -> torch.Tensor:
  quat = _split_state(g1d_full_state(env))[1]
  down = torch.zeros(env.num_envs, 3, device=env.device)
  down[:, 2] = -1.0
  return quat_apply_inverse(quat, down)


def g1d_base_ang_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The AGV's angular velocity in its frame (an ideal IMU on the chassis)."""
  _, quat, _, _, ang, _ = _split_state(g1d_full_state(env))
  return quat_apply_inverse(quat, ang)


def g1d_joint_pos_rel(env: ManagerBasedRlEnv) -> torch.Tensor:
  """State joints with the encoder bias, minus the defaults (``joint_pos_rel``,
  ``biased=True``); the column's bias is the sum of its slides'."""
  cmd = _cmd(env)
  biased = cmd.jm.state_joints(cmd.robot.data.joint_pos + cmd.robot.data.encoder_bias)
  return biased - biased.new_tensor(default_state_q())


def g1d_joint_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  cmd = _cmd(env)
  return cmd.jm.state_joints(cmd.robot.data.joint_vel)


def g1d_last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  return effective_action(env.action_manager.action)


# The goal terms of ``supermimic.tasks.env_cfgs`` on the G1-D state (importing
# them from there would import the task registry, which imports this module).


def g1d_goal_window(env: ManagerBasedRlEnv) -> torch.Tensor:
  cmd = _cmd(env)
  return motion_goal_window(cmd.motion, cmd.time_steps)


def _anchor_window(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  cmd = _cmd(env)
  target_pos, target_quat = motion_anchor_window(cmd.motion, cmd.time_steps)
  root = g1d_full_state(env)
  return assemble_anchor_window(root[:, :3], root[:, 3:7], target_pos, target_quat)


def g1d_anchor_window_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _anchor_window(env)[0]


def g1d_anchor_window_ori(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _anchor_window(env)[1]


def g1d_anchor_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The current reference chassis position in the robot root frame."""
  return _anchor_window(env)[0][:, :3]


def _goal_group() -> ObservationGroupCfg:
  """GenCoord's goal group with the current reference chassis position after the
  command, the layout of ``G1DRepresentation.compute_wm_goal``; the actor goal
  drops only ``motion_anchor_pos_b``, so it keeps ``anchor_pos``."""
  return ObservationGroupCfg(
    terms={
      "command": ObservationTermCfg(func=g1d_goal_window),
      "anchor_pos": ObservationTermCfg(func=g1d_anchor_pos_b),
      "motion_anchor_pos_b": ObservationTermCfg(func=g1d_anchor_window_pos),
      "motion_anchor_ori_b": ObservationTermCfg(func=g1d_anchor_window_ori),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )


def _actor_group() -> ObservationGroupCfg:
  """Sensor proprio ``[projected_gravity, base_ang_vel, joint_pos, joint_vel,
  actions]``, the G1's order; noise is owned by the representation."""
  return ObservationGroupCfg(
    terms={
      "projected_gravity": ObservationTermCfg(func=g1d_projected_gravity),
      "base_ang_vel": ObservationTermCfg(func=g1d_base_ang_vel),
      "joint_pos": ObservationTermCfg(func=g1d_joint_pos_rel),
      "joint_vel": ObservationTermCfg(func=g1d_joint_vel),
      "actions": ObservationTermCfg(func=g1d_last_action),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )


# ---------------------------------------------------------------------------
# Reward and failure rules
# ---------------------------------------------------------------------------

# (term, weight, std) of the G1's tracking reward (mjlab's tracking cfg).
TRACKING_TERMS = (
  ("motion_global_root_pos", 0.5, 0.3),
  ("motion_global_root_ori", 0.5, 0.4),
  ("motion_body_pos", 1.0, 0.3),
  ("motion_body_ori", 1.0, 0.4),
  ("motion_body_lin_vel", 1.0, 1.0),
  ("motion_body_ang_vel", 1.0, 3.14),
)
ACTION_RATE_WEIGHT = -0.1
JOINT_LIMIT_WEIGHT = -2.0


def tracking_errors(
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  body_pos: torch.Tensor,
  body_rot: torch.Tensor,
  body_lin_vel: torch.Tensor,
  body_ang_vel: torch.Tensor,
  ref_pos: torch.Tensor,
  ref_quat: torch.Tensor,
  ref_lin_vel: torch.Tensor,
  ref_ang_vel: torch.Tensor,
) -> torch.Tensor:
  """``(..., 6)`` squared errors of ``TRACKING_TERMS``, as
  :func:`supermimic.utils.tracking_reward.tracking_reward` scores them."""
  tgt_pos, tgt_mat = _reanchor_targets(ref_pos, ref_quat, root_pos, root_quat)
  return torch.stack(
    [
      ((body_pos[..., 0, :] - ref_pos[..., 0, :]) ** 2).sum(-1),
      rotation_error_magnitude(
        body_rot[..., 0, :, :], matrix_from_quat(ref_quat[..., 0, :])
      )
      ** 2,
      ((body_pos - tgt_pos) ** 2).sum(-1).mean(-1),
      (rotation_error_magnitude(body_rot, tgt_mat) ** 2).mean(-1),
      ((body_lin_vel - ref_lin_vel) ** 2).sum(-1).mean(-1),
      ((body_ang_vel - ref_ang_vel) ** 2).sum(-1).mean(-1),
    ],
    dim=-1,
  )


def _step_terms(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  """Reward kernels, penalties and failure errors on the post-step state, once per
  env step (the termination and reward managers both read them)."""
  key = int(env.common_step_counter)
  cache = env.__dict__.get("_g1d_step_terms")
  if cache is not None and cache[0] == key:
    return cache[1]
  cmd = _cmd(env)
  state = g1d_full_state(env)
  root_pos, root_quat, qj, lin, ang, vj = _split_state(state)
  pos, quat, lin_b, ang_b = cmd.motion.fk(root_pos, root_quat, qj, lin, ang, vj)
  ref_pos, ref_quat = cmd._body_pos_w, cmd._body_quat_w
  err = tracking_errors(
    root_pos,
    root_quat,
    pos,
    matrix_from_quat(quat),
    lin_b,
    ang_b,
    ref_pos,
    ref_quat,
    cmd._body_lin_vel_w,
    cmd._body_ang_vel_w,
  )
  terms = {
    name: torch.exp(-err[:, i] / std**2)
    for i, (name, _, std) in enumerate(TRACKING_TERMS)
  }
  action = effective_action(env.action_manager.action)
  prev = effective_action(env.action_manager.prev_action)
  terms["action_rate_l2"] = ((action - prev) ** 2).sum(-1)
  lo, hi = cmd.state_limits.unbind(-1)
  terms["joint_limit"] = ((lo - qj).clamp(min=0) + (qj - hi).clamp(min=0)).sum(-1)
  offset = env.__dict__.setdefault(
    "_g1d_torso_offset", torso_offset_xz(env.sim.mj_model)
  )
  terms["failure"] = failure_errors(
    root_pos, pos, quat, root_quat, ref_pos, ref_quat, offset
  )
  tgt_pos, _ = _reanchor_targets(ref_pos, ref_quat, root_pos, root_quat)
  terms["upper_err"] = (tgt_pos[:, 1:] - pos[:, 1:]).norm(dim=-1).mean(-1)
  terms["joint_err"] = (qj[:, 1:] - cmd.joint_pos[:, 1:]).abs().mean(-1)
  terms["column_err"] = (qj[:, 0] - cmd.joint_pos[:, 0]).abs()
  env.__dict__["_g1d_step_terms"] = (key, terms)
  return terms


def g1d_reward_term(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  return _step_terms(env)[name]


def g1d_failure(env: ManagerBasedRlEnv, rule: str, threshold: float) -> torch.Tensor:
  return _step_terms(env)["failure"][:, FAILURE_RULES.index(rule)] > threshold


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def g1d_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1-D tracking with GenCoord's observation groups (``actor``, ``full_state``,
  ``goal``) and the three DR events the sim context reads."""
  commands = {
    "motion": G1DMotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=False,
      motion_file="",
      anchor_body_name=C.ROOT_BODY,
      body_names=C.TRACKED_BODIES,
      pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.2, 0.2)},
      velocity_range={"x": (-0.5, 0.5), "yaw": (-0.78, 0.78)},
      joint_position_range=(-0.1, 0.1),
    )
  }
  observations = {
    "actor": _actor_group(),
    "full_state": ObservationGroupCfg(
      terms={"state": ObservationTermCfg(func=g1d_full_state)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "goal": _goal_group(),
  }
  # Event names are the ones SimContext reads.
  events = {
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=COLLISION_GEOMS),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(C.LIFT_JOINTS[0], *C.UPPER_JOINTS)
        ),
        "bias_range": (-0.01, 0.01),
      },
    ),
  }
  rewards = {
    name: RewardTermCfg(func=g1d_reward_term, weight=weight, params={"name": name})
    for name, weight, _ in TRACKING_TERMS
  }
  rewards["action_rate_l2"] = RewardTermCfg(
    func=g1d_reward_term, weight=ACTION_RATE_WEIGHT, params={"name": "action_rate_l2"}
  )
  rewards["joint_limit"] = RewardTermCfg(
    func=g1d_reward_term, weight=JOINT_LIMIT_WEIGHT, params={"name": "joint_limit"}
  )
  thresholds = (C.CHASSIS_THRESHOLD, C.UPPER_THRESHOLD, C.COLUMN_THRESHOLD)
  terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    "motion_end": TerminationTermCfg(
      func=track_mdp.motion_end, time_out=True, params={"command_name": "motion"}
    ),
  }
  for rule, threshold in zip(FAILURE_RULES, thresholds, strict=True):
    terminations[rule] = TerminationTermCfg(
      func=g1d_failure, params={"rule": rule, "threshold": threshold}
    )

  cfg = ManagerBasedRlEnvCfg(
    # Every env origin at 0: the base is fixed to the world through the planar
    # joints and every env is its own MuJoCo Warp world, so nothing overlaps.
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=0.0,
      entities={"robot": get_g1d_robot_cfg()},
    ),
    observations=observations,
    actions={"joint_pos": G1DActionCfg(entity_name="robot")},
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
  if play:
    cfg.episode_length_s = int(1e9)
    motion = cfg.commands["motion"]
    assert isinstance(motion, G1DMotionCommandCfg)
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)
    motion.sampling_mode = "start"
  return cfg
