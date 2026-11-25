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

import copy
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
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
  stiff_drive_params,
  upper_action_scale,
)
from supermimic.representations import (
  assemble_anchor_window,
  motion_anchor_window,
  motion_goal_window,
)
from supermimic.utils import distributed
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


def _reanchored_pos(
  ref_pos: torch.Tensor,
  ref_quat: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
) -> torch.Tensor:
  """The reference body positions re-anchored at the robot root (xy + yaw), as the
  reward re-anchors them."""
  zero = torch.zeros_like(ref_pos)
  aux = torch.cat([root_pos, root_quat], dim=-1)
  return _reanchor_targets(ref_pos, ref_quat, zero, zero, aux)[0]


def failure_errors(
  root_pos: torch.Tensor,
  body_pos: torch.Tensor,
  body_quat: torch.Tensor,
  root_quat: torch.Tensor,
  ref_pos: torch.Tensor,
  ref_quat: torch.Tensor,
  offset_xz: tuple[float, float],
  world_upper: bool = False,
) -> torch.Tensor:
  """``(..., 3)`` errors of the failure rules: chassis world xy distance, largest
  upper-body distance to its reference re-anchored at the chassis (xy + yaw), or
  in the world frame with ``world_upper``, column top (Yaw_Link) height
  difference. Bodies are ``C.TRACKED_BODIES``."""
  chassis = (ref_pos[..., 0, :2] - root_pos[..., :2]).norm(dim=-1)
  tgt_pos = (
    ref_pos if world_upper else _reanchored_pos(ref_pos, ref_quat, root_pos, root_quat)
  )
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

  ``cfg.stiff_chassis``: the (vx, wz) command passes a first-order filter (TAU) and
  stiff, traction-limited chassis drives hold the filtered speed, so the upper body
  cannot push the chassis around; otherwise the drives themselves are the TAU lag.
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
    self.plant = {"chassis": "stiff" if cfg.stiff_chassis else "soft"}
    if cfg.stiff_chassis:
      self._stiffen_chassis_drives()
      self._alpha = 1.0 - math.exp(-env.physics_dt / C.TAU)
      self._cmd_vel = torch.zeros(self.num_envs, 2, device=dev)
      self._filt_vel = torch.zeros(self.num_envs, 2, device=dev)
      # A reset writes the start velocity after this term resets: restart the
      # filter from the chassis speed at the next step.
      self._restart = torch.ones(self.num_envs, dtype=torch.bool, device=dev)

  def _stiffen_chassis_drives(self) -> None:
    """Chassis velocity gains over STIFF_LOOP_TAU and traction force limits, in the
    compiled model and every world (no event randomizes these actuators)."""
    sim = self._env.sim
    p = stiff_drive_params()
    gains = (p["kv_xy"], p["kv_xy"], p["kv_yaw"])
    limits = (C.CHASSIS_FORCE_LIMIT, C.CHASSIS_FORCE_LIMIT, C.CHASSIS_TORQUE_LIMIT)

    def world_rows(field: str):
      # A field nothing expanded is one row shared by every world.
      return slice(None) if field in sim.expanded_fields else 0

    for joint, kv, limit in zip(C.BASE_JOINTS, gains, limits, strict=True):
      i = sim.mj_model.actuator(f"{self.cfg.entity_name}/{joint}").id
      sim.mj_model.actuator_gainprm[i, 0] = kv
      sim.mj_model.actuator_biasprm[i, 2] = -kv
      sim.mj_model.actuator_forcerange[i] = (-limit, limit)
      sim.model.actuator_gainprm[world_rows("actuator_gainprm"), i, 0] = kv
      sim.model.actuator_biasprm[world_rows("actuator_biasprm"), i, 2] = -kv
      w = world_rows("actuator_forcerange")
      sim.model.actuator_forcerange[w, i, 0] = -limit
      sim.model.actuator_forcerange[w, i, 1] = limit

  def _chassis_speed(self) -> torch.Tensor:
    """(forward speed along the heading, yaw rate) of the chassis."""
    q, v = self._entity.data.joint_pos, self._entity.data.joint_vel
    yaw = q[:, self.jm.base[2]]
    fwd = v[:, self.jm.base[0]] * yaw.cos() + v[:, self.jm.base[1]] * yaw.sin()
    return torch.stack([fwd, v[:, self.jm.base[2]]], -1)

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
    self._lift_vel = (0.5 * vz).unsqueeze(-1).expand(-1, 2)
    if self.cfg.stiff_chassis:
      self._cmd_vel = torch.stack([vx, wz], -1)
      self._filt_vel = torch.where(
        self._restart.unsqueeze(-1), self._chassis_speed(), self._filt_vel
      )
      self._restart.zero_()
      return
    # Differential drive: the forward speed is along the heading, never sideways.
    yaw = robot.data.joint_pos[:, self.jm.base[2]]
    self._base_vel = torch.stack([vx * yaw.cos(), vx * yaw.sin(), wz], -1)

  def apply_actions(self) -> None:
    robot = self._entity
    if self.cfg.stiff_chassis:  # one filter step per physics step
      self._filt_vel += self._alpha * (self._cmd_vel - self._filt_vel)
      fwd, wz = self._filt_vel.unbind(-1)
      yaw = robot.data.joint_pos[:, self.jm.base[2]]
      self._base_vel = torch.stack([fwd * yaw.cos(), fwd * yaw.sin(), wz], -1)
    robot.set_joint_position_target(self._upper_target, joint_ids=self.upper_ids)
    robot.set_joint_velocity_target(self._base_vel, joint_ids=self.base_ids)
    # The lift's gravity feed-forward lives in its actuator.
    robot.set_joint_velocity_target(self._lift_vel, joint_ids=self.lift_ids)

  def reset(self, env_ids=None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._raw[ids] = 0.0
    if self.cfg.stiff_chassis:
      self._restart[ids] = True


@dataclass(kw_only=True)
class G1DActionCfg(ActionTermCfg):
  stiff_chassis: bool = False
  """Stiff, traction-limited chassis drives behind a TAU command filter (see
  ``G1DAction``); False keeps the soft TAU drives of SuperMimic-Reproduced. The
  checkpoint records it (``run.json`` "plant"), and evaluation follows it."""

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


MOTION_FIELDS = (
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
  "joint_pos",
  "joint_vel",
)


def shard_clips(motion_ends: np.ndarray, rank: int, world: int) -> np.ndarray:
  """Clips of one rank's shard: every ``world``-th clip from ``rank``, so each shard
  mixes the datasets of a packed directory (stored one after another) alike."""
  return np.arange(rank, len(motion_ends), world)


def load_motion_shard(
  root: str, fk, loader_cls, rank: int, world: int, half_precision: bool, device: str
):
  """A ``MotionLoader`` over one rank's shard (``shard_clips``) of a packed motion
  directory: only the shard's frames are read (memory-mapped) and kept."""
  ends = np.load(os.path.join(root, "motion_ends.npy"))
  with open(os.path.join(root, "joint_names.json")) as fh:
    joints = json.load(fh)
  if joints != fk.joint_names:
    raise ValueError(f"motion joints {joints} are not the robot's {fk.joint_names}")
  names_path = os.path.join(root, "motion_names.json")
  names = None
  if os.path.exists(names_path):
    with open(names_path) as fh:
      names = json.load(fh)
  starts = np.concatenate([[0], ends[:-1]])
  clips = shard_clips(ends, rank, world)
  lengths = ends[clips] - starts[clips]
  motion = object.__new__(loader_cls)
  for name in MOTION_FIELDS:
    src = np.load(os.path.join(root, f"{name}.npy"), mmap_mode="r")
    out = np.empty((int(lengths.sum()), src.shape[1]), dtype=src.dtype)
    row = 0
    for c, n in zip(clips, lengths, strict=True):
      out[row : row + n] = src[starts[c] : starts[c] + n]
      row += n
    table = torch.from_numpy(out)
    if half_precision and name in HALF_FIELDS:
      setattr(motion, name, HalfTable(table, device))
    else:
      setattr(motion, name, table.to(device=device, dtype=torch.float32))
  motion.fk = fk
  motion.time_step_total = int(lengths.sum())
  motion.motion_ends = torch.as_tensor(
    np.cumsum(lengths), dtype=torch.long, device=device
  )
  all_names = names or [f"clip_{i}" for i in range(len(ends))]
  motion.motion_names = [all_names[c] for c in clips]
  return motion


def set_bin_count(cmd, bin_count: int) -> None:
  """Re-cut a ``MotionCommand``'s reset-sampling timeline into ``bin_count`` bins
  (its failure counts and external distribution restart uniform)."""
  cmd.bin_count = int(bin_count)
  dev = cmd._external_probs.device
  cmd.bin_failed_count = torch.zeros(cmd.bin_count, device=dev)
  cmd._current_bin_failed = torch.zeros(cmd.bin_count, device=dev)
  cmd._external_probs = torch.full((cmd.bin_count,), 1.0 / cmd.bin_count, device=dev)


@contextmanager
def _g1d_motion_loading(
  body_names: tuple[str, ...], half_precision: bool, shard: bool = False
):
  """Have ``MotionCommand.__init__`` build the G1-D kinematics (mjlab's needs a
  free-joint root) and, with ``half_precision``, a motion table that keeps
  ``HALF_FIELDS`` at half precision (loaded on the CPU first, so the float32
  copy never reaches the GPU). With ``shard`` and several ranks, each rank loads
  only its shard of a packed directory (``load_motion_shard``)."""
  original_fk = commands_module.ForwardKinematics
  original_loader = commands_module.MotionLoader

  def build_fk(mj_model, body_ids, joint_q_adr, body_rows, device="cpu", **_):
    del body_ids, joint_q_adr, body_rows
    return G1DForwardKinematics(mj_model, body_names, device=device)

  def build_loader(motion_file, fk, device="cpu"):
    world = distributed.world_size()
    if shard and world > 1:
      files = commands_module._resolve_motion_file(motion_file)
      if not (len(files) == 1 and os.path.isdir(files[0])):
        raise ValueError("shard_motions needs a packed motion directory")
      return load_motion_shard(
        files[0], fk, original_loader, distributed.rank(), world, half_precision, device
      )
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


def ghost_model(mj_model, rgba) -> object:
  """A copy of the scene showing only the robot, in ``rgba``: its visual geoms
  when it has them (``visual_mesh_dir``), else its colliders."""
  model = copy.deepcopy(mj_model)
  # Visual geoms are unnamed: find the robot's geoms by their bodies.
  robot = [
    g
    for g in range(model.ngeom)
    if model.body(model.geom_bodyid[g]).name.startswith("robot/")
  ]
  visual = [
    g for g in robot if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
  ]
  shown = set(visual or robot)
  for g in range(model.ngeom):
    if g in shown:
      model.geom_rgba[g] = rgba
    else:
      model.geom_rgba[g, 3] = 0
  return model


class G1DMotionCommand(MotionCommand):
  """``MotionCommand`` on the G1-D: the reference joints are the 17 state joints
  and the root is ``AGV_link``, written to the planar and lift joints."""

  cfg: G1DMotionCommandCfg
  forced_frames: torch.Tensor | None = None
  """Per-env reset frames overriding the sampling mode (the benchmark plays each
  clip from its first frame)."""

  def __init__(self, cfg: G1DMotionCommandCfg, env: ManagerBasedRlEnv):
    with _g1d_motion_loading(
      tuple(cfg.body_names), cfg.half_precision_motions, cfg.shard_motions
    ):
      super().__init__(cfg, env)
    self.jm = G1DJointMap(self.robot.joint_names)
    self.state_limits = state_soft_limits(self.robot)
    world = distributed.world_size()
    if cfg.shard_motions and cfg.shard_fine_bins and world > 1:
      set_bin_count(self, self.bin_count * world)
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

  def ghost_qpos(
    self, root_pos: torch.Tensor, root_quat: torch.Tensor, joint_pos: torch.Tensor
  ) -> np.ndarray:
    """Scene qpos posing the robot at a root (AGV) pose ``(3,)`` / ``(4,)`` with the
    17 state joints, the rest of the scene at zero: for ghost renderings."""
    jm = self.jm
    root_pos, root_quat, joint_pos = (
      t.detach().float().cpu().numpy() for t in (root_pos, root_quat, joint_pos)
    )
    w, x, y, z = root_quat
    joints = np.zeros(jm.num_joints)
    joints[jm.base[0]], joints[jm.base[1]] = root_pos[0], root_pos[1]
    joints[jm.base[2]] = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    joints[jm.lift] = 0.5 * joint_pos[0]
    joints[jm.upper] = joint_pos[1:]
    qpos = np.zeros(self._env.sim.mj_model.nq)
    qpos[self.robot.indexing.joint_q_adr.cpu().numpy()] = joints
    return qpos

  def _debug_vis_impl(self, visualizer) -> None:
    """The reference as a ghost robot (the free-joint ghost of ``MotionCommand``
    does not apply to the planar root)."""
    if self.cfg.viz.mode != "ghost":
      return
    if self._ghost_model is None:
      self._ghost_model = ghost_model(self._env.sim.mj_model, self._ghost_color)
    origins = self._env.scene.env_origins
    for batch in visualizer.get_env_indices(self.num_envs):
      frame = self.time_steps[batch]
      qpos = self.ghost_qpos(
        self.motion.root_pos_w[frame] + origins[batch],
        self.motion.root_quat_w[frame],
        self.motion.joint_pos[frame],
      )
      visualizer.add_ghost_entity(
        qpos,
        model=self._ghost_model,
        alpha=float(self._ghost_color[3]),
        label=f"ghost_{batch}",
      )


@dataclass(kw_only=True)
class G1DMotionCommandCfg(MotionCommandCfg):
  half_precision_motions: bool = False
  """Keep the motion table at half precision (``HALF_FIELDS``) to fit a large
  library next to training; float32 otherwise."""
  shard_motions: bool = False
  """With several ranks, each loads only its shard of the packed motion directory
  (every world-size-th clip), so a card holds 1 / world of the library instead of
  all of it. Evaluation (one rank) always loads the whole directory."""
  shard_fine_bins: bool = False
  """With ``shard_motions``, cut each shard into world-size times as many
  reset-sampling bins, about as many as the whole library has: every rank's reset
  sampler then gets the samples per bin of an unsharded run instead of
  world-size times more."""
  world_reward: bool = False
  """Score every tracked body against its reference in the world frame, as ScaleBFM
  does, instead of re-anchoring the bodies at the robot's chassis; the two chassis
  terms go (``apply_reward_rule``). The world model's reward follows."""
  world_failure: bool = False
  """The upper-body failure rule measures each body against its reference in the
  world frame instead of re-anchored at the robot's chassis; the world model's
  imagined termination follows. The thresholds stay the terminations' own."""

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
# ``world_reward``: the body terms alone, each body against its reference in the
# world frame (ScaleBFM's rule); the chassis is a tracked body.
WORLD_TRACKING_TERMS = TRACKING_TERMS[2:]
# The wrist links, whose world-frame error the benchmark reports.
WRIST_ROWS = [
  C.TRACKED_BODIES.index(n) for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")
]


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
  tgt_pos, tgt_mat, tgt_lin_vel, tgt_ang_vel = _reanchor_targets(
    ref_pos, ref_quat, ref_lin_vel, ref_ang_vel, torch.cat([root_pos, root_quat], -1)
  )
  return torch.stack(
    [
      ((body_pos[..., 0, :] - ref_pos[..., 0, :]) ** 2).sum(-1),
      rotation_error_magnitude(
        body_rot[..., 0, :, :], matrix_from_quat(ref_quat[..., 0, :])
      )
      ** 2,
      ((body_pos - tgt_pos) ** 2).sum(-1).mean(-1),
      (rotation_error_magnitude(body_rot, tgt_mat) ** 2).mean(-1),
      ((body_lin_vel - tgt_lin_vel) ** 2).sum(-1).mean(-1),
      ((body_ang_vel - tgt_ang_vel) ** 2).sum(-1).mean(-1),
    ],
    dim=-1,
  )


def world_tracking_errors(
  body_pos: torch.Tensor,
  body_rot: torch.Tensor,
  body_lin_vel: torch.Tensor,
  body_ang_vel: torch.Tensor,
  ref_pos: torch.Tensor,
  ref_quat: torch.Tensor,
  ref_lin_vel: torch.Tensor,
  ref_ang_vel: torch.Tensor,
) -> torch.Tensor:
  """``(..., 4)`` squared errors of ``WORLD_TRACKING_TERMS``: every tracked body
  against its reference in the world frame, nothing re-anchored."""
  return torch.stack(
    [
      ((body_pos - ref_pos) ** 2).sum(-1).mean(-1),
      (rotation_error_magnitude(body_rot, matrix_from_quat(ref_quat)) ** 2).mean(-1),
      ((body_lin_vel - ref_lin_vel) ** 2).sum(-1).mean(-1),
      ((body_ang_vel - ref_ang_vel) ** 2).sum(-1).mean(-1),
    ],
    dim=-1,
  )


def world_tracking_reward(
  body_pos: torch.Tensor,
  body_rot: torch.Tensor,
  body_lin_vel: torch.Tensor,
  body_ang_vel: torch.Tensor,
  joint_state: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  ref_pos: torch.Tensor,
  ref_quat: torch.Tensor,
  ref_lin_vel: torch.Tensor,
  ref_ang_vel: torch.Tensor,
  soft_joint_pos_limits: torch.Tensor,
  dt: float,
) -> torch.Tensor:
  """The ``world_reward`` env's step reward, as its reward manager sums
  ``_terms_from``: ``action`` and ``last_action`` are the effective ones and
  ``joint_state`` is ``[qj, vj]``."""
  err = world_tracking_errors(
    body_pos,
    body_rot,
    body_lin_vel,
    body_ang_vel,
    ref_pos,
    ref_quat,
    ref_lin_vel,
    ref_ang_vel,
  )
  reward = sum(
    weight * torch.exp(-err[..., i] / std**2)
    for i, (_, weight, std) in enumerate(WORLD_TRACKING_TERMS)
  )
  action_rate = ((action - last_action) ** 2).sum(-1)
  joint_pos, _ = joint_state.chunk(2, dim=-1)
  lo, hi = soft_joint_pos_limits.unbind(-1)
  joint_limit = ((lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)).sum(-1)
  return dt * (
    reward + ACTION_RATE_WEIGHT * action_rate + JOINT_LIMIT_WEIGHT * joint_limit
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
  offset = env.__dict__.setdefault(
    "_g1d_torso_offset", torso_offset_xz(env.sim.mj_model)
  )
  terms = (_compiled_terms_from if state.is_cuda else _terms_from)(
    cmd.motion.fk,
    state,
    cmd._body_pos_w,
    cmd._body_quat_w,
    cmd._body_lin_vel_w,
    cmd._body_ang_vel_w,
    env.action_manager.action,
    env.action_manager.prev_action,
    cmd.state_limits,
    cmd.joint_pos,
    offset,
    cmd.cfg.world_reward,
    cmd.cfg.world_failure,
  )
  env.__dict__["_g1d_step_terms"] = (key, terms)
  return terms


def _terms_from(
  fk: G1DForwardKinematics,
  state: torch.Tensor,
  ref_pos: torch.Tensor,
  ref_quat: torch.Tensor,
  ref_lin_vel: torch.Tensor,
  ref_ang_vel: torch.Tensor,
  action: torch.Tensor,
  prev_action: torch.Tensor,
  state_limits: torch.Tensor,
  ref_joint_pos: torch.Tensor,
  offset: tuple[float, float],
  world_reward: bool = False,
  world_failure: bool = False,
) -> dict[str, torch.Tensor]:
  """``_step_terms`` from the post-step state and the reference."""
  root_pos, root_quat, qj, lin, ang, vj = _split_state(state)
  pos, quat, lin_b, ang_b = fk(root_pos, root_quat, qj, lin, ang, vj)
  rot = matrix_from_quat(quat)
  refs = (ref_pos, ref_quat, ref_lin_vel, ref_ang_vel)
  if world_reward:
    err = world_tracking_errors(pos, rot, lin_b, ang_b, *refs)
    rule = WORLD_TRACKING_TERMS
  else:
    err = tracking_errors(root_pos, root_quat, pos, rot, lin_b, ang_b, *refs)
    rule = TRACKING_TERMS
  terms = {
    name: torch.exp(-err[:, i] / std**2) for i, (name, _, std) in enumerate(rule)
  }
  action, prev = effective_action(action), effective_action(prev_action)
  terms["action_rate_l2"] = ((action - prev) ** 2).sum(-1)
  lo, hi = state_limits.unbind(-1)
  terms["joint_limit"] = ((lo - qj).clamp(min=0) + (qj - hi).clamp(min=0)).sum(-1)
  terms["failure"] = failure_errors(
    root_pos, pos, quat, root_quat, ref_pos, ref_quat, offset, world_failure
  )
  tgt_pos = _reanchored_pos(ref_pos, ref_quat, root_pos, root_quat)
  terms["upper_err"] = (tgt_pos[:, 1:] - pos[:, 1:]).norm(dim=-1).mean(-1)
  world = (ref_pos - pos).norm(dim=-1)
  terms["upper_world_err"] = world[:, 1:].mean(-1)
  terms["wrist_world_err"] = world[:, WRIST_ROWS].mean(-1)
  terms["joint_err"] = (qj[:, 1:] - ref_joint_pos[:, 1:]).abs().mean(-1)
  terms["column_err"] = (qj[:, 0] - ref_joint_pos[:, 0]).abs()
  return terms


# The env step is launch-bound: eagerly these terms are some 900 small kernels.
_compiled_terms_from = torch.compile(_terms_from, dynamic=False)


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


def apply_reward_rule(cfg: ManagerBasedRlEnvCfg) -> None:
  """Drop the two chassis terms under ``world_reward``: the chassis is a tracked
  body, so the world-frame body terms already score where it is. Call once the
  command cfg is final, before the env is built."""
  motion = cfg.commands["motion"]
  assert isinstance(motion, G1DMotionCommandCfg)
  if motion.world_reward:
    for name, _, _ in TRACKING_TERMS[:2]:
      cfg.rewards.pop(name, None)
