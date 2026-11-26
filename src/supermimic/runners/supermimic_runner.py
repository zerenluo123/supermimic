from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import torch
import torch.nn as nn
import tyro
import wandb
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv, sample_start
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics, MultiEncoderAE, build_mlp
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.representations import StateRepresentation, create_representation
from supermimic.runners.exporter import export_actor_as_onnx
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.sim_context import SimContext
from supermimic.tasks.task_spec import (
  TasksCfg,
  build_actor_obs_td,
  resolve_task_specs,
)
from supermimic.utils.distributed import broadcast_module, is_main
from supermimic.utils.helpers import (
  extract_action_scale_offset,
  unpack_obs_dims,
)
from supermimic.utils.logger import Logger
from supermimic.utils.wandb_artifacts import download_model_dir, log_model_artifact

if TYPE_CHECKING:
  from mjlab.rl import RslRlVecEnvWrapper

BpttCfg = Annotated[BpttPolicyLearnerCfg, tyro.conf.subcommand(name="bptt")]
ActorCriticCfg = Annotated[
  ActorCriticPolicyLearnerCfg, tyro.conf.subcommand(name="actor-critic")
]


@dataclass
class SuperMimicRunnerCfg:
  wandb_project: str = "supermimic"
  run_name: str | None = None
  wandb_run_path: str | None = None
  headless: bool = True
  buffer_size: int = 1000
  representation: str = "gencoord"  # "gencoord" or "maxcoord"
  actor_obs: str = "rep_state"  # "rep_state" or "proprio"
  g1d_actor_goal: Literal["joints", "urr", "urr_robot", "urr_robot_rel"] = "joints"
  """G1-D actor goal per window frame: the reference joints and their velocities;
  the reference URR keypoint pose (chest and hands, elbow positions) in that frame's
  reference chassis frame; or ScaleBFM's per-body targets in the robot's own chassis
  frame for the chassis and the keypoints, each with its orientation, as
  ``[pos, rot]`` (``urr_robot``) or ``[pos, pos_rel, rot, rot_rel]``
  (``urr_robot_rel``; both need ``g1d_chassis_window``). The critic keeps the
  joints."""
  g1d_critic_urr: bool = False
  """G1-D: also give the critic the window's URR keypoint poses and twists and the
  robot's own keypoints."""
  g1d_chassis_window: bool = False
  """G1-D actor goal: each window frame's reference chassis position and heading
  relative to the robot, in place of the current frame's position alone; the
  odometry noise goes on the robot's own pose."""
  g1d_urr_table: bool = True
  """G1-D: tabulate the motion's URR keypoints once (on the GPU); False computes
  them from the reference joints on each use instead, the same values without the
  table's memory."""
  g1d_planar_chassis: bool = False
  """G1-D world model: keep the chassis in the plane (x, y, yaw) as the plant does,
  dropping the predicted Δv_z, Δw_x, Δw_y. Off, the imagined chassis can lift and
  tilt; its height and tilt channels, constant in every collected state, then enter
  the dynamics through a zero-std normalizer and the actor gradient can blow up."""
  g1d_diff_drive: bool = False
  """G1-D world model, with ``g1d_planar_chassis``: the imagined chassis never moves
  sideways in its own frame, as the plant's differential drive: the predicted Δv_y
  is dropped and v_y held at zero. Off, the barely-trained Δv_y head (the loss sees
  v_y only through the integrated root position) can drive v_y far outside the data,
  where its 0.017 normalizer scale makes the imagined loop unstable."""
  g1d_urr_body_point: Literal["chest", "waist"] = "waist"
  """G1-D URR body keypoint, for the actor goal and the critic's keypoints: the
  ``waist``, the torso_link origin (the waist yaw joint), as ScaleBFM places its
  points at link origins, or the ``chest``, 0.2 m above it along its z (the
  g1d-urr controller's). Runs saved before this field existed used the chest
  (``run_config``)."""
  g1d_goal_modes: dict[str, float] = field(default_factory=dict)
  """G1-D: train the actor on ScaleBFM's modes, e.g.
  ``"{'full': 0.6, 'three_point': 0.4}"``: at every reset each env draws one
  (``C.GOAL_MODES``) with its probability and keeps it for the episode; imagined
  rollouts keep their start's. The actor's normalized goal has the mode's withheld
  parts zeroed, and the actor and the critic read the mode's bit per part. Needs a
  robot-frame ``g1d_actor_goal``. Empty: the whole goal, no mode."""
  log_interval: int = 10
  save_interval: int = 1000
  keep_checkpoints: bool = False
  """Also keep every saved checkpoint under ``<log_dir>/checkpoints/<step>/`` with
  the runner cfg, for offline evaluation of the whole run."""
  history_length: int = 1
  termination_softness: float = 0.05
  dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  dynamics_bf16: bool = True
  """Run the dynamics MLP under bfloat16 autocast (upstream d88804a); False keeps it
  in float32. The imagined actor has its own switch, ``policy.bf16``."""
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)
  max_iterations: int = 1000000
  replay_ratio: int = 16
  latent_dim: int = 32
  actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden_dims: tuple[int, ...] = (256, 128)
  latent_bc_weight: float = 1e-1
  """Weight on the latent BC loss pulling non-anchor encoders toward the
  anchor encoder's latent on the same state. 0 disables."""
  tasks: TasksCfg = field(default_factory=TasksCfg)
  """Per-task hyperparameters."""
  slot_mix: dict[str, float] = field(
    default_factory=lambda: {"motion": 1.0, "velocity": 0.0}
  )
  """Fractional slot allocation per task. Must sum to 1.0."""
  policy: BpttCfg | ActorCriticCfg = field(default_factory=ActorCriticCfg)

  sampling_mode: Literal["uniform", "adaptive", "start", "return"] = "return"
  """RSI reset distribution: 'return' (concentrate resets on the bins where the
  policy earns low return; see ReturnSampler) or one of mjlab's native modes.
  train.py resolves this into the motion command's cfg before the env is built
  ('return' drives mjlab's 'external' mode)."""
  return_sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)


@dataclass
class WorldModel:
  """The world model plus the env dimensions that sized it."""

  representation: StateRepresentation
  dynamics: Dynamics
  sim_context: SimContext
  num_goals: int
  num_actions: int

  def bundle(self) -> dict:
    """The ``world_model.pt`` contents."""
    return {
      "dynamics": self.dynamics.state_dict(),
      "sim_context": self.sim_context.state_dict(),
    }

  def load_bundle(self, bundle: dict) -> None:
    """Load a ``world_model.pt`` bundle."""
    self.dynamics.load_state_dict(bundle["dynamics"])
    self.sim_context.load_state_dict(bundle["sim_context"])


def build_world_model(
  cfg: SuperMimicRunnerCfg, env: RslRlVecEnvWrapper, device: str
) -> WorldModel:
  """Construct the representation and dynamics for ``env`` sized by ``cfg``.

  Resets ``env`` to read its observation dimensions.
  """
  if cfg.history_length > 1 and cfg.actor_obs != "proprio":
    raise RuntimeError(
      "history_length > 1 requires actor_obs='proprio' (--runner.actor-obs proprio)"
    )

  obs, _ = env.reset()
  num_full_state, num_goals = unpack_obs_dims(obs)
  num_actions = env.num_actions

  print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

  sim_context = SimContext.from_env(env.unwrapped)
  sim_ctx_dim = sim_context.output_dim
  print(
    f"[INFO] Sim Context: {sim_ctx_dim} dim ({[s.name for s in sim_context.specs]})"
  )

  default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0].to(device)
  representation = create_representation(
    cfg.representation,
    num_full_state,
    num_actions,
    env.unwrapped.step_dt,
    env.unwrapped.cfg.decimation,
    default_joint_pos,
    actor_obs_mode=cfg.actor_obs,
    g1d_actor_goal=cfg.g1d_actor_goal,
    g1d_critic_urr=cfg.g1d_critic_urr,
    g1d_chassis_window=cfg.g1d_chassis_window,
    g1d_urr_table=cfg.g1d_urr_table,
    g1d_planar_chassis=cfg.g1d_planar_chassis,
    g1d_urr_body_point=cfg.g1d_urr_body_point,
    g1d_goal_modes=bool(cfg.g1d_goal_modes),
    g1d_diff_drive=cfg.g1d_diff_drive,
  ).to(device)

  representation.bind(env.unwrapped, sim_context)

  print(
    f"[INFO] Representation: {cfg.representation} "
    f"(full={representation.full_state_dim}, rep={representation.rep_state_dim}, "
    f"pred={representation.prediction_dim})"
  )

  action_scale, action_offset = extract_action_scale_offset(env)
  action_manager = env.unwrapped.action_manager
  action_term = action_manager.get_term(action_manager.active_terms[0])
  dynamics = Dynamics(
    representation,
    num_actions,
    sim_ctx_dim,
    action_scale,
    action_offset,
    hidden_dims=cfg.dynamics_hidden_dims,
    tanh_mask=getattr(action_term, "tanh_mask", None),
    bf16=cfg.dynamics_bf16,
  ).to(device)

  return WorldModel(
    representation=representation,
    dynamics=dynamics,
    sim_context=sim_context,
    num_goals=num_goals,
    num_actions=num_actions,
  )


def build_actor(
  cfg: SuperMimicRunnerCfg, wm: WorldModel, device: str
) -> MultiEncoderAE:
  """Actor sized by ``cfg``; the resolved task specs live on ``actor.task_specs``."""
  representation = wm.representation
  obs_dim = (
    representation.rep_state_dim
    if cfg.actor_obs == "rep_state"
    else representation.proprio_obs_dim(wm.num_actions)
  )
  return MultiEncoderAE(
    obs_dim=obs_dim,
    action_dim=wm.num_actions,
    task_specs=resolve_task_specs(cfg.tasks, representation, wm.num_goals, device),
    slot_mix=cfg.slot_mix,
    latent_dim=cfg.latent_dim,
    hidden_dims=cfg.actor_hidden_dims,
    latent_bc_weight=cfg.latent_bc_weight,
    history_length=cfg.history_length,
  ).to(device)


def load_wandb_checkpoint(wandb_run_path: str, wm: WorldModel, device: str) -> None:
  """Warm-start ``wm`` from another W&B run's world-model bundle."""
  print(f"[INFO] Loading World Model from W&B: {wandb_run_path}")
  run = wandb.Api().run(wandb_run_path)
  wm.load_bundle(
    torch.load(download_model_dir(run) / "world_model.pt", map_location=device)
  )


class Ratio:
  def __init__(self, ratio: float) -> None:
    self._ratio = ratio
    self._prev: int | None = None

  def __call__(self, step: int) -> int:
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._ratio < 0:
      return 1
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


class SuperMimicRunner:
  """Trains the world model and the multi-encoder actor.

  Each task owns a contiguous slot of the env batch and its own goal encoder.
  All tasks share the motion-tracking reward.
  """

  cfg: SuperMimicRunnerCfg

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    cfg: SuperMimicRunnerCfg,
    task_id: str,
    registry_name: str | None = None,
    device: str = "cuda:0",
  ):
    self.env = env
    self.cfg = cfg
    self.task_id = task_id
    self.registry_name = registry_name
    self.device = device

    self._init_common()
    self._init_policy()

    self.logger = Logger(
      run_name=cfg.run_name,
      num_envs=env.num_envs,
      max_episode_length=env.max_episode_length,
      device=device,
    )

  def _init_common(self) -> None:
    wm = build_world_model(self.cfg, self.env, self.device)
    self._wm = wm
    self._representation = wm.representation
    self._dynamics = wm.dynamics
    self._sim_context = wm.sim_context
    self._num_goals = wm.num_goals
    self._num_actions = wm.num_actions

    if self.cfg.wandb_run_path is not None:
      load_wandb_checkpoint(self.cfg.wandb_run_path, wm, self.device)

    self.action_manager = self.env.unwrapped.action_manager
    self.motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    self.motion = self.motion_cmd.motion
    # A goal in parts (ScaleBFM's modes): the motion command's per-env ``mode``,
    # drawn from the modes the actor trains on (train.py resolves them).
    parts = self._representation.actor_goal_parts()
    self._num_goal_parts = 0 if parts is None else int(parts.max()) + 1
    env_modes = getattr(self.motion_cmd.cfg, "goal_modes", {})
    if dict(env_modes) != dict(self.cfg.g1d_goal_modes):
      raise RuntimeError(
        f"the actor trains on goal modes {self.cfg.g1d_goal_modes} but the env "
        f"draws {env_modes}: set the motion command's goal_modes to match"
      )

    self.replay_buffer = ReplayBuffer(
      schema=self._buffer_schema(),
      buffer_size=self.cfg.buffer_size,
      num_envs=self.env.num_envs,
      device=self.device,
    )
    self.world_model_learner = WorldModelLearner(
      self._dynamics, self._representation, self.cfg.wm, device=self.device
    )

    self._pending_norm_updates: list[tuple] = []
    self._sim_history: torch.Tensor | None = None

  def _seed_sim_history(self) -> None:
    """Seed the real-sim obs-history ring with the noised current sensor."""
    if self.cfg.history_length == 1:
      return
    seed = self._representation.add_proprio_noise(
      self.env.get_observations()["actor"].to(self.device)
    )
    self._sim_history = seed.unsqueeze(1).repeat(1, self.cfg.history_length, 1)

  def _collection_history(self, sensor: torch.Tensor) -> torch.Tensor:
    """The ``(B, history_length, sensor_dim)`` window for a real-sim actor step.
    history_length > 1 needs ``_seed_sim_history()`` first."""
    if self.cfg.history_length == 1:
      return sensor.unsqueeze(1)
    assert self._sim_history is not None
    return self._sim_history

  def _advance_sim_history(self, next_actor: torch.Tensor, dones: torch.Tensor) -> None:
    """Append the noised next sensor; rows whose episode just reset repeat that
    single realization, matching the warm-start padding."""
    if self.cfg.history_length == 1:
      return
    assert self._sim_history is not None
    next_sensor = self._representation.add_proprio_noise(next_actor)
    shifted = torch.cat([self._sim_history[:, 1:], next_sensor.unsqueeze(1)], dim=1)
    reset = dones.bool().view(-1, 1, 1)
    self._sim_history = torch.where(reset, next_sensor.unsqueeze(1), shifted)

  @property
  def _stores_actor_obs(self) -> bool:
    """The buffer carries the real sensor only for its sole reader, the
    history warm-start (``sample_start``)."""
    return (
      self._representation.actor_obs_mode == "proprio" and self.cfg.history_length > 1
    )

  def _buffer_schema(self) -> dict[str, FieldSpec]:
    """Replay buffer fields."""
    schema = {
      "state": FieldSpec(self._representation.full_state_dim),
      "action": FieldSpec(self._num_actions),
      "prev_action": FieldSpec(self._num_actions),
      "motion_time": FieldSpec(dtype=torch.long),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    }
    if self._stores_actor_obs:
      schema["actor_obs"] = FieldSpec(
        self._representation.proprio_obs_dim(self._num_actions)
      )
    schema["sim_ctx"] = FieldSpec(self._sim_context.output_dim)
    if self._num_goal_parts:
      schema["mode"] = FieldSpec(self._num_goal_parts, dtype=torch.bool)
    return schema

  def _step_and_store(
    self,
    actions: torch.Tensor,
    curr_states: torch.Tensor,
    actor_td: TensorDict,
    extra_fields: dict[str, torch.Tensor],
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Step the env, write the transition to the replay buffer and queue
    ``actor_td`` for the actor-normalizer update."""
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    motion_times = motion_cmd.time_steps.clone()

    raw_actions = actions.detach().clone()
    prev_action = self.action_manager.action.clone()

    next_obs, rewards, dones, extras = self.env.step(actions)

    truncated = extras.get(
      "time_outs",
      torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
    )
    terminated = dones.bool() & ~truncated

    fields: dict[str, torch.Tensor] = {
      "state": curr_states,
      "action": raw_actions,
      "prev_action": prev_action,
      "motion_time": motion_times,
      "terminated": terminated,
      "truncated": truncated,
      **extra_fields,
    }
    self.replay_buffer.insert(TensorDict(fields, batch_size=[self.env.num_envs]))

    self._pending_norm_updates.append(
      (curr_states.clone(), raw_actions.clone(), actor_td.detach().clone())
    )
    return next_obs, rewards, dones, extras

  def _flush_normalization(self) -> None:
    """Apply queued normalizer updates. Call outside ``inference_mode()``."""
    if not self._pending_norm_updates:
      return
    states, actions, actor_inputs = zip(*self._pending_norm_updates, strict=True)
    self.world_model_learner.update_normalization(torch.cat(states), torch.cat(actions))
    self._update_actor_normalization(actor_inputs)
    self._pending_norm_updates.clear()

  def _create_log_dir(self) -> Path:
    """Create and return a timestamped log directory."""
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if self.cfg.run_name:  # runs started together get their own directories
      name += f"_{self.cfg.run_name}"
    log_dir = Path("logs", self.task_id, name)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")
    return log_dir

  def _init_wandb(self, log_dir: Path) -> None:
    """Initialize the W&B run and link the registry artifacts."""
    wandb_config = {
      "env_cfg": asdict(self.env.cfg),
      "runner_cfg": asdict(self.cfg),
    }

    default_name = f"{self.task_id}-{log_dir.name}"
    wandb.init(
      project=self.cfg.wandb_project,
      name=self.cfg.run_name or default_name,
      config=wandb_config,
    )

    if self.registry_name:
      api = wandb.Api()
      for name in self.registry_name.split(","):
        name = name.strip()
        if ":" not in name:
          name = name + ":latest"
        print(f"[INFO] Linking registry artifact: {name}")
        artifact = api.artifact(name)
        wandb.run.use_artifact(artifact)

  def _cleanup(self) -> None:
    """Close environment and finish W&B run."""
    env_close_fn = getattr(self.env, "close", None)
    if env_close_fn:
      env_close_fn()
    wandb.finish()

  def _init_policy(self) -> None:
    actor = build_actor(self.cfg, self._wm, self.device)
    self._active_task_specs = [
      ts for ts in actor.task_specs if self.cfg.slot_mix.get(ts.name, 0.0) > 0.0
    ]

    print("[INFO] Sim slot ranges:")
    for ts, (lo, hi) in zip(
      actor.task_specs, actor.slot_ranges(self.env.num_envs), strict=True
    ):
      print(f"  {ts.name}: [{lo}, {hi}) — goal_dim={ts.goal_dim}")
    critic_input_dim = (
      self._representation.rep_state_dim
      + self._num_goals
      + self._sim_context.output_dim
      + self._representation.critic_extra_dim
      + self._num_goal_parts
    )
    critic = build_mlp(
      critic_input_dim,
      self.cfg.critic_hidden_dims,
      output_dim=1,
    ).to(self.device)

    policy_cfg = self.cfg.policy

    self.wm_env = WorldModelEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      task_specs=self._active_task_specs,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
    )
    if torch.cuda.is_available():
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)

    self._configure_reset_sampling()

    if isinstance(policy_cfg, ActorCriticPolicyLearnerCfg):
      self.policy_learner = ActorCriticPolicyLearner(
        actor, critic, self.wm_env, policy_cfg, device=self.device
      )
    else:
      self.policy_learner = BpttPolicyLearner(
        actor, self.wm_env, policy_cfg, device=self.device
      )

  def _build_sim_obs_td(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    sim_actor_obs: torch.Tensor,
    motion_times: torch.Tensor,
    sim_ctx: torch.Tensor,
    mode: torch.Tensor | None = None,
  ) -> TensorDict:
    proprio = self.cfg.actor_obs == "proprio"
    sensor = self._representation.compute_actor_sensor(
      rep_state,
      sim_ctx,
      aux_state=aux_state,
      sim_actor_obs=sim_actor_obs if proprio else None,
      noisy=proprio,
    )
    goals = {
      ts.name: ts.extract_training_goal(self.motion, motion_times, aux_state, rep_state)
      for ts in self._active_task_specs
    }
    modes = None if mode is None else {"motion": mode}
    return build_actor_obs_td(goals, self._collection_history(sensor), None, modes)

  def _configure_reset_sampling(self) -> None:
    self._sampler: ReturnSampler | None = None
    env_mode = self.motion_cmd.cfg.sampling_mode
    expected = (
      "external" if self.cfg.sampling_mode == "return" else self.cfg.sampling_mode
    )
    if env_mode != expected:
      raise RuntimeError(
        f"runner sampling_mode='{self.cfg.sampling_mode}' needs the motion command "
        f"in '{expected}' mode but the env was built with '{env_mode}'; resolve it "
        "into the env cfg before env creation ('external' also needs an mjlab with "
        "MotionCommand external-mode support)"
      )
    if self.cfg.sampling_mode != "return":
      return
    self._sampler = ReturnSampler(
      cfg=self.cfg.return_sampler,
      bin_count=self.motion_cmd.bin_count,
      frame_count=self.motion.time_step_total,
      num_envs=self.env.num_envs,
      max_step_reward=self.wm_env.max_step_reward,
      device=self.device,
    )

  def step(self) -> int:
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states = obs["full_state"].to(self.device)
      sim_actor_obs = obs["actor"].to(self.device)

      # Pre-step frames and modes; env.step() advances time_steps in place and
      # draws the reset envs' modes.
      motion_times = self.motion_cmd.time_steps.to(self.device).clone()
      mode = self.motion_cmd.mode.clone() if self._num_goal_parts else None

      rep_state = self._representation.represent(curr_states)
      aux_state = self._representation.extract_aux_state(curr_states)
      sim_ctx = self._sim_context.read(self.env.unwrapped)
      actor_td = self._build_sim_obs_td(
        rep_state, aux_state, sim_actor_obs, motion_times, sim_ctx, mode
      )

      actions = self.policy_learner.actor(actor_td)
      if self.policy_learner.cfg.collect_noise > 0:
        actions = (
          actions + torch.randn_like(actions) * self.policy_learner.cfg.collect_noise
        )

      extra_fields = {"sim_ctx": sim_ctx}
      if self._stores_actor_obs:
        extra_fields["actor_obs"] = sim_actor_obs
      if mode is not None:
        extra_fields["mode"] = mode.bool()

      next_obs, rewards, dones, extras = self._step_and_store(
        actions,
        curr_states,
        actor_td=actor_td,
        extra_fields=extra_fields,
      )
      self._advance_sim_history(next_obs["actor"].to(self.device), dones)
      terminated = self.env.unwrapped.termination_manager.terminated
      # Only rank 0 logs, and logging is what clears the buffered extras.
      if is_main():
        self.logger.process_env_step(rewards, terminated, extras)

      if self._sampler is not None:
        # A failure on a step that also times out counts as the time-out.
        manager = self.env.unwrapped.termination_manager
        truncated = manager.time_outs.to(self.device, torch.bool)
        clip_end = manager.get_term("motion_end").to(self.device, torch.bool)
        weights = self._sampler.record(
          motion_times,
          rewards,
          terminated.to(self.device, torch.bool) & ~truncated,
          clip_end,
          truncated & ~clip_end,
        )
        if weights is not None:
          self.motion_cmd.set_sampling_distribution(weights)

    self._flush_normalization()
    return self.env.num_envs

  def _critic_module(self) -> nn.Module | None:
    return getattr(self.policy_learner, "critic", None)

  def _keep_checkpoint(self, log_dir: Path, step: int, files: list[str]) -> None:
    """Copy the checkpoint just saved to ``checkpoints/<step>/`` with the cfg."""
    keep = log_dir / "checkpoints" / f"{step:08d}"
    keep.mkdir(parents=True, exist_ok=True)
    for f in files:
      shutil.copy2(f, keep / Path(f).name)
    meta = {"step": step, "task_id": self.task_id, "runner_cfg": asdict(self.cfg)}
    # The plant the actor trained on (the G1-D chassis model), for evaluation.
    actions = self.env.unwrapped.action_manager
    plant = {}
    for name in actions.active_terms:
      plant.update(getattr(actions.get_term(name), "plant", {}))
    if plant:
      meta["plant"] = plant
    (keep / "run.json").write_text(json.dumps(meta, indent=1, default=str))

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    self.policy_learner.actor.train()
    for actor_td in actor_inputs:
      self.policy_learner.actor.update_normalization(actor_td)

  def run(self) -> None:
    # Data-parallel ranks start from rank 0's weights; only rank 0 logs and saves.
    main = is_main()
    for module in (self._dynamics, self.policy_learner.actor, self._critic_module()):
      if module is not None:
        broadcast_module(module)
    log_dir = self._create_log_dir() if main else Path()
    if main:
      self._init_wandb(log_dir)

    self.env.reset()
    self._seed_sim_history()

    print(f"[INFO] Starting training loop for {self.cfg.max_iterations} iterations...")

    should_train = Ratio(
      self.env.num_envs
      / self.cfg.wm.num_batches
      / self.cfg.wm.batch_size
      / self.cfg.wm.horizon
      * self.cfg.replay_ratio
    )

    num_collected_steps = 0
    start_time = time.time()
    wm_learn_time = 0.0
    policy_learn_time = 0.0
    loss_dict: dict[str, torch.Tensor] = {}
    log_count = 0
    uploader = ThreadPoolExecutor(max_workers=1)
    upload: Future | None = None

    for step in range(self.cfg.max_iterations):
      num_collected_steps += self.step()

      for _ in range(should_train(step)):
        if self.replay_buffer.num_transitions < self.cfg.wm.horizon:
          continue

        data = (
          self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
          for _ in range(self.cfg.wm.num_batches)
        )
        starts = (
          sample_start(
            self.replay_buffer, self.cfg.history_length, self.cfg.policy.batch_size
          )
          for _ in range(self.cfg.policy.num_batches)
        )

        start_wm_time = time.time()
        wm_logs = self.world_model_learner.update(data)
        wm_learn_time += time.time() - start_wm_time

        start_policy_time = time.time()
        policy_logs = self.policy_learner.update(starts)
        policy_learn_time += time.time() - start_policy_time

        loss_dict.update(wm_logs)
        loss_dict.update(policy_logs)

      if main and step % self.cfg.log_interval == 0:
        elapsed_time = time.time() - start_time
        total_train_time = wm_learn_time + policy_learn_time
        collect_time = elapsed_time - total_train_time

        self.logger.log(
          it=log_count,
          total_it=self.cfg.max_iterations // self.cfg.log_interval,
          collect_time=collect_time,
          wm_learn_time=wm_learn_time,
          policy_learn_time=policy_learn_time,
          loss_dict=loss_dict,
          num_collected_steps=num_collected_steps,
        )

        log_count += 1
        start_time = time.time()
        wm_learn_time = 0
        policy_learn_time = 0
        num_collected_steps = 0

      if main and step % self.cfg.save_interval == 0:
        assert wandb.run is not None  # _init_wandb ran before the loop
        # The previous version must be logged before its files are overwritten.
        if upload is not None:
          upload.result()
        world_model_path = str(log_dir / "world_model.pt")
        actor_path = str(log_dir / "actor.pt")
        torch.save(self._wm.bundle(), world_model_path)
        torch.save(self.policy_learner.actor.state_dict(), actor_path)
        checkpoint_files = [world_model_path, actor_path]
        critic = getattr(self.policy_learner, "critic", None)
        if critic is not None:
          critic_path = str(log_dir / "critic.pt")
          torch.save(critic.state_dict(), critic_path)
          checkpoint_files.append(critic_path)
        if self.cfg.keep_checkpoints:
          self._keep_checkpoint(log_dir, step, checkpoint_files)
        # Skip ONNX export on huge datasets and on the G1-D (no deploy contract).
        if (
          self._representation.actor_obs_mode == "proprio"
          and self.motion.time_step_total <= 15_000
          and self.cfg.representation != "g1d"
        ):
          checkpoint_files += export_actor_as_onnx(
            actor=self.policy_learner.actor,
            env=self.env.unwrapped,
            path=str(log_dir),
            run_name=wandb.run.name,
            term_widths=self._representation.proprio_term_widths(self._num_actions),
          )
        upload = uploader.submit(log_model_artifact, checkpoint_files)

    uploader.shutdown()
    self._cleanup()
