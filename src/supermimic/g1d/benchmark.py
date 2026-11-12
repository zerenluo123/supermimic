"""Tracking benchmark of the G1-D, the protocol of SuperMimic-Reproduced (g1d).

Every clip is played once from its first frame with the deterministic policy: no
action, proprio, goal or reset noise and no domain randomization. A clip *fails*
when a failure rule fires (the training termination, or another ruler's
thresholds) and stops there. Normalized return = summed step reward / (0.1 *
steps played), also over the whole clip (``return_full_clip``). Also reported:
which rule ended each failed clip and the tracking precision over the steps
played (column height error, chassis world xy distance, mean upper-body distance
to the re-anchored reference, mean upper-joint error).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from supermimic.g1d import constants as C
from supermimic.g1d.env_cfg import FAILURE_RULES, G1DMotionCommand, g1d_env_cfg
from supermimic.models import MultiEncoderAE
from supermimic.representations import create_representation
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import WorldModel, build_actor
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.helpers import unpack_obs_dims

MAX_STEP_REWARD = 0.1  # dt * the positive tracking weights


@dataclass(frozen=True)
class Ruler:
  """Failure thresholds (m) of ``FAILURE_RULES``; 0 leaves a rule out."""

  chassis_xy: float = C.CHASSIS_THRESHOLD
  upper_body: float = C.UPPER_THRESHOLD
  column: float = C.COLUMN_THRESHOLD

  def thresholds(self) -> tuple[float, float, float]:
    return self.chassis_xy, self.upper_body, self.column


# The three rulers of the Reproduced comparison: the training rule, the common
# ruler (the earlier 0.5 m chassis rule) with the column rule, and without it.
RULERS = {
  "train": Ruler(),
  "public": Ruler(chassis_xy=0.5),
  "public_nocol": Ruler(chassis_xy=0.5, column=0.0),
}


@dataclass
class ClipResult:
  name: str
  frames: int
  steps: int
  norm_return: float
  return_full_clip: float
  failed: bool
  fail_rules: list[str] = field(default_factory=list)
  col_err_mean: float = 0.0
  xy_err_mean: float = 0.0
  upper_err_mean: float = 0.0
  joint_err_mean: float = 0.0


def make_benchmark_env(
  motion_file: str, ruler: Ruler, num_envs: int, device: str
) -> ManagerBasedRlEnv:
  """The play cfg without randomization, the ruler's failure rules."""
  cfg = g1d_env_cfg(play=True)
  cfg.events = {}
  cfg.commands["motion"].motion_file = motion_file
  for rule, threshold in zip(FAILURE_RULES, ruler.thresholds(), strict=True):
    if threshold > 0:
      cfg.terminations[rule].params["threshold"] = threshold
    else:
      cfg.terminations.pop(rule)
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def load_actor(
  checkpoint_dir: Path, env: ManagerBasedRlEnv, device: str
) -> tuple[MultiEncoderAE, object, int]:
  """The checkpoint's actor, the representation that builds its goal (noise off),
  and its history length."""
  run = json.loads((checkpoint_dir / "run.json").read_text())
  cfg = load_runner_cfg({"runner_cfg": run["runner_cfg"]})
  obs, _ = env.reset()
  full_dim, num_goals = unpack_obs_dims(obs)
  rep = create_representation(
    "g1d",
    full_dim,
    env.action_manager.total_action_dim,
    env.step_dt,
    env.cfg.decimation,
    env.scene["robot"].data.default_joint_pos[0],
    actor_obs_mode=cfg.actor_obs,
  )
  rep.actor_goal_noise = False
  wm = WorldModel(rep, None, None, num_goals, env.action_manager.total_action_dim)  # type: ignore[arg-type]
  actor = build_actor(cfg, wm, device)
  actor.load_state_dict(torch.load(checkpoint_dir / "actor.pt", map_location=device))
  actor.eval()
  return actor, rep, cfg.history_length


@torch.no_grad()
def run_benchmark(
  actor: MultiEncoderAE, rep, history_length: int, env: ManagerBasedRlEnv
) -> tuple[list[ClipResult], dict]:
  device = env.device
  cmd: G1DMotionCommand = env.command_manager.get_term("motion")  # type: ignore[assignment]
  motion = cmd.motion
  ends = motion.motion_ends
  starts = torch.cat([ends.new_zeros(1), ends[:-1]])
  n_clips, num_envs = ends.numel(), env.num_envs
  current = torch.arange(num_envs, device=device)
  nxt = current + num_envs
  cmd.forced_frames = starts[current.clamp(max=n_clips - 1)]
  obs, _ = env.reset()
  cmd.forced_frames = starts[nxt.clamp(max=n_clips - 1)]
  active = current < n_clips

  def zeros():
    return torch.zeros(num_envs, device=device)

  ret, col, xy, up, joint = zeros(), zeros(), zeros(), zeros(), zeros()
  steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  pooled_col: list[torch.Tensor] = []
  results: dict[int, ClipResult] = {}
  thresholds = {
    rule: env.cfg.terminations[rule].params["threshold"]
    for rule in FAILURE_RULES
    if rule in env.cfg.terminations
  }
  hist = obs["actor"].unsqueeze(1).repeat(1, history_length, 1)
  while len(results) < n_clips:
    aux = obs["full_state"][:, :7]
    _, goal = rep.compute_wm_goal(motion, cmd.time_steps, aux)
    action = actor(build_actor_obs_td({"motion": goal}, hist))
    obs, reward, terminated, truncated, _ = env.step(action)
    terms = env.__dict__["_g1d_step_terms"][1]
    steps += 1
    ret += reward
    col += terms["column_err"]
    xy += terms["failure"][:, 0]
    up += terms["upper_err"]
    joint += terms["joint_err"]
    pooled_col.append(terms["column_err"][active])
    done = (terminated | truncated) & active
    for e in done.nonzero().squeeze(-1).tolist():
      ci, st = int(current[e]), max(int(steps[e]), 1)
      failed = bool(terminated[e])
      length = int(ends[ci] - starts[ci])
      rules = [
        rule
        for i, rule in enumerate(FAILURE_RULES)
        if failed
        and rule in thresholds
        and float(terms["failure"][e, i]) > thresholds[rule]
      ]
      results[ci] = ClipResult(
        name=motion.motion_names[ci],
        frames=length,
        steps=st,
        norm_return=float(ret[e]) / (MAX_STEP_REWARD * st),
        return_full_clip=float(ret[e]) / (MAX_STEP_REWARD * max(length - 1, 1)),
        failed=failed,
        fail_rules=rules,
        col_err_mean=float(col[e]) / st,
        xy_err_mean=float(xy[e]) / st,
        upper_err_mean=float(up[e]) / st,
        joint_err_mean=float(joint[e]) / st,
      )
    reset = terminated | truncated
    if reset.any():
      ids = reset.nonzero().squeeze(-1)
      current[ids] = nxt[ids]
      nxt[ids] += num_envs
      active = current < n_clips
      cmd.forced_frames[ids] = starts[nxt[ids].clamp(max=n_clips - 1)]
      for t in (ret, col, xy, up, joint):
        t[ids] = 0.0
      steps[ids] = 0
    hist = torch.cat([hist[:, 1:], obs["actor"].unsqueeze(1)], dim=1)
    hist[reset] = obs["actor"][reset].unsqueeze(1)
  clips = [results[i] for i in range(n_clips)]
  cols = torch.cat(pooled_col)
  pooled = {
    "col_err_mean": float(cols.mean()),
    "col_err_p99": float(torch.quantile(cols[: 2**24].float(), 0.99)),
  }
  return clips, pooled


def summarize(clips: list[ClipResult], pooled: dict) -> dict:
  n = len(clips)
  total = sum(c.steps for c in clips)

  def weighted(key: str) -> float:
    return sum(getattr(c, key) * c.steps for c in clips) / max(total, 1)

  return {
    "clips": n,
    "failed": sum(c.failed for c in clips),
    "mean_norm_return": sum(c.norm_return for c in clips) / max(n, 1),
    "step_weighted_norm_return": weighted("norm_return"),
    "return_full_clip": sum(c.return_full_clip for c in clips) / max(n, 1),
    "fail_rule_counts": {
      rule: sum(rule in c.fail_rules for c in clips) for rule in FAILURE_RULES
    },
    "xy_err_mean": weighted("xy_err_mean"),
    "upper_err_mean": weighted("upper_err_mean"),
    "joint_err_mean": weighted("joint_err_mean"),
    "failed_clips": [c.name for c in clips if c.failed],
    **pooled,
  }
