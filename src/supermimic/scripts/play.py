"""Play a SuperMimic multi-encoder actor against the sim."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.g1d.robot import get_g1d_robot_cfg
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.viewers import (
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)
from supermimic.utils.wandb_artifacts import (
  download_artifact_cached,
  download_model_dir,
)


@dataclass
class PlayConfig:
  wandb_path: str | None = None
  checkpoint_dir: str | None = None
  """A kept checkpoint (``<run>/checkpoints/<step>``, with its ``run.json``) to play
  instead of the W&B run's latest; needs ``motion_file``."""
  motion_file: str | None = None
  initial_task: Literal["motion", "velocity"] = "motion"
  """Task selected when the viewer opens. Switch live via the GUI dropdown."""
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"
  eval_tag: str | None = None
  """G1-D: show the clip results of ``<checkpoint_dir>/eval_<tag>_*.json``."""
  visual_mesh_dir: str | None = None
  """G1-D: directory with the robot's visual STL meshes (drawn instead of the
  colliders)."""
  replay_dir: str | None = None
  """G1-D: play a disk recording on the CPU, without loading the policy or motions."""
  port: int = 8080
  """Port for offline replay."""


def play(task_id: str, cfg: PlayConfig) -> None:
  if cfg.replay_dir is not None:
    if task_id != "SuperMimic-GenCoord-G1D":
      raise ValueError("--replay-dir is only supported for G1-D")
    from supermimic.g1d.replay_viewer import G1DReplayViewer

    G1DReplayViewer(Path(cfg.replay_dir), cfg.visual_mesh_dir, cfg.port).run()
    return
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  print(f"[INFO] Loading environment: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = True

  if cfg.checkpoint_dir is not None:
    if cfg.motion_file is None:
      raise ValueError("--checkpoint-dir needs --motion-file")
    model_dir = Path(cfg.checkpoint_dir)
    run_info = json.loads((model_dir / "run.json").read_text())
    runner_cfg = load_runner_cfg({"runner_cfg": run_info["runner_cfg"]})
    motion_artifacts = []
  else:
    import wandb

    if cfg.wandb_path is None:
      raise ValueError("give --wandb-path or --checkpoint-dir")
    print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
    run = wandb.Api().run(cfg.wandb_path)
    runner_cfg = load_runner_cfg(run.config)
    motion_artifacts = [a for a in run.used_artifacts() if a.type == "motions"]
    model_dir = None
  check_representation_matches(task_id, runner_cfg.representation)
  g1d = runner_cfg.representation == "g1d"
  if g1d:
    # The benchmark's nominal plant, the reference as a ghost, visual meshes.
    for name, key in (("base_com", "ranges"), ("encoder_bias", "bias_range")):
      params = env_cfg.events[name].params
      params[key] = (
        {axis: (0.0, 0.0) for axis in params[key]}
        if isinstance(params[key], dict)
        else (0.0, 0.0)
      )
    env_cfg.commands["motion"].debug_vis = True
    # The ghosts sit almost on the grey robot: saturated colors set them apart.
    env_cfg.commands["motion"].viz.ghost_color = (0.1, 0.85, 0.2, 0.6)
    if cfg.visual_mesh_dir is not None:
      env_cfg.scene.entities["robot"] = get_g1d_robot_cfg(cfg.visual_mesh_dir)

  command_term = env_cfg.commands["motion"]
  assert isinstance(command_term, MotionCommandCfg)
  motion_names: list[str] = []
  if cfg.motion_file is not None:
    expanded = os.path.expanduser(cfg.motion_file)
    if any(c in expanded for c in "*?[]"):
      files = sorted(glob.glob(expanded))
    else:
      files = [expanded]
    if not files:
      raise ValueError(f"No motion files matched: {cfg.motion_file}")
    command_term.motion_file = files
    motion_names = [Path(f).stem for f in files]
    print(f"[INFO] Using {len(files)} local motion(s)")
  elif motion_artifacts:
    motion_files = []
    for artifact in motion_artifacts:
      print(f"[INFO] Found linked motion artifact: {artifact.name}")
      motion_files.append(str(download_artifact_cached(artifact) / "motion.npz"))
    command_term.motion_file = (
      motion_files if len(motion_files) > 1 else motion_files[0]
    )
    motion_names = [a.name.split(":")[0] for a in motion_artifacts]
    print(f"[INFO] Loaded {len(motion_files)} motion(s)")

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
  if cfg.motion_file is not None:
    # The loader names clips from their files or a packed directory's list.
    motion = env.unwrapped.command_manager.get_term("motion").motion
    motion_names = list(motion.motion_names)

  wm = build_world_model(runner_cfg, env, device)
  representation = wm.representation
  dynamics = wm.dynamics
  if g1d:
    representation.actor_goal_noise = False

  if model_dir is None:
    model_dir = download_model_dir(run)
  wm.load_bundle(torch.load(model_dir / "world_model.pt", map_location=device))
  dynamics.eval()

  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(torch.load(model_dir / "actor.pt", map_location=device))
  actor.eval()

  wm_vis_model = create_wm_vis_model(env.unwrapped.sim.mj_model)
  if g1d:
    wm_vis_model.geom_rgba[:, 0:3] = [0.2, 0.45, 1.0]

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = cfg.viewer
  print(f"[INFO] Starting viewer: {resolved}")

  if resolved == "native":
    SuperMimicNativeViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
    ).run()
  else:
    viewer_cls, extra = SuperMimicViserViewer, {}
    if g1d:
      from supermimic.g1d.viewer import G1DViserViewer, load_clip_table

      viewer_cls = G1DViserViewer
      if cfg.eval_tag is not None and cfg.checkpoint_dir is not None:
        extra["clip_table"] = load_clip_table(Path(cfg.checkpoint_dir), cfg.eval_tag)
    viewer_cls(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      motion_names=motion_names,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
      **extra,
    ).run()

  env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(PlayConfig, args=remaining_args, config=mjlab.TYRO_FLAGS)
  play(chosen_task, args)


if __name__ == "__main__":
  main()
