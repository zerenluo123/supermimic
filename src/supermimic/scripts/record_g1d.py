"""Record deterministic G1-D sim/reference/closed-loop WM poses, without rendering.

Run with --checkpoint-dir, --motion-file, --output-dir and optionally --eval-tag.
Completed clips are resumable. A different checkpoint/data/code needs a new cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import tyro

from supermimic.g1d import constants as C
from supermimic.g1d.replay import FORMAT_VERSION, load_clip, save_clip, write_json


def recording_never_terminate(env, **_kwargs):
  return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def fingerprint(checkpoint: Path, files: list[str]) -> str:
  digest = hashlib.sha256()
  for path in [checkpoint / n for n in ("actor.pt", "world_model.pt", "run.json")]:
    with path.open("rb") as f:
      for block in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(block)
  for filename in files:
    path = Path(filename).resolve()
    children = sorted(path.glob("*")) if path.is_dir() else [path]
    for child in children:
      if child.is_file():
        stat = child.stat()
        digest.update(f"{child}:{stat.st_size}:{stat.st_mtime_ns}".encode())
  source = Path(__file__).resolve().parents[1]
  for path in sorted(source.rglob("*.py")):
    if path.name not in ("replay_viewer.py", "viewer.py", "play_g1d_replay.py"):
      digest.update(path.relative_to(source).as_posix().encode())
      digest.update(path.read_bytes())
  return digest.hexdigest()


def pose_from_state(state: np.ndarray, names: list[str]) -> np.ndarray:
  """Batched inverse of the G1-D state layout, without FK or per-joint transfers."""
  qpos = np.zeros((*state.shape[:-1], len(names)), dtype=np.float32)
  index = {name: i for i, name in enumerate(names)}
  qpos[..., index[C.BASE_JOINTS[0]]] = state[..., 0]
  qpos[..., index[C.BASE_JOINTS[1]]] = state[..., 1]
  w, x, y, z = np.moveaxis(state[..., 3:7], -1, 0)
  qpos[..., index[C.BASE_JOINTS[2]]] = np.arctan2(
    2 * (w * z + x * y), 1 - 2 * (y * y + z * z)
  )
  for name in C.LIFT_JOINTS:
    qpos[..., index[name]] = state[..., 7] / 2
  for i, name in enumerate(C.UPPER_JOINTS):
    qpos[..., index[name]] = state[..., 8 + i]
  return qpos


@torch.no_grad()
def main(
  checkpoint_dir: Path,
  motion_file: str,
  output_dir: Path,
  eval_tag: str | None = None,
  device: str = "cuda:0",
  num_envs: int = 128,
  max_clips: int | None = None,
) -> None:
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.tracking.mdp.commands import _resolve_motion_file

  import supermimic.tasks  # noqa: F401
  from supermimic.g1d.env_cfg import _step_terms, g1d_env_cfg
  from supermimic.g1d.robot import load_model
  from supermimic.g1d.viewer import load_clip_table
  from supermimic.runners.run_config import load_runner_cfg
  from supermimic.runners.supermimic_runner import build_actor, build_world_model
  from supermimic.scripts.train_utils import setup_device
  from supermimic.tasks.task_spec import build_actor_obs_td

  if num_envs < 1 or (max_clips is not None and max_clips < 1):
    raise ValueError("num_envs and max_clips must be positive")
  setup_device(device)
  files = _resolve_motion_file(motion_file)
  key = fingerprint(checkpoint_dir, files)
  output_dir.mkdir(parents=True, exist_ok=True)
  index_path = output_dir / "index.json"
  index = (
    json.loads(index_path.read_text())
    if index_path.exists()
    else {
      "version": FORMAT_VERSION,
      "fingerprint": key,
      "clips": {},
      "checkpoint": str(checkpoint_dir.resolve()),
      "motion_file": motion_file,
      "complete": False,
    }
  )
  if index["version"] != FORMAT_VERSION or index["fingerprint"] != key:
    raise ValueError(
      "Cache belongs to different weights/data/code. Use a new --output-dir."
    )
  # One recorder per cache, including after a crash (the OS releases the lock).
  import fcntl

  with (output_dir / ".record.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = g1d_env_cfg(play=True)
    cfg.commands["motion"].motion_file = motion_file
    # Keep the threshold schema used to bind the representation, but suppress
    # resets while recording (including motion-end and time limits).
    for term in cfg.terminations.values():
      term.func = recording_never_terminate
    cfg.observations["actor"].enable_corruption = False
    # Keep SimContext's schema, with the benchmark's nominal physics.
    cfg.events["base_com"].params["ranges"] = {i: (0.0, 0.0) for i in range(3)}
    cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
    model = load_model()
    friction = float(model.geom_friction[0, 0])
    cfg.events["foot_friction"].params["ranges"] = (friction, friction)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
      wrapped = RslRlVecEnvWrapper(env)
      runner_cfg = load_runner_cfg(
        {
          "runner_cfg": json.loads((checkpoint_dir / "run.json").read_text())[
            "runner_cfg"
          ]
        }
      )
      if runner_cfg.representation != "g1d":
        raise ValueError("This recorder requires a G1-D checkpoint")
      wm = build_world_model(runner_cfg, wrapped, device)
      wm.load_bundle(torch.load(checkpoint_dir / "world_model.pt", map_location=device))
      wm.dynamics.eval()
      rep = wm.representation
      rep.actor_goal_noise = False
      actor = build_actor(runner_cfg, wm, device)
      actor.load_state_dict(
        torch.load(checkpoint_dir / "actor.pt", map_location=device)
      )
      actor.eval()
      actor.set_slot_mix({"motion": 1.0})
      cmd = env.command_manager.get_term("motion")
      motion = cmd.motion
      ends = motion.motion_ends.cpu().numpy()
      starts = np.r_[0, ends[:-1]]
      lengths = ends - starts
      names = [
        env.sim.mj_model.joint(i).name.removeprefix("robot/")
        for i in range(env.sim.mj_model.njnt)
      ]
      if len(names) != env.sim.mj_model.nq:
        raise ValueError("Expected scalar G1-D joints")
      index.update(
        dt=env.step_dt,
        joint_names=names,
        total_clips=len(lengths),
        total_frames=int(lengths.sum()),
        benchmark=load_clip_table(checkpoint_dir, eval_tag) if eval_tag else {},
      )
      pending = []
      for ci in range(len(lengths)):
        entry = index["clips"].get(str(ci))
        if entry:
          try:
            load_clip(output_dir, entry, len(names))
            continue
          except (OSError, ValueError, KeyError):
            pass
        pending.append(ci)
      if max_clips is not None:
        pending = pending[:max_clips]
      write_json(index_path, index)
      if not pending:
        print("[record] All requested clips already cached", flush=True)
        return

      # Manual resets after recording the last frame. Freeze each reference at
      # its own last frame so the library's global wrap cannot overwrite qpos.
      limits = torch.zeros(num_envs, dtype=torch.long, device=device)

      def advance_command(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.set_time_steps(ids, torch.minimum(self.time_steps[ids] + 1, limits[ids]))
        self.update_relative_body_poses()

      cmd._update_command = MethodType(advance_command, cmd)
      cmd.forced_frames = torch.zeros(num_envs, dtype=torch.long, device=device)
      active = np.full(num_envs, -1, dtype=int)
      buffers = {}
      steps = np.zeros(num_envs, dtype=int)
      reset_history = torch.ones(num_envs, dtype=torch.bool, device=device)
      hist = wm_hist = aux = state = last_action = None
      queue = iter(pending)
      t0 = time.monotonic()

      def reset_slots(slots):
        nonlocal aux, state, last_action
        for slot in slots:
          ci = next(queue, -1)
          active[slot], steps[slot] = ci, 0
          source_ci = max(ci, 0)
          cmd.forced_frames[slot] = int(starts[source_ci])
          limits[slot] = int(ends[source_ci] - 1)
          if ci >= 0:
            n = int(lengths[ci])
            buffers[slot] = {
              "poses": np.zeros((n, 3, len(names)), np.float32),
              "errors": np.zeros((n, 3), np.float32),
              "metrics": np.zeros((n, 4), np.float32),
              "wm_valid": np.ones(n, bool),
            }
        ids = torch.as_tensor(slots, device=device)
        env.reset(env_ids=ids)
        # reset() advances the command once; undo that for a true frame-0 start.
        cmd.set_time_steps(ids, cmd.forced_frames[ids])
        cmd.update_relative_body_poses()
        obs = env.observation_manager.compute(update_history=True)
        full = obs["full_state"]
        if aux is None:
          aux, state = rep.extract_aux_state(full), rep.represent(full)
          last_action = torch.zeros(
            num_envs, env.action_manager.total_action_dim, device=device
          )
        else:
          aux[ids], state[ids] = (
            rep.extract_aux_state(full)[ids],
            rep.represent(full)[ids],
          )
          last_action[ids] = 0
        reset_history[ids] = True
        initial = env.sim.data.qpos.float().cpu().numpy()
        ref = reference_pose()
        for slot in slots:
          if active[slot] >= 0:
            buffers[slot]["poses"][0] = np.stack(
              (initial[slot], ref[slot], initial[slot])
            )
        return obs

      def reference_pose():
        t = cmd.time_steps
        ref = torch.cat(
          (motion.root_pos_w[t], motion.root_quat_w[t], motion.joint_pos[t]), -1
        )
        return pose_from_state(ref.float().cpu().numpy(), names)

      obs = reset_slots(list(range(num_envs)))
      sim_ctx = wm.sim_context.read(env)
      while (active >= 0).any():
        sensor = obs["actor"]
        _, goal = rep.compute_wm_goal(
          motion, cmd.time_steps, rep.extract_aux_state(obs["full_state"])
        )
        wm_goal, _ = rep.compute_wm_goal(motion, cmd.time_steps, aux)
        from supermimic.representations import extract_actor_goal

        wm_sensor = rep.compute_actor_sensor(
          state,
          sim_ctx,
          aux_state=aux,
          joint_state=rep.joint_state(state),
          last_action=last_action,
        )

        def history(previous, current):
          result = (
            current[:, None].repeat(1, runner_cfg.history_length, 1)
            if previous is None
            else torch.cat((previous[:, 1:], current[:, None]), 1)
          )
          result[reset_history] = current[reset_history, None]
          return result

        hist, wm_hist = history(hist, sensor), history(wm_hist, wm_sensor)
        reset_history[:] = False
        action = actor(build_actor_obs_td({"motion": goal}, hist))
        last_action = actor(
          build_actor_obs_td({"motion": extract_actor_goal(wm_goal)}, wm_hist)
        )
        aux, state = wm.dynamics(state, last_action, aux, sim_ctx)
        obs, reward, _, _, _ = env.step(action)
        terms = _step_terms(env)
        actual = env.sim.data.qpos.float().cpu().numpy()
        pred = pose_from_state(
          rep.reconstruct_full_state(aux, state).float().cpu().numpy(), names
        )
        reference = reference_pose()
        errors = terms["failure"].float().cpu().numpy()
        metrics = (
          torch.stack(
            (reward, terms["column_err"], terms["upper_err"], terms["joint_err"]), -1
          )
          .float()
          .cpu()
          .numpy()
        )
        finished = []
        for slot in np.flatnonzero(active >= 0):
          ci, buf = active[slot], buffers[slot]
          steps[slot] += 1
          frame = steps[slot]
          finite = np.isfinite(actual[slot]).all() and np.isfinite(errors[slot]).all()
          # Do not send NaNs to the browser. A divergent WM is hidden from that
          # frame onwards; a divergent simulator ends this recording explicitly.
          if finite and frame < len(buf["poses"]):
            valid = bool(np.isfinite(pred[slot]).all())
            buf["poses"][frame] = np.stack(
              (actual[slot], reference[slot], pred[slot] if valid else actual[slot])
            )
            buf["errors"][frame], buf["metrics"][frame] = errors[slot], metrics[slot]
            buf["wm_valid"][frame] = valid
          if frame >= len(buf["poses"]) - 1 or not finite:
            n = min(frame + int(finite), len(buf["poses"]))
            entry = save_clip(
              output_dir / f"clip_{ci:06d}.npz", **{k: v[:n] for k, v in buf.items()}
            )
            entry.update(
              name=motion.motion_names[ci],
              source_frames=int(lengths[ci]),
              truncated=not finite,
            )
            index["clips"][str(ci)] = entry
            index["complete"] = len(index["clips"]) == len(lengths)
            write_json(index_path, index)
            del buffers[slot]
            finished.append(int(slot))
            print(
              f"[record] {len(index['clips'])}/{len(lengths)} {entry['name']} "
              f"{n} frames, {entry['bytes'] / 1024:.0f} KiB, "
              f"failure={entry['first_failure']} ({time.monotonic() - t0:.1f}s)",
              flush=True,
            )
        if finished:
          obs = reset_slots(finished)
      print(
        f"[record] Done: {sum(c['bytes'] for c in index['clips'].values()) / 2**20:.1f} MiB",
        flush=True,
      )
    finally:
      env.close()


if __name__ == "__main__":
  tyro.cli(main)
