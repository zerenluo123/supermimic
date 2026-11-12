"""Benchmark every kept checkpoint of a G1-D run (``--runner.keep-checkpoints``).

Builds one benchmark env per ruler and reuses it; writes
``<run>/eval_<tag>_<ruler>.csv`` (one row per checkpoint) and
``checkpoints/<step>/eval_<tag>_<ruler>.json`` (summary + per-clip results).
Evaluated checkpoints are skipped, so it can be re-run or left watching.

  uv run src/supermimic/scripts/eval_g1d.py --run-dir logs/.../<time> \\
    --motion-file 'data/lafan1_g1d_v3/*.npz' --tag lafan --watch
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import tyro

COLS = (
  "step",
  "eval_at",
  "norm_return",
  "step_weighted",
  "return_full_clip",
  "n_failed",
  "col_err_mean_cm",
  "col_err_p99_cm",
  "xy_err_mean_cm",
  "upper_err_mean_cm",
  "joint_err_mean",
  "fail_rules",
  "failed",
)


def main(
  run_dir: Path,
  motion_file: str,
  tag: str,
  rulers: tuple[str, ...] = ("train", "public", "public_nocol"),
  device: str = "cuda:0",
  num_envs: int = 1024,
  watch: bool = False,
  every: int = 1,
) -> None:
  import mjlab.tasks  # noqa: F401
  import numpy as np
  import torch

  import supermimic.tasks  # noqa: F401
  from supermimic.g1d.benchmark import (
    RULERS,
    load_actor,
    make_benchmark_env,
    run_benchmark,
    summarize,
  )
  from supermimic.scripts.train_utils import setup_device

  setup_device(device)
  from mjlab.tasks.tracking.mdp.commands import _resolve_motion_file

  files = _resolve_motion_file(motion_file)
  if len(files) == 1 and Path(files[0]).is_dir():  # a packed directory
    num_clips = len(np.load(Path(files[0]) / "motion_ends.npy"))
  else:
    num_clips = len(files)
  envs = {
    name: make_benchmark_env(
      motion_file, RULERS[name], min(num_envs, num_clips), device
    )
    for name in rulers
  }
  csvs = {name: run_dir / f"eval_{tag}_{name}.csv" for name in rulers}
  done: dict[str, set[str]] = {}
  for name, csv in csvs.items():
    if not csv.exists():
      csv.write_text(",".join(COLS) + "\n")
    done[name] = {line.split(",")[0] for line in csv.read_text().splitlines()[1:]}
  while True:
    ckpts = sorted(
      p for p in (run_dir / "checkpoints").glob("*") if (p / "actor.pt").exists()
    )
    ckpts = [c for i, c in enumerate(ckpts) if i % every == 0]
    for ckpt in ckpts:
      for name in rulers:
        if ckpt.name in done[name]:
          continue
        t0 = time.time()
        env = envs[name]
        actor, rep, history = load_actor(ckpt, env, device)
        clips, pooled = run_benchmark(actor, rep, history, env)
        s = summarize(clips, pooled)
        row = [
          ckpt.name,
          time.strftime("%Y-%m-%d %H:%M:%S"),
          f"{s['mean_norm_return']:.4f}",
          f"{s['step_weighted_norm_return']:.4f}",
          f"{s['return_full_clip']:.4f}",
          s["failed"],
          f"{100 * s['col_err_mean']:.3f}",
          f"{100 * s['col_err_p99']:.3f}",
          f"{100 * s['xy_err_mean']:.2f}",
          f"{100 * s['upper_err_mean']:.2f}",
          f"{s['joint_err_mean']:.4f}",
          " ".join(f"{k}:{v}" for k, v in s["fail_rule_counts"].items() if v),
          "|".join(s["failed_clips"]),
        ]
        with open(csvs[name], "a") as f:
          f.write(",".join(str(x) for x in row) + "\n")
        out = {"summary": s, "clips": [asdict(c) for c in clips]}
        (ckpt / f"eval_{tag}_{name}.json").write_text(json.dumps(out, indent=1))
        print(
          f"[eval] {ckpt.name} {tag}/{name}: return {s['mean_norm_return']:.4f}, "
          f"failed {s['failed']}/{s['clips']}, column {100 * s['col_err_mean']:.2f} cm "
          f"({time.time() - t0:.0f} s)",
          flush=True,
        )
        done[name].add(ckpt.name)
        del actor
        torch.cuda.empty_cache()
    if not watch:
      break
    time.sleep(60)


if __name__ == "__main__":
  tyro.cli(main)
