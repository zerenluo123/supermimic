"""Small, per-clip disk caches and playback state. No simulator or GPU required."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
FAILURE_RULES = ("chassis_xy", "upper_body", "column")
THRESHOLDS = (0.1, 0.25, 0.1)


def write_json(path: Path, value: dict) -> None:
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(value, indent=2) + "\n")
  os.replace(temporary, path)


def first_failure(errors: np.ndarray, thresholds=THRESHOLDS) -> int | None:
  over = (errors > np.asarray(thresholds)).any(axis=1)
  hits = np.flatnonzero(over)
  return int(hits[0]) if len(hits) else None


def save_clip(
  path: Path,
  poses: np.ndarray,
  errors: np.ndarray,
  metrics: np.ndarray,
  wm_valid: np.ndarray,
) -> dict:
  """Commit a complete clip atomically; poses are [frame, sim/ref/wm, qpos]."""
  if len(poses) == 0 or not np.isfinite(poses).all():
    raise ValueError("A recording must contain finite poses")
  temporary = path.with_suffix(".tmp")
  with temporary.open("wb") as f:
    np.savez_compressed(
      f, poses=poses, errors=errors, metrics=metrics, wm_valid=wm_valid
    )
  os.replace(temporary, path)
  failure = first_failure(errors)
  return {
    "file": path.name,
    "frames": len(poses),
    "first_failure": failure,
    "fail_rules": []
    if failure is None
    else [
      rule
      for rule, err, threshold in zip(
        FAILURE_RULES, errors[failure], THRESHOLDS, strict=True
      )
      if err > threshold
    ],
    "bytes": path.stat().st_size,
    "memory_bytes": poses.nbytes + errors.nbytes + metrics.nbytes + wm_valid.nbytes,
  }


def load_clip(directory: Path, entry: dict, nq: int) -> dict[str, np.ndarray]:
  path = directory / entry["file"]
  if path.parent.resolve() != directory.resolve():
    raise ValueError("Clip file must be inside the cache directory")
  with np.load(path, allow_pickle=False) as archive:
    arrays = {key: archive[key] for key in ("poses", "errors", "metrics", "wm_valid")}
  n = entry["frames"]
  expected = {
    "poses": (n, 3, nq),
    "errors": (n, 3),
    "metrics": (n, 4),
    "wm_valid": (n,),
  }
  for key, shape in expected.items():
    if arrays[key].shape != shape:
      raise ValueError(f"Invalid {key} shape in {path}")
  if not np.isfinite(arrays["poses"]).all():
    raise ValueError(f"Invalid poses in {path}")
  return arrays


class ReplayCursor:
  """Frame indices are zero based and include the failure frame itself."""

  def __init__(self, frames: int, failure: int | None):
    if frames < 1 or (failure is not None and not 0 <= failure < frames):
      raise ValueError("Invalid clip length or failure frame")
    self.frames, self.failure = frames, failure
    self.frame = 0
    self.end_on_failure = True

  @property
  def end(self) -> int:
    return (
      self.failure
      if self.end_on_failure and self.failure is not None
      else self.frames - 1
    )

  def seek(self, frame: int) -> None:
    self.frame = max(0, min(int(frame), self.end))

  def advance(self) -> bool:
    """Advance one recorded frame, returning whether playback wrapped."""
    wrapped = self.frame >= self.end
    self.frame = 0 if wrapped else self.frame + 1
    return wrapped
