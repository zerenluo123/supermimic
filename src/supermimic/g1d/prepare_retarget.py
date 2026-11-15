"""Stream ScaleRetarget v1 pkl clips into one compact, 50 Hz motion directory.

Run on a CPU data worker with numpy and joblib installed. The resampling follows
SuperMimic-Reproduced's motion_convert.py; planar root kinematics are analytic.
Only v1/bvh is read. Full relative paths preserve Embody3D capture/person names.
The source scan hashes every pkl; the second pass verifies it before conversion.
Outputs are published only when complete, on a normal filesystem (not OBS).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from supermimic.g1d import constants as C

FIELDS = {
  "root_pos_w": (3, "float32"),
  "root_quat_w": (4, "float16"),
  "joint_pos": (C.NJ, "float16"),
  "root_lin_vel_w": (3, "float16"),
  "root_ang_vel_w": (3, "float16"),
  "joint_vel": (C.NJ, "float16"),
}
FPS = 50.0


def frame_count(n: int, fps: float) -> int:
  if n < 1 or not math.isfinite(fps) or fps <= 0:
    raise ValueError(f"Invalid input length/fps: {n}, {fps}")
  return math.floor(((n - 1) / fps) / (1 / FPS) + 1e-6) + 1


def resample(raw: dict) -> dict[str, np.ndarray]:
  """Return unquantized fp64 fields, matching the existing 30 -> 50 Hz path."""
  pos = np.asarray(raw["root_pos"], dtype=np.float64)
  quat = np.asarray(raw["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]
  dof = np.asarray(raw["dof_pos"], dtype=np.float64)
  n, fps = len(pos), float(raw["fps"])
  n_out = frame_count(n, fps)
  if pos.shape != (n, 3) or quat.shape != (n, 4) or dof.shape != (n, 18):
    raise ValueError(f"Unexpected input shapes: {pos.shape}, {quat.shape}, {dof.shape}")
  if not all(np.isfinite(a).all() for a in (pos, quat, dof)):
    raise ValueError("Nonfinite motion data")
  norms = np.linalg.norm(quat, axis=1, keepdims=True)
  if np.any(norms < 1e-8):
    raise ValueError("Zero root quaternion")
  quat /= norms
  if np.max(np.abs(pos[:, 2] - C.AGV_HEIGHT)) > 1e-6:
    raise ValueError("Root height differs from the G1-D planar base")
  if np.max(np.abs(quat[:, 1:3])) > 1e-6:
    raise ValueError("G1-D root must rotate around the vertical axis only")
  joints = np.concatenate((dof[:, :1] + dof[:, 1:2], dof[:, 2:]), axis=1)
  duration = (n - 1) / fps
  times = np.arange(n_out, dtype=np.float64) / FPS
  phase = times / duration if duration > 0 else np.zeros_like(times)
  i0 = np.clip(np.floor(phase * (n - 1)).astype(np.int64), 0, n - 1)
  i1 = np.minimum(i0 + 1, n - 1)
  blend = (phase * (n - 1) - i0)[:, None]
  root_pos = pos[i0] * (1 - blend) + pos[i1] * blend
  joint_pos = joints[i0] * (1 - blend) + joints[i1] * blend
  q0, q1 = quat[i0], quat[i1]
  dot = np.sum(q0 * q1, axis=1, keepdims=True)
  q1 = np.where(dot < 0, -q1, q1)
  dot = np.abs(dot)
  theta = np.arccos(np.clip(dot, -1, 1))
  linear = dot > 0.9995
  denominator = np.where(linear, 1.0, np.sin(theta))
  w0 = np.where(linear, 1 - blend, np.sin((1 - blend) * theta) / denominator)
  w1 = np.where(linear, blend, np.sin(blend * theta) / denominator)
  q = w0 * q0 + w1 * q1
  q /= np.linalg.norm(q, axis=1, keepdims=True)
  yaw = np.arctan2(
    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
  )
  # MuJoCo's planar base reconstructs the root from x, y and this yaw.
  root_pos[:, 2] = C.AGV_HEIGHT
  root_quat = np.zeros_like(q)
  root_quat[:, 0], root_quat[:, 3] = np.cos(yaw / 2), np.sin(yaw / 2)
  linear_vel = (
    np.gradient(root_pos, 1 / FPS, axis=0) if n_out > 1 else np.zeros_like(root_pos)
  )
  joint_vel = (
    np.gradient(joint_pos, 1 / FPS, axis=0) if n_out > 1 else np.zeros_like(joint_pos)
  )
  angular_vel = np.zeros((n_out, 3))
  if n_out >= 3:
    delta = yaw[2:] - yaw[:-2]
    omega = np.arctan2(np.sin(delta), np.cos(delta)) * (FPS / 2)
    angular_vel[:, 2] = np.concatenate((omega[:1], omega, omega[-1:]))
  return dict(
    root_pos_w=root_pos,
    root_quat_w=root_quat,
    joint_pos=joint_pos,
    root_lin_vel_w=linear_vel,
    root_ang_vel_w=angular_vel,
    joint_vel=joint_vel,
  )


def load_source(path: Path) -> tuple[dict, str]:
  import joblib  # CPU preparation dependency; not needed by training.

  data = path.read_bytes()
  return joblib.load(io.BytesIO(data)), hashlib.sha256(data).hexdigest()


def inspect_source(item: tuple[str, Path, str]) -> dict:
  dataset, path, relative = item
  raw, digest = load_source(path)
  n, fps = len(raw["root_pos"]), float(raw["fps"])
  return dict(
    dataset=dataset,
    source=str(path),
    name=f"{dataset}/{relative}",
    sha256=digest,
    input_frames=n,
    input_fps=fps,
    frames=frame_count(n, fps),
  )


def convert_source(entry: dict) -> dict[str, np.ndarray]:
  raw, digest = load_source(Path(entry["source"]))
  if digest != entry["sha256"]:
    raise ValueError(f"Source changed during conversion: {entry['source']}")
  try:
    arrays = resample(raw)
    for field, (_, dtype) in FIELDS.items():
      # The old compact converter rounds through fp32 before storing fp16.
      arrays[field] = arrays[field].astype(np.float32).astype(dtype)
      if not np.isfinite(arrays[field]).all():
        raise ValueError(f"Nonfinite {field} after quantization")
    if len(arrays["root_pos_w"]) != entry["frames"]:
      raise ValueError("Resampling length changed")
    return arrays
  except Exception as exc:
    raise ValueError(f"{entry['source']}: {exc}") from exc


def bounded_map(executor, function, items, batch_size):
  # Python <3.14 Executor.map otherwise submits the entire dataset eagerly.
  for start in range(0, len(items), batch_size):
    yield from executor.map(function, items[start : start + batch_size])


def prepare(
  root: Path,
  out: Path,
  datasets: list[str],
  workers: int = 8,
  limit_per_dataset: int = 0,
) -> dict:
  root, out = root.resolve(), out.resolve()
  if out.exists() or out.with_name(out.name + ".partial").exists():
    raise FileExistsError(f"Output or partial output already exists: {out}")
  if workers < 1 or limit_per_dataset < 0 or len(set(datasets)) != len(datasets):
    raise ValueError("Invalid worker count, limit or duplicate dataset")
  sources, versions = [], {}
  for dataset in datasets:
    version = root / dataset / "v1"
    bvh = version / "bvh"
    files = sorted(bvh.rglob("*.pkl"))
    if not files:
      raise ValueError(f"No pkl clips in {bvh}")
    if limit_per_dataset:
      # Evenly spaced, deterministic coverage across the sorted tree.
      indices = np.linspace(
        0, len(files) - 1, min(limit_per_dataset, len(files)), dtype=int
      )
      files = [files[i] for i in indices]
    sources.extend(
      (dataset, f, f.relative_to(bvh).with_suffix("").as_posix()) for f in files
    )
    versions[dataset] = json.loads((version / "run.json").read_text())
  print(f"Scanning {len(sources)} clips with {workers} CPU workers", flush=True)
  with ThreadPoolExecutor(workers) as executor:
    entries = []
    for entry in bounded_map(executor, inspect_source, sources, 2 * workers):
      entries.append(entry)
      if len(entries) % 5000 == 0:
        print(f"Scanned {len(entries)}/{len(sources)}", flush=True)
  names = [e["name"] for e in entries]
  if len(set(names)) != len(names):
    raise ValueError("Duplicate full motion names")
  ends = np.cumsum([e["frames"] for e in entries], dtype=np.int64)
  total = int(ends[-1])
  summary = dict(
    format_version=1,
    fps=FPS,
    clips=len(entries),
    frames=total,
    hours=total / FPS / 3600,
    compact_bytes=total * 100,
    datasets={},
    versions=versions,
    converter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  )
  for dataset in datasets:
    selected = [e for e in entries if e["dataset"] == dataset]
    frames = sum(e["frames"] for e in selected)
    summary["datasets"][dataset] = dict(
      clips=len(selected), frames=frames, hours=frames / FPS / 3600
    )
  print(json.dumps(summary, indent=2), flush=True)
  partial = out.with_name(out.name + ".partial")
  partial.mkdir(parents=True)
  (partial / "source_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
  tables = {
    name: np.lib.format.open_memmap(
      partial / f"{name}.npy", mode="w+", dtype=dtype, shape=(total, width)
    )
    for name, (width, dtype) in FIELDS.items()
  }
  start = 0
  with ThreadPoolExecutor(workers) as executor:
    for index, arrays in enumerate(
      bounded_map(executor, convert_source, entries, 2 * workers)
    ):
      end = int(ends[index])
      for name, table in tables.items():
        table[start:end] = arrays[name]
      start = end
      if (index + 1) % 5000 == 0:
        print(f"Converted {index + 1}/{len(entries)}", flush=True)
  for table in tables.values():
    table.flush()
  tables.clear()
  np.save(partial / "motion_ends.npy", ends)
  (partial / "joint_names.json").write_text(json.dumps(list(C.STATE_JOINTS)) + "\n")
  (partial / "motion_names.json").write_text(json.dumps(names) + "\n")
  (partial / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
  partial.rename(out)
  print(f"COMPLETE {out}", flush=True)
  return summary


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--datasets", nargs="+", default=["lafan1", "bones_seed", "embody3d"]
  )
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument("--limit-per-dataset", type=int, default=0)
  args = parser.parse_args()
  prepare(args.root, args.out, args.datasets, args.workers, args.limit_per_dataset)
