"""Convert SuperMimic-Reproduced G1-D clips to mjlab's motion format.

Reproduced's G1-D npz (``g1d/motion_convert.py``) holds every body's world link
state and the 17 state joints at 50 fps; mjlab's ``MotionLoader`` reads the root
link's pose and twist and the joints. The root is ``AGV_link``.

A Reproduced packed library (``build_library.py``, a ``.pt`` of 47-wide rows
``[root_pos, root_quat, joint_pos, root_lin_vel_w, root_ang_vel_w, joint_vel]``,
fp16 with fp32 root positions) becomes mjlab's packed directory instead: one
``<field>.npy`` each, ``motion_ends.npy``, ``joint_names.json`` and
``motion_names.json``, which loads without opening every clip.

  uv run python -m supermimic.g1d.convert_motions --src <npz dir | library .pt> \
    --out-dir <out dir>
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro

from supermimic.g1d import constants as C


def convert_clip(src: Path, dst: Path) -> int:
  d = np.load(src)
  if list(d["joint_names"]) != list(C.STATE_JOINTS):
    raise ValueError(f"{src}: joints {list(d['joint_names'])}")
  if float(d["fps"][0]) != 50.0:
    raise ValueError(f"{src}: {float(d['fps'][0])} fps, the control rate is 50")
  root = list(d["body_names"]).index(C.ROOT_BODY)
  np.savez(
    dst,
    root_pos_w=d["body_pos_w"][:, root],
    root_quat_w=d["body_quat_w"][:, root],
    root_lin_vel_w=d["body_lin_vel_w"][:, root],
    root_ang_vel_w=d["body_ang_vel_w"][:, root],
    joint_pos=d["joint_pos"],
    joint_vel=d["joint_vel"],
    joint_names=np.array(C.STATE_JOINTS),
    fps=d["fps"],
  )
  return int(d["joint_pos"].shape[0])


def convert_library(src: Path, out_dir: Path) -> tuple[int, int]:
  import torch

  pack = torch.load(src, map_location="cpu", weights_only=False)
  ref = pack["ref"].float().numpy()
  if ref.shape[1] != 13 + 2 * C.NJ:
    raise ValueError(f"{src}: {ref.shape[1]}-wide rows")
  starts, lengths = np.asarray(pack["starts"]), np.asarray(pack["lengths"])
  if starts[0] != 0 or not (starts[1:] == starts[:-1] + lengths[:-1]).all():
    raise ValueError(f"{src}: clips are not contiguous")
  nj = C.NJ
  fields = {
    "root_pos_w": pack["root_pos"].float().numpy()
    if "root_pos" in pack
    else ref[:, 0:3],
    "root_quat_w": ref[:, 3:7],
    "joint_pos": ref[:, 7 : 7 + nj],
    "root_lin_vel_w": ref[:, 7 + nj : 10 + nj],
    "root_ang_vel_w": ref[:, 10 + nj : 13 + nj],
    "joint_vel": ref[:, 13 + nj :],
  }
  out_dir.mkdir(parents=True, exist_ok=True)
  for name, arr in fields.items():
    np.save(out_dir / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float32))
  np.save(out_dir / "motion_ends.npy", (starts + lengths).astype(np.int64))
  (out_dir / "joint_names.json").write_text(json.dumps(list(C.STATE_JOINTS)))
  (out_dir / "motion_names.json").write_text(
    json.dumps([str(n) for n in pack["names"]])
  )
  return len(starts), int(ref.shape[0])


def main(src: Path, out_dir: Path) -> None:
  if src.suffix == ".pt":
    clips, frames = convert_library(src, out_dir)
  else:
    files = sorted(src.glob("*.npz"))
    if not files:
      raise SystemExit(f"no npz in {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    clips, frames = len(files), sum(convert_clip(f, out_dir / f.name) for f in files)
  print(f"{clips} clips, {frames} frames ({frames / 50 / 3600:.2f} h) -> {out_dir}")


if __name__ == "__main__":
  tyro.cli(main)
