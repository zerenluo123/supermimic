"""URR keypoints of the G1-D (``constants.URR_BODIES``) in its own chassis frame.

A pose is ``[positions (5×3), rot6d of the chest and hands (3×6)]`` (33), a twist
``[linear velocities (5×3), angular velocities of the chest and hands (3×3)]`` (24),
all in the chassis (``AGV_link``) frame: the root sits at the origin with the
identity orientation, and its twist is given in its own frame, so the velocities
are the world ones turned into the chassis frame. A reference frame's keypoints
depend only on that frame, so the motion's are tabulated once
(``motion_urr_table``), or computed from the reference on each use
(``motion_urr_rows``), which keeps no table.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from supermimic.g1d import constants as C
from supermimic.g1d.env_cfg import HalfTable
from supermimic.utils.helpers import quat_to_rot6d

_TABLE_CHUNK = 1 << 16


def urr_features(
  fk,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  root_lin_vel_b: torch.Tensor,
  root_ang_vel_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """``(pose (..., 33), twist (..., 24))`` of the keypoints from the state joints and
  the chassis twist in its own frame; ``fk`` is a ``G1DForwardKinematics`` whose
  bodies include ``URR_BODIES``."""
  rows = [fk.body_names.index(name) for name in C.URR_BODIES]
  batch = joint_pos.shape[:-1]
  root_pos = joint_pos.new_zeros(*batch, 3)
  root_quat = joint_pos.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(*batch, 4)
  bodies = fk(root_pos, root_quat, joint_pos, root_lin_vel_b, root_ang_vel_b, joint_vel)
  pos, quat, lin, ang = (b[..., rows, :] for b in bodies)
  offset = quat_apply(quat, pos.new_tensor(C.URR_OFFSETS).expand_as(pos))
  oriented = slice(0, C.URR_NUM_ORIENTED)
  pose = torch.cat(
    [(pos + offset).flatten(-2), quat_to_rot6d(quat[..., oriented, :]).flatten(-2)], -1
  )
  twist = torch.cat(
    [
      (lin + torch.cross(ang, offset, dim=-1)).flatten(-2),
      ang[..., oriented, :].flatten(-2),
    ],
    -1,
  )
  return pose, twist


def motion_urr_rows(motion, idx: torch.Tensor, with_twist: bool) -> torch.Tensor:
  """``motion_urr_table(motion, with_twist)[idx]`` computed from the reference at
  ``idx`` on use, rounded to half precision as the table would store it."""
  root_quat = motion.root_quat_w[idx]
  pose, twist = urr_features(
    motion.fk,
    motion.joint_pos[idx],
    motion.joint_vel[idx],
    quat_apply_inverse(root_quat, motion.root_lin_vel_w[idx]),
    quat_apply_inverse(root_quat, motion.root_ang_vel_w[idx]),
  )
  rows = torch.cat([pose, twist], -1) if with_twist else pose
  if isinstance(motion.joint_pos, HalfTable):
    rows = round_to_half(rows)
  return rows


def round_to_half(x: torch.Tensor) -> torch.Tensor:
  """``x.half().float()`` by exact power-of-two arithmetic, which the compiler keeps
  (it drops a ``.half().float()`` round trip as a no-op)."""
  _, e = torch.frexp(x)  # |x| in [2^(e-1), 2^e); half keeps 11 significant bits
  scale = torch.ldexp(torch.ones_like(x), 11 - e.clamp(min=-13))
  return torch.round(x * scale) / scale


@torch.no_grad()
def motion_urr_table(motion, with_twist: bool):
  """Per-frame reference keypoints ``[pose, twist]`` (57 wide; 33 without
  ``with_twist``) in that frame's reference chassis frame, tabulated on the motion
  at its first use and kept there; half precision when its joints are."""
  width = C.URR_POSE_DIM + (C.URR_VEL_DIM if with_twist else 0)
  table = getattr(motion, "_urr_table", None)
  if table is not None and table.shape[1] >= width:
    return table
  out = torch.empty(
    motion.time_step_total, width, device=motion.joint_pos.device, dtype=torch.float32
  )
  for start in range(0, motion.time_step_total, _TABLE_CHUNK):
    rows = slice(start, min(start + _TABLE_CHUNK, motion.time_step_total))
    root_quat = motion.root_quat_w[rows]
    pose, twist = urr_features(
      motion.fk,
      motion.joint_pos[rows],
      motion.joint_vel[rows],
      quat_apply_inverse(root_quat, motion.root_lin_vel_w[rows]),
      quat_apply_inverse(root_quat, motion.root_ang_vel_w[rows]),
    )
    out[rows] = torch.cat([pose, twist], -1)[:, :width]
  if isinstance(motion.joint_pos, HalfTable):
    out = HalfTable(out, str(out.device))
  motion._urr_table = out
  return out
