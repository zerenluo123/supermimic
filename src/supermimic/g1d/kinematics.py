"""Differentiable forward kinematics of the G1-D in mjlab's ForwardKinematics interface.

mjlab's ``ForwardKinematics`` needs a free-joint root with only hinges below it.
On the G1-D the root (``AGV_link``) sits on planar joints and the column is two
lift slides, each carrying half the column height. The root pose and twist are
given directly, the 17 state joints are ``[column, 16 upper]``:

  slide:  p = p_frame + axis_w x             v = v_parent + w_parent x (p - p_parent) + axis_w xdot
  hinge:  turn about the anchor by (q - qpos0)

Link-origin velocities in the world frame, as ``body_link_lin_vel_w``.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_angle_axis,
  quat_mul,
)

from supermimic.g1d import constants as C

_FIXED, _SLIDE, _HINGE = 0, 1, 2


class G1DForwardKinematics:
  """World pose and twist of ``body_names`` from the root (``AGV_link``) state and
  the state joints, as mjlab's ``ForwardKinematics.__call__``."""

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    body_names: tuple[str, ...] | list[str],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
  ):
    # Names in an attached scene carry a prefix; match by the last path part.
    names = {mj_model.body(b).name.split("/")[-1]: b for b in range(mj_model.nbody)}
    jnames = {mj_model.joint(j).name.split("/")[-1]: j for j in range(mj_model.njnt)}
    root = names[C.ROOT_BODY]
    need = set()
    for name in body_names:
      b = names[name]
      while b != root:
        need.add(b)
        b = int(mj_model.body_parentid[b])
    keep = [root, *sorted(need)]  # MuJoCo numbers parents before children
    local = {b: i for i, b in enumerate(keep)}
    lift = {jnames[n] for n in C.LIFT_JOINTS}
    self._parent, self._kind, self._col, self._scale = [-1], [_FIXED], [0], [0.0]
    pos, quat, jpos, jaxis, q0 = [], [], [], [], []
    for b in keep:
      pos.append(mj_model.body_pos[b])
      quat.append(mj_model.body_quat[b])
      jpos.append(np.zeros(3))
      jaxis.append(np.zeros(3))
      q0.append(0.0)
      if b == root:
        continue
      self._parent.append(local[int(mj_model.body_parentid[b])])
      if int(mj_model.body_jntnum[b]) == 0:
        self._kind.append(_FIXED)
        self._col.append(0)
        self._scale.append(0.0)
        continue
      assert int(mj_model.body_jntnum[b]) == 1, mj_model.body(b).name
      j = int(mj_model.body_jntadr[b])
      jpos[-1], jaxis[-1] = mj_model.jnt_pos[j], mj_model.jnt_axis[j]
      q0[-1] = float(mj_model.qpos0[mj_model.jnt_qposadr[j]])
      if j in lift:
        assert mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
        self._kind.append(_SLIDE)
        self._col.append(0)
        self._scale.append(0.5)
      else:
        assert mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        self._kind.append(_HINGE)
        self._col.append(C.STATE_JOINTS.index(mj_model.joint(j).name.split("/")[-1]))
        self._scale.append(1.0)

    def t(a):
      return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    self._body_pos, self._body_quat = t(pos), t(quat)
    self._jnt_pos, self._jnt_axis, self._qpos0 = t(jpos), t(jaxis), t(q0)
    self.rows = [keep.index(names[n]) for n in body_names]
    self.joint_names = list(C.STATE_JOINTS)

  def __call__(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Positions ``(..., R, 3)``, wxyz quaternions ``(..., R, 4)``, linear and
    angular velocities ``(..., R, 3)`` of the bodies, in the world frame."""
    batch = root_pos.shape[:-1]
    root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True)
    pos, quat, lin, ang = [root_pos], [root_quat], [root_lin_vel], [root_ang_vel]
    for i in range(1, len(self._kind)):
      p, kind = self._parent[i], self._kind[i]
      pp, pq = pos[p], quat[p]
      b_pos = pp + quat_apply(pq, self._body_pos[i].expand(*batch, 3))
      b_quat = quat_mul(pq, self._body_quat[i].expand(*batch, 4))
      if kind == _FIXED:
        w = ang[p]
        v = lin[p] + torch.cross(ang[p], b_pos - pp, dim=-1)
      else:
        x = self._scale[i] * joint_pos[..., self._col[i]] - self._qpos0[i]
        xd = (self._scale[i] * joint_vel[..., self._col[i]]).unsqueeze(-1)
        axis = self._jnt_axis[i].expand(*batch, 3)
        axis_w = quat_apply(b_quat, axis)
        if kind == _SLIDE:
          b_pos = b_pos + axis_w * x.unsqueeze(-1)
          w = ang[p]
          v = lin[p] + torch.cross(ang[p], b_pos - pp, dim=-1) + axis_w * xd
        else:
          anchor_local = self._jnt_pos[i].expand(*batch, 3)
          anchor = b_pos + quat_apply(b_quat, anchor_local)
          b_quat = quat_mul(b_quat, quat_from_angle_axis(x, axis))
          b_pos = anchor - quat_apply(b_quat, anchor_local)
          w_joint = axis_w * xd
          w = ang[p] + w_joint
          v = (
            lin[p]
            + torch.cross(ang[p], b_pos - pp, dim=-1)
            + torch.cross(w_joint, b_pos - anchor, dim=-1)
          )
      pos.append(b_pos)
      quat.append(b_quat)
      lin.append(v)
      ang.append(w)
    rows = self.rows
    return (
      torch.stack([pos[r] for r in rows], dim=-2),
      torch.stack([quat[r] for r in rows], dim=-2),
      torch.stack([lin[r] for r in rows], dim=-2),
      torch.stack([ang[r] for r in rows], dim=-2),
    )
