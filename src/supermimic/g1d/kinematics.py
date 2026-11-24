"""Differentiable forward kinematics of the G1-D in mjlab's ForwardKinematics interface.

mjlab's ``ForwardKinematics`` needs a free-joint root with only hinges below it.
On the G1-D the root (``AGV_link``) sits on planar joints and the column is two
lift slides, each carrying half the column height. The root pose and twist are
given directly, the 17 state joints are ``[column, 16 upper]``.

The pass is mjlab's: every returned body's path from the root is walked at once,
one depth level per loop turn, on per-component ``(..., R)`` tensors, so every op
has the same shape and a compiler fuses the pass into a few kernels. A step is a
body's fixed offset, then its joint: a hinge turns ``q - qpos0`` about its anchor,
a lift slide moves ``x = q / 2 - qpos0`` along its axis. Each body composes the
steps on its path root first and sums their twists, taken at the root origin; the
root pose and twist apply last:

  slide:  p = p_frame + axis_w x             v = v_parent + w_parent x (p - p_parent) + axis_w xdot
  hinge:  turn about the anchor by (q - qpos0)

Link-origin velocities in the world frame, as ``body_link_lin_vel_w``. When a
gradient is wanted, the pass is recomputed in the backward instead of keeping its
per-level intermediates (some 10 KB per sample for the tracked bodies).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from supermimic.g1d import constants as C

_FIXED, _SLIDE, _HINGE = 0, 1, 2

Vec = tuple[torch.Tensor, ...]


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
    lift = {jnames[n] for n in C.LIFT_JOINTS}
    zero, identity = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def step(body: int) -> tuple:
      """``(kind, joint column, scale, qpos0, axis, anchor)`` of the body's joint."""
      if int(mj_model.body_jntnum[body]) == 0:
        return _FIXED, 0, 0.0, 0.0, zero, zero
      assert int(mj_model.body_jntnum[body]) == 1, mj_model.body(body).name
      j = int(mj_model.body_jntadr[body])
      qpos0 = float(mj_model.qpos0[mj_model.jnt_qposadr[j]])
      if j in lift:
        assert mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
        return _SLIDE, 0, 0.5, qpos0, mj_model.jnt_axis[j], zero
      assert mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
      col = C.STATE_JOINTS.index(mj_model.joint(j).name.split("/")[-1])
      return _HINGE, col, 1.0, qpos0, mj_model.jnt_axis[j], mj_model.jnt_pos[j]

    # Each returned body's path below the root, root side first, front-padded with
    # identity steps to one depth.
    paths = []
    for name in body_names:
      b, path = names[name], []
      while b != root:
        path.append(b)
        b = int(mj_model.body_parentid[b])
      paths.append(path[::-1])
    depth = max(1, *map(len, paths))
    paths = [[None] * (depth - len(path)) + path for path in paths]
    # Per path step, with o its offset quaternion: the joint anchor and axis in the
    # parent frame, the anchor in the step frame, o, and o * (0, axis); a hinge
    # step's quaternion is cos(h) o + sin(h) o * (0, axis) for h = (q - qpos0) / 2.
    # A step without a joint reads column J, a zero pad after the J joints.
    rows = len(paths)
    consts = np.zeros((depth, rows, 17))
    col = np.full((depth, rows), len(C.STATE_JOINTS), dtype=np.int64)
    scale, qpos0, hinge, slide = (np.zeros((depth, rows)) for _ in range(4))
    for r, path in enumerate(paths):
      for d, body in enumerate(path):
        if body is None:
          consts[d, r, 9:13] = identity
          continue
        kind, c, s, q0, axis, anchor = step(body)
        offset = mj_model.body_quat[body]
        parent_anchor, parent_axis, turned = np.zeros(3), np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(parent_anchor, anchor, offset)
        mujoco.mju_rotVecQuat(parent_axis, axis, offset)
        mujoco.mju_mulQuat(turned, offset, np.concatenate([[0.0], axis]))
        consts[d, r] = np.concatenate(
          [mj_model.body_pos[body] + parent_anchor, parent_axis, anchor, offset, turned]
        )
        if kind != _FIXED:
          col[d, r], scale[d, r], qpos0[d, r] = c, s, q0
          hinge[d, r], slide[d, r] = float(kind == _HINGE), float(kind == _SLIDE)

    def t(a):
      return torch.as_tensor(a, dtype=dtype, device=device)

    self.consts = t(consts)
    self.col = torch.as_tensor(col, device=device)
    self.scale, self.qpos0, self.hinge, self.slide = map(
      t, (scale, qpos0, hinge, slide)
    )
    self.body_names = tuple(body_names)
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
    args = (root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel)
    if torch.is_grad_enabled() and any(a.requires_grad for a in args):
      return checkpoint(self._pass, *args, use_reentrant=False)
    return self._pass(*args)

  def _pass(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    joint_pos, joint_vel = (
      torch.nn.functional.pad(x, (0, 1)) for x in (joint_pos, joint_vel)
    )
    zero, one = root_pos.new_zeros(()), root_pos.new_ones(())
    pos: Vec = (zero, zero, zero)
    quat: Vec = (one, zero, zero, zero)
    lin: Vec = (zero, zero, zero)
    ang: Vec = (zero, zero, zero)
    for col, scale, qpos0, hinge, slide, consts in zip(
      self.col, self.scale, self.qpos0, self.hinge, self.slide, self.consts, strict=True
    ):
      c = consts.unbind(-1)
      x = joint_pos[..., col] * scale - qpos0
      xd = joint_vel[..., col] * scale
      half = x * hinge * 0.5
      # The joint anchor and axis in the root frame, from the parent's pose.
      anchor = _add(pos, _rotate(quat, c[:3]))
      axis = _rotate(quat, c[3:6])
      twist = tuple(a * (xd * hinge) for a in axis)
      turn = tuple(
        half.cos() * o + half.sin() * t for o, t in zip(c[9:13], c[13:], strict=True)
      )
      quat = _mul(quat, turn)
      shift, shift_rate = x * slide, xd * slide
      pos = _add(_sub(anchor, _rotate(quat, c[6:9])), tuple(a * shift for a in axis))
      ang = _add(ang, twist)
      lin = _add(_add(lin, _cross(anchor, twist)), tuple(a * shift_rate for a in axis))

    w, x, y, z = root_quat.unsqueeze(-2).unbind(-1)
    norm = (w * w + x * x + y * y + z * z).sqrt()
    root = (w / norm, x / norm, y / norm, z / norm)
    pos = _rotate(root, pos)
    ang = _add(root_ang_vel.unsqueeze(-2).unbind(-1), _rotate(root, ang))
    lin = _add(root_lin_vel.unsqueeze(-2).unbind(-1), _rotate(root, lin))
    lin = _add(lin, _cross(ang, pos))
    pos = _add(root_pos.unsqueeze(-2).unbind(-1), pos)
    state = torch.stack([*pos, *_mul(root, quat), *lin, *ang], dim=-1)
    pos_w, quat_w, lin_vel_w, ang_vel_w = state.split((3, 4, 3, 3), dim=-1)
    return pos_w, quat_w, lin_vel_w, ang_vel_w


def _add(a: Vec, b: Vec) -> Vec:
  return tuple(x + y for x, y in zip(a, b, strict=True))


def _sub(a: Vec, b: Vec) -> Vec:
  return tuple(x - y for x, y in zip(a, b, strict=True))


def _cross(a: Vec, b: Vec) -> Vec:
  return (
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  )


def _rotate(q: Vec, v: Vec) -> Vec:
  t = tuple(2 * x for x in _cross(q[1:], v))
  return _add(_add(v, tuple(q[0] * x for x in t)), _cross(q[1:], t))


def _mul(a: Vec, b: Vec) -> Vec:
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return (
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  )
