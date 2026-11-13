"""mjlab entity for the G1-D, the plant of SuperMimic-Reproduced (g1d).

* Upper body: MuJoCo position actuators with the G1 motor model (kp = armature *
  (10 Hz)^2, damping ratio 2, BeyondMimic action scale 0.25 * effort / kp); the
  waist pitch takes the Unitree self-locking gains.
* Chassis: velocity drives on the planar x / y / yaw joints with a first-order time
  constant TAU (kv = mass / TAU on x and y, inertia / TAU on yaw). The command is
  (vx, wz) in the heading frame.
* Column: velocity drives on both slides (kv = 0.5 * m_stage / TAU each) plus a
  constant gravity feed-forward; the MJCF equality keeps the two slides equal.
* No self-collision; robot geoms still collide with the ground plane.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np
from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  BuiltinVelocityActuator,
  BuiltinVelocityActuatorCfg,
)
from mjlab.actuator import builtin_group as _builtin_group
from mjlab.actuator.actuator import TransmissionType
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1c
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from supermimic.g1d import constants as C

COLLISION_GEOMS = r".*_col_[0-9]+"


def load_spec(visual_mesh_dir: str | None = None) -> mujoco.MjSpec:
  """G1-D spec for simulation: visual geoms (and the meshes only they use), the
  XML floor and light are dropped, so only the small collider meshes ship.

  ``visual_mesh_dir`` (a directory with the XML's visual STL meshes, which do not
  ship) keeps the visual geoms, for viewing; they collide with nothing."""
  spec = mujoco.MjSpec.from_file(str(C.G1D_XML))
  for g in list(spec.geoms):
    if (g.classname.name == "visual" and visual_mesh_dir is None) or g.name == "floor":
      spec.delete(g)
  used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
  for m in list(spec.meshes):
    if m.name not in used:
      spec.delete(m)
    elif visual_mesh_dir is not None and m.file.lower().endswith(".stl"):
      m.file = os.path.join(os.path.abspath(visual_mesh_dir), os.path.basename(m.file))
  for light in list(spec.lights):
    spec.delete(light)
  # The XML's <size memory> conflicts with the scene when mjlab attaches the entity.
  spec.memory = -1
  return spec


@functools.lru_cache(maxsize=1)
def load_model() -> mujoco.MjModel:
  return load_spec().compile()


def _subtree_mass(model: mujoco.MjModel, body: str) -> float:
  root = model.body(body).id
  ids = [root]
  for b in range(root + 1, model.nbody):  # bodies are topologically ordered
    if int(model.body_parentid[b]) in ids:
      ids.append(b)
  return float(model.body_mass[ids].sum())


@functools.lru_cache(maxsize=1)
def drive_params() -> dict[str, float]:
  """Velocity-drive gains and the column feed-forward, at the zero pose."""
  m = load_model()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  M = np.zeros((m.nv, m.nv))
  mujoco.mj_fullM(m, d, M)
  yaw_dof = int(m.jnt_dofadr[m.joint(C.BASE_JOINTS[2]).id])
  mass = float(m.body_mass.sum())
  m_mt, m_it = _subtree_mass(m, "LZ_mt_Link"), _subtree_mass(m, "LZ_it_Link")
  # Effective mass of one slide coordinate when both slides carry half the column.
  m_stage = (m_mt - m_it) + 4.0 * m_it
  g = 9.81
  return {
    "kv_xy": mass / C.TAU,
    "kv_yaw": float(M[yaw_dof, yaw_dof]) / C.TAU,
    "kv_lift": 0.5 * m_stage / C.TAU,
    "ff_mt": m_mt * g,
    "ff_it": m_it * g,
  }


def _motor(kind: str) -> tuple[float, float, float, float]:
  """(stiffness, damping, effort, armature) of an upper-body motor class."""
  if kind == "5020":
    return (
      g1c.STIFFNESS_5020,
      g1c.DAMPING_5020,
      g1c.ACTUATOR_5020.effort_limit,
      g1c.ARMATURE_5020,
    )
  if kind == "4010":
    return (
      g1c.STIFFNESS_4010,
      g1c.DAMPING_4010,
      g1c.ACTUATOR_4010.effort_limit,
      g1c.ARMATURE_4010,
    )
  if kind == "waist_yaw":
    return (
      g1c.STIFFNESS_7520_14,
      g1c.DAMPING_7520_14,
      g1c.ACTUATOR_7520_14.effort_limit,
      g1c.ARMATURE_7520_14,
    )
  if kind == "waist_pitch":
    return (
      C.WAIST_PITCH_KP,
      C.WAIST_PITCH_KD,
      C.WAIST_PITCH_EFFORT,
      2.0 * g1c.ARMATURE_5020,
    )
  raise KeyError(kind)


def upper_action_scale() -> list[float]:
  """16 position scales: 0.25 * effort / kp (G1 rule), the waist pitch by hand."""
  out = []
  for kind in C.UPPER_MOTORS:
    if kind == "waist_pitch":
      out.append(C.WAIST_PITCH_SCALE)
      continue
    kp, _, effort, _ = _motor(kind)
    out.append(0.25 * effort / kp)
  return out


def default_state_q() -> list[float]:
  """Nominal state joints ``[column, 16 upper]``."""
  return [0.0] + [C.DEFAULT_JOINT_POS.get(n, 0.0) for n in C.UPPER_JOINTS]


class _VelocityFeedForwardActuator(BuiltinVelocityActuator):
  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    super().edit_spec(spec, target_names)
    for act, name in zip(self._mjs_actuators, target_names, strict=True):
      act.biasprm[0] = self.cfg.feed_forward[name]


# mjlab routes builtin actuators to their target tensor by exact type: ctrl of this
# one is the velocity target.
_builtin_group._TARGET_TENSOR_MAP[
  (_VelocityFeedForwardActuator, TransmissionType.JOINT)
] = "joint_vel_target"


@dataclass(kw_only=True)
class VelocityFeedForwardActuatorCfg(BuiltinVelocityActuatorCfg):
  """<velocity> drive plus a constant force: f = kv (ctrl - qdot) + ff (MuJoCo's
  bias term, so implicitfast still integrates the damping implicitly). Used for
  the lift slides with their gravity load as ff."""

  feed_forward: dict[str, float] = field(default_factory=dict)

  def build(self, entity, target_ids, target_names):
    return _VelocityFeedForwardActuator(self, entity, target_ids, target_names)


def _actuators() -> tuple:
  acts = []
  for name, kind in zip(C.UPPER_JOINTS, C.UPPER_MOTORS, strict=True):
    kp, kd, effort, arm = _motor(kind)
    acts.append(
      BuiltinPositionActuatorCfg(
        target_names_expr=(name,),
        stiffness=kp,
        damping=kd,
        effort_limit=effort,
        armature=arm,
      )
    )
  p = drive_params()
  acts.append(
    BuiltinVelocityActuatorCfg(
      target_names_expr=C.BASE_JOINTS[:2], damping=p["kv_xy"], effort_limit=1.0e5
    )
  )
  acts.append(
    BuiltinVelocityActuatorCfg(
      target_names_expr=C.BASE_JOINTS[2:], damping=p["kv_yaw"], effort_limit=1.0e5
    )
  )
  acts.append(
    VelocityFeedForwardActuatorCfg(
      target_names_expr=C.LIFT_JOINTS,
      damping=p["kv_lift"],
      effort_limit=1.0e5,
      feed_forward={C.LIFT_JOINTS[0]: p["ff_mt"], C.LIFT_JOINTS[1]: p["ff_it"]},
    )
  )
  return tuple(acts)


NO_SELF_COLLISION = CollisionCfg(
  geom_names_expr=(COLLISION_GEOMS,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=0,
)


def get_g1d_robot_cfg(visual_mesh_dir: str | None = None) -> EntityCfg:
  """``visual_mesh_dir``: see :func:`load_spec` (viewing only)."""
  return EntityCfg(
    # mjlab wraps a fixed-base entity in a mocap body at init_state.pos; AGV_link
    # keeps its XML offset (0.1113 m) below it. Unmatched joints default to 0.
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
      joint_pos=dict(C.DEFAULT_JOINT_POS),
      joint_vel={".*": 0.0},
    ),
    collisions=(NO_SELF_COLLISION,),
    spec_fn=functools.partial(load_spec, visual_mesh_dir),
    articulation=EntityArticulationInfoCfg(
      actuators=_actuators(), soft_joint_pos_limit_factor=0.9
    ),
  )
