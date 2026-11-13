"""Replay boundary/serialization tests, independent of trained weights or a GPU."""

import mujoco
import numpy as np
import pytest

from supermimic.g1d import constants as C
from supermimic.g1d.replay import ReplayCursor, first_failure, load_clip, save_clip
from supermimic.g1d.robot import load_model
from supermimic.scripts.record_g1d import pose_from_state


def test_failure_boundary_and_recovery():
  errors = np.zeros((8, 3), np.float32)
  errors[2:4, 0] = 0.106
  failure = first_failure(errors)
  assert failure == 2
  cursor = ReplayCursor(8, failure)
  cursor.seek(100)
  assert cursor.frame == 2  # Failure frame is visible, not the reset pose.
  assert cursor.advance() and cursor.frame == 0
  cursor.end_on_failure = False
  cursor.seek(5)  # Recovery can be viewed without another simulation.
  assert not cursor.advance() and cursor.frame == 6
  cursor.seek(-1)
  assert cursor.frame == 0
  cursor.seek(7)
  assert cursor.advance() and cursor.frame == 0


def test_empty_and_one_frame_clips():
  with pytest.raises(ValueError):
    ReplayCursor(0, None)
  cursor = ReplayCursor(1, None)
  assert cursor.advance() and cursor.frame == 0
  assert first_failure(np.zeros((1, 3))) is None


def test_cache_roundtrip_and_reject_corruption(tmp_path):
  poses = np.random.default_rng(1).normal(size=(10, 3, 21)).astype(np.float32)
  errors = np.zeros((10, 3), np.float32)
  errors[4, 2] = 0.11
  metrics = np.zeros((10, 4), np.float32)
  valid = np.ones(10, bool)
  entry = save_clip(tmp_path / "clip.npz", poses, errors, metrics, valid)
  assert entry["first_failure"] == 4
  assert entry["fail_rules"] == ["column"]
  loaded = load_clip(tmp_path, entry, 21)
  np.testing.assert_array_equal(loaded["poses"], poses)
  assert entry["memory_bytes"] == sum(a.nbytes for a in loaded.values())
  with pytest.raises(ValueError):
    load_clip(tmp_path, entry, 22)
  with pytest.raises(ValueError):
    load_clip(tmp_path, {**entry, "file": "../elsewhere.npz"}, 21)
  poses[0, 0, 0] = np.nan
  with pytest.raises(ValueError):
    save_clip(tmp_path / "bad.npz", poses, errors, metrics, valid)


def test_recorded_pose_reconstructs_chassis_and_column():
  model = load_model()
  names = [model.joint(i).name for i in range(model.njnt)]
  state = np.zeros((2, 7 + C.NJ), np.float32)
  state[:, :3] = [[1, -2, C.AGV_HEIGHT], [-4, 3, C.AGV_HEIGHT]]
  yaw = np.array([0.7, -2.3])
  state[:, 3], state[:, 6] = np.cos(yaw / 2), np.sin(yaw / 2)
  state[:, 7] = [0.2, 0.3]
  pose = pose_from_state(state, names)
  for i in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = pose[i]
    mujoco.mj_kinematics(model, data)
    np.testing.assert_allclose(
      data.xpos[model.body(C.ROOT_BODY).id], state[i, :3], atol=1e-6
    )
    assert data.qpos[model.joint(C.BASE_JOINTS[2]).qposadr][0] == pytest.approx(
      yaw[i], abs=1e-6
    )
    assert sum(
      data.qpos[model.joint(n).qposadr][0] for n in C.LIFT_JOINTS
    ) == pytest.approx(state[i, 7])
