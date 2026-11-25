"""``actor_obs`` is stored only where it is read back: the history warm-start.
Schema and writes share the ``_stores_actor_obs`` predicate. The ``mode`` is stored
only for a goal in parts, which imagination reads it for."""

from types import SimpleNamespace

import torch

from supermimic.replay_buffer import FieldSpec
from supermimic.runners.supermimic_runner import SuperMimicRunner


class _Runner(SuperMimicRunner):
  """A runner holding only what ``_buffer_schema`` reads."""

  def __init__(
    self, actor_obs_mode: str, history_length: int, num_goal_parts: int = 0
  ) -> None:
    self._representation = SimpleNamespace(
      actor_obs_mode=actor_obs_mode, full_state_dim=71, proprio_obs_dim=lambda n: 64 + n
    )
    self._num_actions = 29
    self._sim_context = SimpleNamespace(output_dim=3)
    self._num_goal_parts = num_goal_parts
    self.cfg = SimpleNamespace(history_length=history_length)


def test_actor_obs_stored_only_for_proprio_with_history():
  assert "actor_obs" not in _Runner("proprio", 1)._buffer_schema()
  assert "actor_obs" in _Runner("proprio", 4)._buffer_schema()
  assert "actor_obs" not in _Runner("rep_state", 4)._buffer_schema()


def test_mode_stored_only_for_a_goal_in_parts():
  assert "mode" not in _Runner("proprio", 4)._buffer_schema()
  assert _Runner("proprio", 4, 6)._buffer_schema()["mode"] == FieldSpec(6, torch.bool)
