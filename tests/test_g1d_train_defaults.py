"""The G1-D trains on proprio with a 10-frame history unless told otherwise."""

from __future__ import annotations

import pytest

from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
from supermimic.scripts.train import apply_g1d_defaults


def test_g1d_defaults_fill_in_proprio_and_history():
  cfg = SuperMimicRunnerCfg()
  apply_g1d_defaults(cfg, [])
  assert (cfg.representation, cfg.actor_obs, cfg.history_length) == (
    "g1d",
    "proprio",
    10,
  )


def test_g1d_keeps_an_explicit_history():
  cfg = SuperMimicRunnerCfg(history_length=1)
  apply_g1d_defaults(cfg, ["--runner.history-length", "1"])
  assert cfg.history_length == 1
  cfg = SuperMimicRunnerCfg(history_length=5)
  apply_g1d_defaults(cfg, ["--runner.history_length=5"])
  assert cfg.history_length == 5


@pytest.mark.parametrize(
  "args", [["--runner.actor-obs", "rep_state"], ["--runner.actor_obs=rep_state"]]
)
def test_g1d_refuses_the_true_state(args):
  with pytest.raises(ValueError, match="proprio"):
    apply_g1d_defaults(SuperMimicRunnerCfg(), args)
