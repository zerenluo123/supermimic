"""The G1-D trains on proprio with a 10-frame history, the planar diff-drive chassis,
narrow reward kernels and an averaged actor unless told otherwise."""

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


def test_g1d_defaults_turn_on_the_planar_diff_drive_chassis():
  cfg = SuperMimicRunnerCfg()
  apply_g1d_defaults(cfg, [])
  assert (cfg.g1d_planar_chassis, cfg.g1d_diff_drive) == (True, True)


@pytest.mark.parametrize(
  ("args", "expected"),
  [
    (["--runner.g1d-diff-drive", "False"], (True, False)),
    (["--runner.g1d_diff_drive=False"], (True, False)),
    (["--runner.g1d-planar-chassis", "False"], (False, False)),
    (["--runner.g1d_planar_chassis=False"], (False, False)),
  ],
)
def test_g1d_keeps_an_explicit_chassis_choice(args, expected):
  cfg = SuperMimicRunnerCfg()  # tyro has parsed the explicit False into the cfg
  apply_g1d_defaults(cfg, args)
  assert (cfg.g1d_planar_chassis, cfg.g1d_diff_drive) == expected


def test_g1d_defaults_turn_on_the_narrow_kernels_and_the_actor_ema():
  cfg = SuperMimicRunnerCfg()
  apply_g1d_defaults(cfg, [])
  assert (cfg.wm_reward_std_scale, cfg.policy.actor_ema) == (0.35, 0.995)


@pytest.mark.parametrize(
  "args",
  [
    ["--runner.wm-reward-std-scale", "1.0", "--runner.policy.actor-ema", "0.0"],
    ["--runner.wm_reward_std_scale=1.0", "--runner.policy.actor_ema=0.0"],
  ],
)
def test_g1d_keeps_explicit_wide_kernels_and_no_ema(args):
  cfg = SuperMimicRunnerCfg()  # tyro has parsed the explicit values into the cfg
  apply_g1d_defaults(cfg, args)
  assert (cfg.wm_reward_std_scale, cfg.policy.actor_ema) == (1.0, 0.0)
