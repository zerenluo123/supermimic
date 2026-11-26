import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
  gae_returns,
)

NUM_ENVS = 2


class _DummyActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(()))

  def forward(self, obs: TensorDict) -> torch.Tensor:
    return obs["actor"] * self.weight


class _DummyRepresentation(nn.Module):
  dt = 1.0


class _TwoStepEnv:
  """Reward 1 per step; the first step terminates with probability 0.5."""

  max_step_reward = float("inf")

  def __init__(self):
    self.dynamics = nn.Linear(1, 1)
    self.representation = _DummyRepresentation()
    self.step_idx = 0

  def reset(self, batch: TensorDict) -> TensorDict:
    del batch
    self.step_idx = 0
    return self._obs()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
    rewards = 1.0 + actions.squeeze(-1)
    term_prob = torch.full((NUM_ENVS,), 0.5 if self.step_idx == 0 else 0.0)
    self.step_idx += 1
    return self._obs(), rewards, term_prob

  def _obs(self) -> TensorDict:
    obs = {"actor": torch.zeros(NUM_ENVS, 1), "critic": torch.zeros(NUM_ENVS, 1)}
    return TensorDict(obs, batch_size=[NUM_ENVS])


class _ConstantCritic(nn.Module):
  def __init__(self, value: float):
    super().__init__()
    self.value = nn.Parameter(torch.tensor(value))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return obs * 0.0 + self.value


def test_survival_masking_counts_current_transition_reward_before_done():
  learner = BpttPolicyLearner(
    _DummyActor(),
    _TwoStepEnv(),
    BpttPolicyLearnerCfg(horizon=2, gamma=1.0, survival_masking=True),
  )

  stats = learner.update([TensorDict({}, batch_size=[NUM_ENVS])])

  # Return 1 + (1 - 0.5) * 1, averaged over the horizon.
  assert stats["Policy/reward"].item() == pytest.approx(1.5 / 2)


def test_actor_critic_bootstraps_the_surviving_last_state():
  learner = ActorCriticPolicyLearner(
    _DummyActor(),
    _ConstantCritic(1.0),
    _TwoStepEnv(),
    ActorCriticPolicyLearnerCfg(horizon=2, gamma=0.99, survival_masking=True),
  )

  stats = learner.update([TensorDict({}, batch_size=[NUM_ENVS])])

  # Return 1 + 0.99 * 0.5 * 1, plus the value 1 discounted by 0.99^2 and by
  # survival 0.5, averaged over the horizon.
  assert stats["Policy/reward"].item() == pytest.approx((1.495 + 0.49005) / 2)
  assert torch.isfinite(stats["Policy/value_function"])


def test_gae_returns_discount_by_soft_done():
  returns = gae_returns(
    values=torch.zeros(2, 1),
    last_value=torch.zeros(1),
    rewards=torch.ones(2, 1),
    dones=torch.tensor([[0.5], [0.0]]),
    gamma=0.99,
    lam=1.0,
  )

  assert torch.allclose(returns, torch.tensor([[1.495], [1.0]]))


def test_actor_ema_averages_from_the_first_update():
  actor = _DummyActor()
  learner = BpttPolicyLearner(
    actor, _TwoStepEnv(), BpttPolicyLearnerCfg(horizon=2, actor_ema=0.9)
  )
  batch = [TensorDict({}, batch_size=[NUM_ENVS])]

  learner.update(batch)
  first = actor.weight.detach().clone()
  assert learner.deployed_state_dict()["weight"] == first
  with torch.no_grad():  # the dummy env gives no gradient; move the weight by hand
    actor.weight.fill_(1.0)
  learner.update(batch)
  second = actor.weight.detach().clone()
  assert second != first

  average = 0.9 * first + 0.1 * second
  torch.testing.assert_close(learner.deployed_state_dict()["weight"], average)
  with learner.deployed_weights():
    torch.testing.assert_close(actor.weight.detach(), average)
  assert actor.weight.detach() == second


def test_no_actor_ema_stores_the_live_actor():
  actor = _DummyActor()
  learner = BpttPolicyLearner(actor, _TwoStepEnv(), BpttPolicyLearnerCfg(horizon=2))
  learner.update([TensorDict({}, batch_size=[NUM_ENVS])])
  assert not learner.keeps_ema
  assert learner.deployed_state_dict()["weight"] == actor.weight.detach()


def test_critic_lr_defaults_to_the_actor_lr():
  def critic_lr(**cfg) -> float:
    learner = ActorCriticPolicyLearner(
      _DummyActor(),
      _ConstantCritic(1.0),
      _TwoStepEnv(),
      ActorCriticPolicyLearnerCfg(horizon=2, lr=6e-4, **cfg),
    )
    return float(learner.critic_optimizers[0].param_groups[0]["lr"])

  assert critic_lr() == pytest.approx(6e-4)
  assert critic_lr(critic_lr=2e-4) == pytest.approx(2e-4)
