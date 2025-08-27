"""Actor-critic network with a factorised categorical policy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def _layer(n_in: int, n_out: int, gain: float) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(n_in: int, hidden: Sequence[int], n_out: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [_layer(n_in, h, float(np.sqrt(2.0))), nn.Tanh()]
        n_in = h
    layers.append(_layer(n_in, n_out, out_gain))
    return nn.Sequential(*layers)


class MultiCategorical:
    """Independent categoricals, one per action dimension. The joint log-probability
    and entropy are sums over dimensions."""

    def __init__(self, logits: torch.Tensor, nvec: Sequence[int]) -> None:
        self.dists = [Categorical(logits=chunk) for chunk in torch.split(logits, list(nvec), -1)]

    def sample(self) -> torch.Tensor:
        return torch.stack([d.sample() for d in self.dists], dim=-1)

    def mode(self) -> torch.Tensor:
        return torch.stack([d.probs.argmax(-1) for d in self.dists], dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [d.log_prob(actions[..., i]) for i, d in enumerate(self.dists)], dim=-1
        ).sum(-1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(-1)


class ActorCritic(nn.Module):
    """Separate policy and value MLPs (no shared trunk, so the value loss scale does
    not interfere with the policy features). The policy head is initialised with a
    small gain so the initial policy is close to uniform."""

    def __init__(self, obs_dim: int, nvec: Sequence[int], hidden: Sequence[int]) -> None:
        super().__init__()
        self.nvec = list(nvec)
        self.actor = _mlp(obs_dim, hidden, sum(self.nvec), out_gain=0.01)
        self.critic = _mlp(obs_dim, hidden, 1, out_gain=1.0)

    def dist(self, obs: torch.Tensor) -> MultiCategorical:
        return MultiCategorical(self.actor(obs), self.nvec)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        v: torch.Tensor = self.critic(obs)
        return v.squeeze(-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(actions, log_probs, values)``; actions have shape ``(B, len(nvec))``."""
        d = self.dist(obs)
        a = d.mode() if deterministic else d.sample()
        return a, d.log_prob(a), self.value(obs)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(log_probs, entropy, values)`` for given actions."""
        d = self.dist(obs)
        return d.log_prob(actions), d.entropy(), self.value(obs)
