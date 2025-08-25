"""Running statistics for observation and reward normalisation."""

from __future__ import annotations

from typing import Any

import numpy as np


class RunningMeanStd:
    """Mean and variance of a stream of batches, merged with Chan et al.'s parallel
    update so the result equals the statistics of everything seen so far."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """``x`` has shape ``(batch, *shape)``."""
        x = np.asarray(x, dtype=np.float64)
        self.update_from_moments(x.mean(axis=0), x.var(axis=0), x.shape[0])

    def update_from_moments(self, mean: np.ndarray, var: np.ndarray, count: int) -> None:
        delta = mean - self.mean
        total = self.count + count
        new_mean = self.mean + delta * count / total
        m2 = self.var * self.count + var * count + delta**2 * self.count * count / total
        self.mean, self.var, self.count = new_mean, m2 / total, total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


class ObsNormalizer:
    """``clip((obs - mean) / sqrt(var + eps), -clip, clip)`` with running stats."""

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, eps: float = 1e-8) -> None:
        self.rms = RunningMeanStd(shape)
        self.clip = clip
        self.eps = eps

    def __call__(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.rms.update(obs)
        out = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.eps)
        return np.asarray(np.clip(out, -self.clip, self.clip), dtype=np.float32)


class RewardNormalizer:
    """Scales rewards by the running std of the discounted return.

    The mean is not subtracted: shifting rewards would change the optimal policy
    when episodes can end early, scaling does not.
    """

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, eps: float = 1e-8) -> None:
        self.rms = RunningMeanStd(())
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma
        self.clip = clip
        self.eps = eps

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """``dones`` marks steps that ended an episode (terminated or truncated)."""
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        out = rewards / np.sqrt(self.rms.var + self.eps)
        self.returns[dones] = 0.0
        return np.asarray(np.clip(out, -self.clip, self.clip), dtype=np.float32)
