"""PPO must solve small problems quickly, or something in the loop is broken."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlmm.ppo import PPOConfig, PPOTrainer


class ContextBandit(gym.Env[np.ndarray, np.ndarray]):
    """Eight steps per episode. Each step shows a random sign; the first action
    head scores 1 for matching it, the second scores 0.5 for always picking 2.
    Optimal return is 12, a uniform policy gets about 5.3."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 3])
        self._ctx = 0
        self._t = 0

    def _obs(self) -> np.ndarray:
        return np.array([2 * self._ctx - 1, self._t / 8], dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._t = 0
        self._ctx = int(self.np_random.integers(2))
        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        reward = float(action[0] == self._ctx) + 0.5 * float(action[1] == 2)
        self._t += 1
        self._ctx = int(self.np_random.integers(2))
        return self._obs(), reward, self._t == 8, False, {}


class Corridor(gym.Env[np.ndarray, np.int64]):
    """Walk right along 6 cells to a reward of 1; each step costs 0.01. A time limit
    of 20 steps truncates. Tests credit assignment across steps with a plain
    Discrete action space."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._pos = 0
        self._t = 0

    def _obs(self) -> np.ndarray:
        o = np.zeros(6, dtype=np.float32)
        o[self._pos] = 1.0
        return o

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._pos, self._t = 0, 0
        return self._obs(), {}

    def step(
        self, action: np.int64
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self._t += 1
        self._pos = min(5, self._pos + 1) if int(action) == 1 else max(0, self._pos - 1)
        done = self._pos == 5
        return self._obs(), (1.0 if done else 0.0) - 0.01, done, self._t >= 20 and not done, {}


def test_ppo_solves_contextual_bandit() -> None:
    cfg = PPOConfig(
        total_timesteps=24_000,
        num_envs=8,
        num_steps=64,
        num_minibatches=4,
        update_epochs=6,
        learning_rate=3e-3,
        hidden_sizes=(32,),
        ent_coef=0.0,
        seed=3,
    )
    result = PPOTrainer(ContextBandit, cfg).train()
    final = np.mean([h["episode_return"] for h in result.history[-5:]])
    assert final > 11.0, final
    first = result.history[0]["episode_return"]
    assert first < 7.0


def test_ppo_solves_corridor_with_truncation() -> None:
    cfg = PPOConfig(
        total_timesteps=16_000,
        num_envs=8,
        num_steps=64,
        num_minibatches=4,
        update_epochs=4,
        learning_rate=3e-3,
        hidden_sizes=(32,),
        ent_coef=0.0,
        value_clip=0.2,
        norm_reward=False,
        seed=0,
    )
    result = PPOTrainer(Corridor, cfg).train()
    final_len = np.mean([h["episode_length"] for h in result.history[-4:]])
    assert final_len < 6.5, final_len
