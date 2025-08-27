"""A small synchronous vector env with same-step autoreset.

Gymnasium's vector envs (1.x) reset on the step *after* an episode ends, which
leaves a dummy transition in the rollout. Here an env that ends is reset inside
the same ``step`` call, and its final observation is returned separately so the
learner can bootstrap from it on truncation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class VecStep:
    obs: np.ndarray
    """Next observations, already reset for envs whose episode ended."""
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_obs: np.ndarray
    """Like ``obs`` but holding the last observation of any episode that ended."""
    infos: list[dict[str, Any]]


def action_nvec(space: gym.Space[Any]) -> list[int]:
    if isinstance(space, spaces.Discrete):
        return [int(space.n)]
    if isinstance(space, spaces.MultiDiscrete):
        return [int(n) for n in space.nvec.ravel()]
    raise TypeError(f"unsupported action space {space}")


class SyncVectorEnv:
    def __init__(self, env_fns: Sequence[Callable[[], gym.Env[Any, Any]]]) -> None:
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.nvec = action_nvec(self.single_action_space)
        self._discrete = isinstance(self.single_action_space, spaces.Discrete)

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = [
            env.reset(seed=None if seed is None else seed + i)[0] for i, env in enumerate(self.envs)
        ]
        return np.stack(obs).astype(np.float32)

    def step(self, actions: np.ndarray) -> VecStep:
        n = self.num_envs
        obs_list: list[np.ndarray] = []
        final_list: list[np.ndarray] = []
        rewards = np.zeros(n, dtype=np.float64)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        infos: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            a = int(actions[i, 0]) if self._discrete else actions[i]
            o, r, term, trunc, info = env.step(a)
            rewards[i] = float(r)
            terminated[i], truncated[i] = term, trunc
            final_list.append(o)
            if term or trunc:
                o, _ = env.reset()
            obs_list.append(o)
            infos.append(info)
        return VecStep(
            obs=np.stack(obs_list).astype(np.float32),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            final_obs=np.stack(final_list).astype(np.float32),
            infos=infos,
        )

    def close(self) -> None:
        for env in self.envs:
            cast(Callable[[], None], env.close)()
