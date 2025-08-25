"""Generalised advantage estimation."""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_end: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) over a ``(T, N)`` rollout.

    ``next_values[t]`` is ``V(s_{t+1})``, the value of the state reached by step
    ``t``. When step ``t`` ended an episode that is the value of the final
    observation, not of the reset observation that replaced it.

    Termination and truncation are different. On termination the next state has no
    value, so ``delta_t = r_t - V(s_t)``. On truncation (a time limit the state does
    not know about) the episode could have continued, so the TD error still
    bootstraps from ``V(s_{t+1})``. Either way the advantage does not carry across
    the boundary into the next episode:

        delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma * lam * (1 - episode_end_t) * A_{t+1}

    Returns ``(advantages, returns)`` with ``returns = advantages + values``.
    """
    T = rewards.shape[0]
    not_term = 1.0 - terminated.astype(np.float64)
    not_end = 1.0 - episode_end.astype(np.float64)
    adv = np.zeros(rewards.shape, dtype=np.float64)
    last = np.zeros(rewards.shape[1:], dtype=np.float64)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * not_term[t] * next_values[t] - values[t]
        last = delta + gamma * lam * not_end[t] * last
        adv[t] = last
    return adv, adv + values
