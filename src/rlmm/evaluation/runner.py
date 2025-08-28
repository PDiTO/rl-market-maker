"""Run policies over many held-out markets, in parallel."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, SupportsFloat

import numpy as np

from rlmm.env.config import EVAL_SEED_BASE, EnvConfig
from rlmm.env.market_making import EpisodeTrace, LobMarketMakingEnv, MarketView
from rlmm.evaluation.metrics import EpisodeMetrics, episode_metrics
from rlmm.ppo.agent import Agent

StepOut = tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]


class Quoter(Protocol):
    """A hand-written strategy: picks prices from the agent's delayed view."""

    name: str

    def reset(self) -> None: ...
    def quote(self, view: MarketView) -> tuple[int | None, int | None]: ...


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...
    def step(self, env: LobMarketMakingEnv, obs: np.ndarray) -> StepOut: ...


class QuoterPolicy:
    def __init__(self, quoter: Quoter) -> None:
        self.quoter = quoter
        self.name = quoter.name

    def reset(self) -> None:
        self.quoter.reset()

    def step(self, env: LobMarketMakingEnv, obs: np.ndarray) -> StepOut:
        bid, ask = self.quoter.quote(env.market_view())
        return env.step_quotes(bid, ask)


class AgentPolicy:
    """A trained PPO agent acting greedily (the mode of each action head)."""

    def __init__(self, agent: Agent, name: str, deterministic: bool = True) -> None:
        self.agent = agent
        self.name = name
        self.deterministic = deterministic

    def reset(self) -> None:
        pass

    def step(self, env: LobMarketMakingEnv, obs: np.ndarray) -> StepOut:
        return env.step(self.agent.act(obs, deterministic=self.deterministic))


def eval_seeds(n: int, offset: int = 0) -> list[int]:
    return [EVAL_SEED_BASE + offset + i for i in range(n)]


def run_episode(env: LobMarketMakingEnv, policy: Policy, feed_seed: int) -> EpisodeTrace:
    policy.reset()
    obs, _ = env.reset(seed=feed_seed, options={"feed_seed": feed_seed})
    done = False
    while not done:
        obs, _, term, trunc, _ = policy.step(env, obs)
        done = term or trunc
    return env.trace


def evaluate(policy: Policy, cfg: EnvConfig, seeds: Sequence[int]) -> list[EpisodeMetrics]:
    env = LobMarketMakingEnv(cfg, record=True)
    out = []
    for s in seeds:
        trace = run_episode(env, policy, s)
        out.append(episode_metrics(trace, s, cfg.step_ms / 1000.0, cfg.warmup_s, cfg.horizon_s))
    return out


@dataclass(frozen=True)
class Job:
    key: str
    policy: Policy
    cfg: EnvConfig
    seeds: tuple[int, ...]


def _run_job(job: Job) -> tuple[str, list[EpisodeMetrics]]:
    import torch

    torch.set_num_threads(1)
    return job.key, evaluate(job.policy, job.cfg, job.seeds)


def run_jobs(
    jobs: Sequence[Job], workers: int | None = None, chunk: int = 50
) -> dict[str, list[EpisodeMetrics]]:
    """Runs every job, splitting seeds into chunks across processes. Results per key
    come back in seed order."""
    pieces: list[Job] = []
    for j in jobs:
        for i in range(0, len(j.seeds), chunk):
            pieces.append(Job(j.key, j.policy, j.cfg, j.seeds[i : i + chunk]))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    results: dict[str, list[EpisodeMetrics]] = {j.key: [] for j in jobs}
    if workers == 1:
        for p in pieces:
            k, m = _run_job(p)
            results[k].extend(m)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, m in pool.map(_run_job, pieces):
                results[k].extend(m)
    for k in results:
        results[k].sort(key=lambda m: m.feed_seed)
    return results
