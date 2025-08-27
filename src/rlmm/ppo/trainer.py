"""Proximal policy optimisation, written out in full.

The loop follows the standard recipe (Schulman et al., 2017, with the
implementation details catalogued by Huang et al., 2022): vectorised rollouts,
GAE(lambda), clipped surrogate with optional value clipping, entropy bonus,
per-minibatch advantage normalisation, running observation and reward
normalisation, linear learning-rate annealing and global gradient clipping.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rlmm.ppo.agent import Agent
from rlmm.ppo.gae import compute_gae
from rlmm.ppo.losses import clipped_policy_loss, explained_variance, value_loss
from rlmm.ppo.normalize import RewardNormalizer
from rlmm.ppo.vec_env import SyncVectorEnv

Logger = Callable[[dict[str, float]], None]
EvalFn = Callable[[Agent], dict[str, float]]


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 5_000_000
    num_envs: int = 16
    num_steps: int = 256
    """Rollout length per env; the batch is ``num_envs * num_steps``."""
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 10
    clip_coef: float = 0.2
    value_clip: float | None = None
    """Clip range for value updates, in normalised-return units. None disables it."""
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    """Stop the epoch loop early once approx-KL passes this."""
    norm_adv: bool = True
    norm_obs: bool = True
    norm_reward: bool = True
    obs_clip: float = 10.0
    reward_clip: float = 10.0
    hidden_sizes: tuple[int, ...] = (128, 128)
    seed: int = 1
    device: str = "cpu"
    torch_threads: int = 1
    eval_every: int = 10
    """Updates between calls to the evaluation hook."""

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    agent: Agent
    history: list[dict[str, float]] = field(default_factory=list)
    wall_clock_s: float = 0.0


class PPOTrainer:
    def __init__(
        self,
        env_fn: Callable[[], gym.Env[Any, Any]],
        cfg: PPOConfig,
        logger: Logger | None = None,
        evaluate: EvalFn | None = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.evaluate = evaluate
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(cfg.torch_threads)
        self.rng = np.random.default_rng(cfg.seed)
        self.envs = SyncVectorEnv([env_fn for _ in range(cfg.num_envs)])
        space = self.envs.single_observation_space
        assert space.shape is not None
        self.obs_dim = int(np.prod(space.shape))
        self.agent = Agent(
            self.obs_dim,
            self.envs.nvec,
            cfg.hidden_sizes,
            obs_clip=cfg.obs_clip,
            normalize_obs=cfg.norm_obs,
            device=cfg.device,
        )
        self.optimizer = torch.optim.Adam(
            self.agent.policy.parameters(), lr=cfg.learning_rate, eps=1e-5
        )
        self.reward_norm = (
            RewardNormalizer(cfg.num_envs, cfg.gamma, clip=cfg.reward_clip)
            if cfg.norm_reward
            else None
        )

    def train(self) -> TrainResult:
        cfg, agent, envs = self.cfg, self.agent, self.envs
        policy = agent.policy
        T, N, A = cfg.num_steps, cfg.num_envs, len(envs.nvec)

        obs_buf = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((T, N, A), dtype=np.int64)
        logp_buf = np.zeros((T, N), dtype=np.float32)
        val_buf = np.zeros((T, N), dtype=np.float32)
        next_val_buf = np.zeros((T, N), dtype=np.float32)
        rew_buf = np.zeros((T, N), dtype=np.float32)
        term_buf = np.zeros((T, N), dtype=bool)
        end_buf = np.zeros((T, N), dtype=bool)

        raw_obs = envs.reset(seed=cfg.seed)
        ep_return = np.zeros(N)
        ep_length = np.zeros(N, dtype=np.int64)
        history: list[dict[str, float]] = []
        global_step = 0
        start = time.perf_counter()

        for update in range(1, cfg.num_updates + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / cfg.num_updates
                self.optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            episode_stats: dict[str, list[float]] = defaultdict(list)
            policy.eval()
            obs_t = agent.preprocess(raw_obs, update=True)
            for t in range(T):
                with torch.no_grad():
                    action, logp, value = policy.act(obs_t)
                obs_buf[t] = obs_t.cpu().numpy()
                act_buf[t] = action.cpu().numpy()
                logp_buf[t] = logp.cpu().numpy()
                val_buf[t] = value.cpu().numpy()

                step = envs.step(act_buf[t])
                global_step += N
                ended = step.terminated | step.truncated
                ep_return += step.rewards
                ep_length += 1
                for i in np.flatnonzero(ended):
                    episode_stats["episode_return"].append(float(ep_return[i]))
                    episode_stats["episode_length"].append(float(ep_length[i]))
                    for k, v in step.infos[i].get("episode_stats", {}).items():
                        episode_stats[f"episode_{k}"].append(float(v))
                ep_return[ended] = 0.0
                ep_length[ended] = 0

                rew_buf[t] = (
                    self.reward_norm(step.rewards, ended)
                    if self.reward_norm is not None
                    else step.rewards
                )
                term_buf[t] = step.terminated
                end_buf[t] = ended

                raw_obs = step.obs
                obs_t = agent.preprocess(raw_obs, update=True)
                with torch.no_grad():
                    next_v = policy.value(obs_t).cpu().numpy()
                    trunc_idx = np.flatnonzero(step.truncated)
                    if trunc_idx.size:
                        final = agent.preprocess(step.final_obs[trunc_idx])
                        next_v[trunc_idx] = policy.value(final).cpu().numpy()
                next_val_buf[t] = next_v

            adv, returns = compute_gae(
                rew_buf, val_buf, next_val_buf, term_buf, end_buf, cfg.gamma, cfg.gae_lambda
            )
            stats = self._update(obs_buf, act_buf, logp_buf, val_buf, adv, returns)

            elapsed = time.perf_counter() - start
            row: dict[str, float] = {
                "update": float(update),
                "global_step": float(global_step),
                "wall_clock_s": elapsed,
                "sps": global_step / elapsed,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                **stats,
            }
            for k, v in episode_stats.items():
                row[k] = float(np.mean(v))
            if self.evaluate is not None and (
                update % cfg.eval_every == 0 or update == cfg.num_updates
            ):
                row.update(self.evaluate(agent))
            history.append(row)
            if self.logger is not None:
                self.logger(row)

        return TrainResult(agent, history, time.perf_counter() - start)

    def _update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logp: np.ndarray,
        values: np.ndarray,
        adv: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        cfg, policy, device = self.cfg, self.agent.policy, self.agent.device
        policy.train()

        def flat(x: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(x.reshape(cfg.batch_size, *x.shape[2:]), dtype=dtype).to(device)

        b_obs = flat(obs, torch.float32)
        b_act = flat(actions, torch.int64)
        b_logp = flat(logp, torch.float32)
        b_val = flat(values, torch.float32)
        b_adv = flat(adv, torch.float32)
        b_ret = flat(returns, torch.float32)

        sums: dict[str, list[float]] = defaultdict(list)
        idx = np.arange(cfg.batch_size)
        epochs_run = 0
        for _ in range(cfg.update_epochs):
            self.rng.shuffle(idx)
            epoch_kl: list[float] = []
            for s in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = torch.as_tensor(idx[s : s + cfg.minibatch_size], device=device)
                new_logp, entropy, new_v = policy.evaluate(b_obs[mb], b_act[mb])
                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss, pstats = clipped_policy_loss(new_logp, b_logp[mb], mb_adv, cfg.clip_coef)
                v_loss = value_loss(new_v, b_val[mb], b_ret[mb], cfg.value_clip)
                ent = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                sums["policy_loss"].append(pg_loss.item())
                sums["value_loss"].append(v_loss.item())
                sums["entropy"].append(ent.item())
                sums["approx_kl"].append(pstats.approx_kl)
                sums["clip_fraction"].append(pstats.clip_fraction)
                sums["grad_norm"].append(float(grad_norm))
                epoch_kl.append(pstats.approx_kl)
            epochs_run += 1
            if cfg.target_kl is not None and np.mean(epoch_kl) > cfg.target_kl:
                break

        with torch.no_grad():
            ev = explained_variance(b_val, b_ret)
        out = {k: float(np.mean(v)) for k, v in sums.items()}
        out["explained_variance"] = ev
        out["epochs"] = float(epochs_run)
        return out
