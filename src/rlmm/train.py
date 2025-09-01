"""Training entry point: one PPO run per seed, with validation and checkpoints."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from rlmm.baselines.calibration import calibration_seeds
from rlmm.config import RunConfig
from rlmm.env import EnvConfig, LobMarketMakingEnv
from rlmm.evaluation.runner import AgentPolicy, evaluate
from rlmm.ppo import Agent, PPOTrainer
from rlmm.ppo.trainer import EvalFn

VALIDATION_OFFSET = 100_000

COLUMNS = (
    "update",
    "global_step",
    "wall_clock_s",
    "sps",
    "learning_rate",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
    "explained_variance",
    "epochs",
    "episode_return",
    "episode_length",
    "episode_pnl",
    "episode_fills",
    "episode_mean_abs_inventory",
    "episode_final_position",
    "val_pnl",
    "val_reward",
    "val_mean_abs_inventory",
    "val_fills",
)


def validation(env_cfg: EnvConfig, episodes: int) -> EvalFn:
    seeds = calibration_seeds(episodes, offset=VALIDATION_OFFSET)

    def run(agent: Agent) -> dict[str, float]:
        ms = evaluate(AgentPolicy(agent, "ppo"), env_cfg, seeds)
        return {
            "val_pnl": float(np.mean([m.pnl for m in ms])),
            "val_reward": float(np.mean([m.reward for m in ms])),
            "val_mean_abs_inventory": float(np.mean([m.mean_abs_inventory for m in ms])),
            "val_fills": float(np.mean([m.fills for m in ms])),
        }

    return run


class _CsvLog:
    def __init__(self, f: TextIO) -> None:
        self.f = f
        self.w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", restval="")
        self.w.writeheader()

    def write(self, row: dict[str, float]) -> None:
        self.w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        self.f.flush()


def train_seed(run: RunConfig, seed: int, out_root: str | Path | None = None) -> Path:
    """Trains one seed. Writes ``metrics.csv``, ``config.json`` and ``final.pt``."""
    out = Path(out_root or run.train.out_dir) / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    ppo = replace(run.ppo, seed=seed)
    env_cfg = run.env
    meta: dict[str, Any] = {
        "seed": seed,
        "env": env_cfg.to_dict(),
        "ppo": ppo.to_dict(),
        "config_path": str(run.path) if run.path else None,
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2))

    def make_env() -> LobMarketMakingEnv:
        return LobMarketMakingEnv(env_cfg)

    with (out / "metrics.csv").open("w", newline="") as f:
        log = _CsvLog(f)
        trainer: PPOTrainer | None = None
        started = time.time()

        def on_update(row: dict[str, float]) -> None:
            log.write(row)
            u = int(row["update"])
            if trainer is not None and u % run.train.checkpoint_every == 0:
                trainer.agent.save(out / "checkpoints" / f"update{u:05d}.pt", meta)
            parts = [
                f"seed {seed}",
                f"update {u}/{ppo.num_updates}",
                f"step {int(row['global_step'])}",
                f"sps {row['sps']:.0f}",
            ]
            for k in ("episode_pnl", "episode_return", "val_pnl", "approx_kl", "entropy"):
                if k in row:
                    parts.append(f"{k} {row[k]:.3g}")
            print(" | ".join(parts), flush=True)

        trainer = PPOTrainer(
            make_env,
            ppo,
            logger=on_update,
            evaluate=validation(env_cfg, run.train.val_episodes),
        )
        result = trainer.train()
    meta["wall_clock_s"] = result.wall_clock_s
    meta["global_steps"] = ppo.num_updates * ppo.batch_size
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    meta["started"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))
    result.agent.save(out / "final.pt", meta)
    (out / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


def summarize_config(run: RunConfig) -> dict[str, Any]:
    return {"env": run.env.to_dict(), "ppo": asdict(run.ppo), "train": asdict(run.train)}
