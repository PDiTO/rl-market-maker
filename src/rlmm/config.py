"""TOML run configuration: environment, PPO and training settings."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from rlmm.env.config import EnvConfig
from rlmm.ppo.trainer import PPOConfig


@dataclass(frozen=True)
class TrainSettings:
    seeds: tuple[int, ...] = (1, 2, 3)
    out_dir: str = "runs/ppo"
    val_episodes: int = 16
    """Validation markets scored during training (calibration range, never eval)."""
    checkpoint_every: int = 50
    """Updates between checkpoints."""


@dataclass(frozen=True)
class EvalSettings:
    episodes: int = 500
    robustness_episodes: int = 150
    workers: int = 0
    """0 means one fewer than the number of CPUs."""
    policies: dict[str, str] = field(default_factory=dict)
    """Name to checkpoint path for PPO agents to evaluate."""
    calibration: str = "results/calibration.json"
    out_dir: str = "results"
    latency_ns: tuple[int, ...] = (0, 50_000, 1_000_000, 5_000_000, 20_000_000)
    volatility: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    informed: tuple[float, ...] = (0.3, 0.5, 0.7)
    mean_reversion: tuple[float, ...] = (0.5,)
    """Speeds for an extra regime where fair value is mean-reverting, per second."""


@dataclass(frozen=True)
class RunConfig:
    env: EnvConfig
    ppo: PPOConfig
    train: TrainSettings
    eval: EvalSettings
    path: Path | None = None


def _build(cls: Any, d: dict[str, Any], where: str) -> Any:
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown keys in [{where}]: {sorted(unknown)}")
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
    return cls(**kwargs)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - {"env", "ppo", "train", "eval"}
    if unknown:
        raise KeyError(f"unknown sections: {sorted(unknown)}")
    env = EnvConfig.from_dict(raw.get("env", {}))
    ppo = _build(PPOConfig, raw.get("ppo", {}), "ppo")
    train = _build(TrainSettings, raw.get("train", {}), "train")
    ev = dict(raw.get("eval", {}))
    policies = ev.pop("policies", {})
    evs = _build(EvalSettings, ev, "eval")
    evs = EvalSettings(**{**evs.__dict__, "policies": dict(policies)})
    return RunConfig(env, ppo, train, evs, path)
