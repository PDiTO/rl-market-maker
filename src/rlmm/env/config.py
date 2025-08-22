"""Configuration for the market-making environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rlmm.sim.engine import SimConfig

# Feed seeds live in disjoint ranges so training never sees an evaluation market.
TRAIN_SEED_SPACE = 1 << 40
CALIBRATION_SEED_BASE = 1 << 41
EVAL_SEED_BASE = 1 << 42


@dataclass(frozen=True)
class MarketConfig:
    """The ``lob`` synthetic market (its ``[market]`` table), minus seed and duration."""

    initial_price: int = 10_000
    fair_value: str = "random_walk"
    volatility: float = 2.0
    """Fair value volatility, ticks per sqrt(second)."""
    mean_reversion: float = 0.05
    limit_rate: float = 80.0
    market_rate: float = 10.0
    cancel_rate: float = 0.25
    mean_offset: float = 5.0
    max_qty: int = 10
    max_market_qty: int = 12
    informed: float = 0.5
    participants: int = 500
    initial_levels: int = 10

    def to_lob(self, seed: int, duration_s: float) -> dict[str, Any]:
        d = asdict(self)
        d["seed"] = seed
        d["duration_s"] = duration_s
        return d


@dataclass(frozen=True)
class RewardConfig:
    """Per-step reward: change in mark-to-market PnL, less penalties.

    ``r_t = dPnL_t - phi * q_t^2 * dt - eta * max(0, q_{t-1} * dmid_t) - liquidation``
    """

    inventory_penalty: float = 0.005
    """phi, in ticks per lot^2 per second. A running quadratic inventory cost."""
    dampening: float = 0.0
    """eta in [0, 1]. Asymmetric dampening: removes that share of any gain that came
    from price moves on inventory, but none of the losses (Spooner et al., 2018)."""
    terminal_liquidation: bool = True
    """Charge the cost of flattening the final position into the book at the end."""


@dataclass(frozen=True)
class EnvConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    step_ms: float = 100.0
    """Decision interval."""
    horizon_s: float = 60.0
    """Session length after warm-up. Episodes are exactly horizon / step decisions."""
    warmup_s: float = 5.0
    """Market time before the first decision, so the book starts in steady state."""
    quote_size: int = 5
    max_position: int = 50
    """Hard limit. A side that would take the known position past it is not quoted."""

    @property
    def step_ns(self) -> int:
        return round(self.step_ms * 1e6)

    @property
    def warmup_ns(self) -> int:
        return round(self.warmup_s * 1e9)

    @property
    def n_steps(self) -> int:
        return round(self.horizon_s * 1000.0 / self.step_ms)

    def replace(self, **changes: Any) -> EnvConfig:
        """A copy with top-level fields, or nested ``market``/``sim``/``reward``
        dicts, replaced."""
        d: dict[str, Any] = {
            "market": self.market,
            "sim": self.sim,
            "reward": self.reward,
            "step_ms": self.step_ms,
            "horizon_s": self.horizon_s,
            "warmup_s": self.warmup_s,
            "quote_size": self.quote_size,
            "max_position": self.max_position,
        }
        for k, v in changes.items():
            if k not in d:
                raise KeyError(f"unknown env setting {k!r}")
            if k in ("market", "sim", "reward") and isinstance(v, dict):
                cur = asdict(d[k])
                unknown = set(v) - set(cur)
                if unknown:
                    raise KeyError(f"unknown {k} settings: {sorted(unknown)}")
                cur.update(v)
                v = type(d[k])(**cur)
            d[k] = v
        return EnvConfig(**d)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EnvConfig:
        return EnvConfig().replace(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
