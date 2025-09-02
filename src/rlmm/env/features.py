"""Observation vector for the market-making environment.

Everything here comes from the agent's delayed view: the book as of
``market_data_latency_ns`` ago, its position and orders from private messages that
have arrived, and the queue position of its resting orders as a feed-watching system
with L3 data would know it at that time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rlmm.sim.engine import OwnQueue, Snapshot
from rlmm.sim.orders import AgentCtx

# Features that say something about where the price is going next, as opposed to
# the agent's own state. The ablation run hides these.
DIRECTIONAL = (
    "imbalance_l1",
    "imbalance_l5",
    "micro_minus_mid",
    "ret_1",
    "ret_10",
    "ret_50",
    "trade_imbalance",
)

# name, low, high
FEATURES: tuple[tuple[str, float, float], ...] = (
    ("inventory", -2.0, 2.0),  # known position / max_position
    ("time_left", 0.0, 1.0),  # share of the session remaining
    ("spread", 0.0, 1.0),  # min(spread, 10) / 10, in ticks
    ("imbalance_l1", -1.0, 1.0),  # (bid qty - ask qty) / sum, top level
    ("imbalance_l5", -1.0, 1.0),  # same over five levels
    ("micro_minus_mid", -2.0, 2.0),  # microprice - mid, ticks
    ("ret_1", -10.0, 10.0),  # mid change over the last step, ticks
    ("ret_10", -20.0, 20.0),  # over the last 10 steps
    ("ret_50", -40.0, 40.0),  # over the last 50 steps
    ("volatility", 0.0, 10.0),  # EWMA std of one-step mid changes, ticks
    ("trade_imbalance", -1.0, 1.0),  # (bought - sold) / total, aggressor volume last step
    ("trade_volume", 0.0, 10.0),  # log1p(aggressor volume last step)
    ("bid_live", 0.0, 1.0),  # we have a bid resting
    ("bid_depth", 0.0, 10.0),  # ticks behind the best bid
    ("bid_ahead", 0.0, 10.0),  # log1p(lots ahead of us in the queue)
    ("bid_ahead_frac", 0.0, 1.0),  # lots ahead / level size
    ("bid_qty", 0.0, 1.0),  # open qty / quote size
    ("ask_live", 0.0, 1.0),
    ("ask_depth", 0.0, 10.0),
    ("ask_ahead", 0.0, 10.0),
    ("ask_ahead_frac", 0.0, 1.0),
    ("ask_qty", 0.0, 1.0),
    ("bid_filled", 0.0, 4.0),  # lots bought last step / quote size
    ("ask_filled", 0.0, 4.0),  # lots sold last step / quote size
)

FEATURE_NAMES = tuple(f[0] for f in FEATURES)
OBS_LOW = np.array([f[1] for f in FEATURES], dtype=np.float32)
OBS_HIGH = np.array([f[2] for f in FEATURES], dtype=np.float32)
OBS_DIM = len(FEATURES)
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

_VOL_ALPHA = 1.0 - 0.5 ** (1.0 / 20.0)  # 20-step half-life


def _imbalance(b: float, a: float) -> float:
    return (b - a) / (b + a) if b + a > 0 else 0.0


@dataclass(slots=True)
class _Side:
    live: float = 0.0
    depth: float = 0.0
    ahead: float = 0.0
    ahead_frac: float = 0.0
    qty: float = 0.0


class FeatureBuilder:
    """Keeps the short history the observation needs and builds the vector."""

    def __init__(self, quote_size: int, max_position: int, hide: tuple[str, ...] = ()) -> None:
        unknown = set(hide) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")
        self.quote_size = quote_size
        self.max_position = max_position
        self._hide = [_IDX[h] for h in hide]
        self.mids: list[float] = []
        self.var = 0.0

    def reset(self) -> None:
        self.mids.clear()
        self.var = 0.0

    def _record_mid(self, mid: float) -> None:
        if self.mids:
            d = mid - self.mids[-1]
            self.var += _VOL_ALPHA * (d * d - self.var)
        self.mids.append(mid)

    def _ret(self, k: int) -> float:
        m = self.mids
        return m[-1] - m[max(0, len(m) - 1 - k)]

    def _side(self, orders: list[OwnQueue], best: int | None, sign: int) -> _Side:
        s = _Side()
        if not orders:
            return s
        o = max(orders, key=lambda x: x.id)  # the newest is the one we are managing
        s.live = 1.0
        if best is not None:
            s.depth = float(max(0, sign * (best - o.price)))
        s.ahead = math.log1p(o.ahead)
        s.ahead_frac = o.ahead / o.level_qty if o.level_qty > 0 else 0.0
        s.qty = o.qty / self.quote_size
        return s

    def build(
        self,
        snap: Snapshot,
        ctx: AgentCtx,
        time_left: float,
        bought: int,
        sold: int,
    ) -> np.ndarray:
        bb, ba = snap.best_bid(), snap.best_ask()
        mid = snap.mid()
        if mid is None:
            mid = self.mids[-1] if self.mids else 0.0
        self._record_mid(mid)

        x = np.zeros(OBS_DIM, dtype=np.float64)
        x[_IDX["inventory"]] = ctx.position / self.max_position
        x[_IDX["time_left"]] = time_left
        if bb is not None and ba is not None:
            x[_IDX["spread"]] = min(ba[0] - bb[0], 10) / 10.0
            x[_IDX["imbalance_l1"]] = _imbalance(bb[1], ba[1])
            micro = snap.microprice()
            assert micro is not None
            x[_IDX["micro_minus_mid"]] = micro - mid
        x[_IDX["imbalance_l5"]] = _imbalance(
            sum(q for _, q, _ in snap.bids), sum(q for _, q, _ in snap.asks)
        )
        x[_IDX["ret_1"]] = self._ret(1)
        x[_IDX["ret_10"]] = self._ret(10)
        x[_IDX["ret_50"]] = self._ret(50)
        x[_IDX["volatility"]] = math.sqrt(self.var)
        vol = snap.buy_volume + snap.sell_volume
        x[_IDX["trade_imbalance"]] = _imbalance(snap.buy_volume, snap.sell_volume)
        x[_IDX["trade_volume"]] = math.log1p(vol)

        # Queue features describe the order we are managing on each side, not ones
        # we have already asked to cancel.
        cancelling = {o.id for o in ctx.open_order_list() if o.pending == "cancel"}
        bids = [o for o in snap.own if o.side == "bid" and o.id not in cancelling]
        asks = [o for o in snap.own if o.side == "ask" and o.id not in cancelling]
        for prefix, side in (
            ("bid", self._side(bids, bb[0] if bb else None, 1)),
            ("ask", self._side(asks, ba[0] if ba else None, -1)),
        ):
            x[_IDX[f"{prefix}_live"]] = side.live
            x[_IDX[f"{prefix}_depth"]] = side.depth
            x[_IDX[f"{prefix}_ahead"]] = side.ahead
            x[_IDX[f"{prefix}_ahead_frac"]] = side.ahead_frac
            x[_IDX[f"{prefix}_qty"]] = side.qty
        x[_IDX["bid_filled"]] = bought / self.quote_size
        x[_IDX["ask_filled"]] = sold / self.quote_size
        x[self._hide] = 0.0
        return np.clip(x, OBS_LOW, OBS_HIGH).astype(np.float32)
