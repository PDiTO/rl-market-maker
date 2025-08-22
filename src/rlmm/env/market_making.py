"""``LobMarketMakingEnv``: quote a limit order book with latency and queue position.

Each step the agent picks where to quote on each side, relative to the touch it can
see. The simulator then runs one decision interval of order flow through the real
matching engine. The agent's orders join the back of their queue, reach the exchange
after order latency, and the agent hears about fills after market-data latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, SupportsFloat

import gymnasium as gym
import lob
import numpy as np
from gymnasium import spaces

from rlmm.env.config import TRAIN_SEED_SPACE, EnvConfig
from rlmm.env.features import OBS_HIGH, OBS_LOW, FeatureBuilder
from rlmm.sim.accounting import money_to_ticks
from rlmm.sim.engine import Fill, ReplayEngine, Snapshot
from rlmm.sim.orders import AgentCtx, sync_quotes

LEVELS = ("off", "improve", "join", "back1", "back2")
"""Per-side action: no quote, one tick inside the touch (if the spread allows),
at the touch, or one or two ticks behind it."""


@dataclass(frozen=True, slots=True)
class MarketView:
    """What a hand-written strategy gets to see: the same delayed view as the agent."""

    t_s: float
    time_left_s: float
    best_bid: tuple[int, int] | None
    best_ask: tuple[int, int] | None
    mid: float | None
    microprice: float | None
    position: int
    quote_size: int
    max_position: int


@dataclass
class EpisodeTrace:
    """Per-step series for evaluation. Recorded at the exchange, not the agent."""

    t_s: list[float] = field(default_factory=list)
    mid: list[float] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    pnl: list[float] = field(default_factory=list)
    """Mark-to-market PnL at mid, ticks x lots."""
    reward: list[float] = field(default_factory=list)
    bid_level: list[int] = field(default_factory=list)
    ask_level: list[int] = field(default_factory=list)
    liquidation_cost: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    mid_ts: list[int] = field(default_factory=list)
    mid_px: list[float] = field(default_factory=list)
    submitted_qty: int = 0
    fees: float = 0.0


def level_price(side: str, level: int, bid: int, ask: int) -> int | None:
    """Price for a per-side ``LEVELS`` index given the visible touch."""
    if level == 0:
        return None
    sign = 1 if side == "bid" else -1
    best = bid if side == "bid" else ask
    if level == 1:
        return best + sign if ask - bid > 1 else best
    return best - sign * (level - 2)


def liquidation_cost(book: lob.OrderBook, position: int, taker_fee: float) -> float:
    """Cost against mid of flattening ``position`` with a market order, walking the
    visible book, plus the taker fee. Lots beyond the visible book are charged at
    one tick worse than the last level."""
    if position == 0:
        return 0.0
    mid = book.mid()
    l2 = book.l2(1_000)
    levels = l2["bids"] if position > 0 else l2["asks"]
    if mid is None:
        return 0.0
    remaining = abs(position)
    cost = 0.0
    last_px = mid
    for price, qty, _ in levels:
        take = min(remaining, qty)
        cost += take * abs(price - mid)
        remaining -= take
        last_px = price
        if remaining == 0:
            break
    if remaining > 0:
        cost += remaining * (abs(last_px - mid) + 1.0)
    return cost + abs(position) * taker_fee


class LobMarketMakingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-instrument market making on the ``lob`` synthetic market.

    Observation: see :mod:`rlmm.env.features`. Action: ``MultiDiscrete([5, 5])``, one
    :data:`LEVELS` index for the bid and one for the ask, at a fixed quote size.
    Episodes are exactly ``horizon_s / step_ms`` decisions and end in termination
    (time remaining is part of the state), with the final inventory liquidated.

    ``reset(options={"feed_seed": n})`` pins the market; otherwise the feed seed is
    drawn from the environment's RNG, so everything is deterministic given
    ``reset(seed=...)``.
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(self, config: EnvConfig | None = None, record: bool = False) -> None:
        self.cfg = config or EnvConfig()
        if self.cfg.sim.market_data_latency_ns >= self.cfg.step_ns:
            raise ValueError("market-data latency must be shorter than the step")
        self.record = record
        self.action_space = spaces.MultiDiscrete([len(LEVELS), len(LEVELS)])
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
        self._features = FeatureBuilder(self.cfg.quote_size, self.cfg.max_position)
        self._engine: ReplayEngine | None = None
        self._ctx = AgentCtx()
        self._k = 0
        self._pnl = 0.0
        self._mid = 0.0
        self._position = 0
        self._ep_reward = 0.0
        self._n_fills = 0
        self._abs_inv_sum = 0.0
        self.feed_seed = 0
        self.trace = EpisodeTrace()

    # -- gym API --------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.cfg
        if options and "feed_seed" in options:
            self.feed_seed = int(options["feed_seed"])
        else:
            self.feed_seed = int(self.np_random.integers(0, TRAIN_SEED_SPACE))
        # One spare second of flow so the last decision interval is always covered.
        duration = cfg.warmup_s + cfg.horizon_s + 1.0
        feed = lob.generate_feed(**cfg.market.to_lob(self.feed_seed, duration))
        self._engine = ReplayEngine(feed, cfg.sim)
        self._ctx = AgentCtx()
        self._features.reset()
        self._k = 0
        self._ep_reward = 0.0
        self._n_fills = 0
        self._abs_inv_sum = 0.0
        self.trace = EpisodeTrace()

        self._engine.schedule_timer(cfg.warmup_ns)
        snap = self._advance()
        self._mid = self._exchange_mid()
        self._pnl = 0.0
        self._position = 0
        obs = self._observe(snap, 0, 0)
        return obs, {"feed_seed": self.feed_seed}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        bid_level, ask_level = int(action[0]), int(action[1])
        bid, ask = self.action_to_prices(bid_level, ask_level)
        return self._step(bid, ask, bid_level, ask_level)

    def step_quotes(
        self, bid: int | None, ask: int | None
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps with explicit quote prices instead of a discrete action. Used by the
        baselines; post-only rules and the position limit still apply."""
        return self._step(bid, ask, -1, -1)

    # -- helpers used by strategies --------------------------------------------------

    def action_to_prices(self, bid_level: int, ask_level: int) -> tuple[int | None, int | None]:
        bb, ba = self._ctx.best_bid(), self._ctx.best_ask()
        if bb is None or ba is None:
            return None, None
        return (
            level_price("bid", bid_level, bb[0], ba[0]),
            level_price("ask", ask_level, bb[0], ba[0]),
        )

    def market_view(self) -> MarketView:
        snap = self._ctx.snapshot
        assert snap is not None, "call reset() first"
        t = (snap.ts - self.cfg.warmup_ns) / 1e9
        return MarketView(
            t_s=t,
            time_left_s=self.cfg.horizon_s - t,
            best_bid=snap.best_bid(),
            best_ask=snap.best_ask(),
            mid=snap.mid(),
            microprice=snap.microprice(),
            position=self._ctx.position,
            quote_size=self.cfg.quote_size,
            max_position=self.cfg.max_position,
        )

    @property
    def engine(self) -> ReplayEngine:
        assert self._engine is not None, "call reset() first"
        return self._engine

    @property
    def ctx(self) -> AgentCtx:
        return self._ctx

    # -- internals ------------------------------------------------------------------

    def _advance(self) -> Snapshot:
        snap = self.engine.run_to_timer()
        if snap is None:
            raise RuntimeError("feed ended before the session; this is a bug")
        return snap

    def _exchange_mid(self) -> float:
        mid = self.engine.last_mid
        return self._mid if mid is None else mid

    def _step(
        self, bid: int | None, ask: int | None, bid_level: int, ask_level: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        cfg, engine, ctx = self.cfg, self.engine, self._ctx
        if self._k >= cfg.n_steps:
            raise RuntimeError("episode is over; call reset()")
        size = cfg.quote_size
        if bid is not None and ctx.position + size > cfg.max_position:
            bid = None
        if ask is not None and ctx.position - size < -cfg.max_position:
            ask = None
        sync_quotes(ctx, bid, ask, size)
        ctx.flush(engine)

        self._k += 1
        n_fills_before = len(engine.fills)
        engine.schedule_timer(cfg.warmup_ns + self._k * cfg.step_ns)
        snap = self._advance()

        # Reward from the exchange's books at the decision time.
        mid_prev, pos_prev = self._mid, self._position
        mid = self._exchange_mid()
        pos = engine.acct.position
        pnl = engine.acct.net_ticks(mid)
        d_pnl = pnl - self._pnl
        rc = cfg.reward
        inv_pen = rc.inventory_penalty * pos * pos * cfg.step_ms / 1000.0
        damp = rc.dampening * max(0.0, pos_prev * (mid - mid_prev))
        terminated = self._k == cfg.n_steps
        liq = 0.0
        if terminated and rc.terminal_liquidation:
            liq = liquidation_cost(engine.book, pos, cfg.sim.taker_fee)
        reward = d_pnl - inv_pen - damp - liq
        self._pnl, self._mid, self._position = pnl, mid, pos
        self._ep_reward += reward

        new_fills = engine.fills[n_fills_before:]
        self._n_fills += len(new_fills)
        self._abs_inv_sum += abs(pos)

        # What the agent learns this step: fills that reached it.
        bought = sold = 0
        for msg in snap.private:
            if msg.kind == "fill" and msg.fill is not None:
                if msg.fill.side > 0:
                    bought += msg.fill.qty
                else:
                    sold += msg.fill.qty
        obs = self._observe(snap, bought, sold)

        if self.record:
            tr = self.trace
            tr.t_s.append(self._k * cfg.step_ms / 1000.0)
            tr.mid.append(mid)
            tr.position.append(pos)
            tr.pnl.append(pnl)
            tr.reward.append(reward)
            tr.bid_level.append(bid_level)
            tr.ask_level.append(ask_level)

        info: dict[str, Any] = {
            "pnl_change": d_pnl,
            "inventory_penalty": inv_pen,
            "dampening": damp,
            "liquidation_cost": liq,
            "position": pos,
        }
        if terminated:
            info["episode_stats"] = {
                "reward": self._ep_reward,
                "pnl": pnl - liq,
                "fills": self._n_fills,
                "mean_abs_inventory": self._abs_inv_sum / cfg.n_steps,
                "final_position": pos,
            }
            if self.record:
                tr = self.trace
                tr.liquidation_cost = liq
                tr.fills = list(engine.fills)
                tr.mid_ts = list(engine.mid_ts)
                tr.mid_px = list(engine.mid_px)
                tr.fees = money_to_ticks(engine.acct.fees)
                tr.submitted_qty = engine.submitted_qty
        return obs, float(reward), terminated, False, info

    def _observe(self, snap: Snapshot, bought: int, sold: int) -> np.ndarray:
        self._ctx.update(snap)
        time_left = 1.0 - self._k / self.cfg.n_steps
        return self._features.build(snap, self._ctx, time_left, bought, sold)
