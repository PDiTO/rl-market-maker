"""Per-episode metrics and PnL decomposition from an :class:`EpisodeTrace`."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from rlmm.env.market_making import NO_QUOTE, EpisodeTrace

MARKOUT_HORIZONS_S = (0.1, 1.0, 5.0)


@dataclass(frozen=True)
class EpisodeMetrics:
    feed_seed: int
    pnl: float
    """Net PnL after flattening the final position into the book, ticks x lots."""
    pnl_mtm: float
    """Mark-to-market PnL at the final mid, before liquidation."""
    reward: float
    sharpe: float
    """Mean over std of 1-second PnL changes within the episode (not annualised)."""
    max_drawdown: float
    mean_abs_inventory: float
    max_abs_inventory: int
    final_inventory: int
    fills: int
    filled_lots: int
    fill_rate: float
    """Filled lots over lots accepted by the exchange."""
    spread_capture: float
    """Sum over fills of side * qty * (mid before fill - price)."""
    inventory_pnl: float
    """Sum over fills of side * qty * (final mid - mid before fill): the PnL from
    holding inventory while the mid moved."""
    rebates: float
    liquidation_cost: float
    markout_0_1s: float
    """Mid move over 0.1 s after our fills, in our favour, per lot. Negative means
    adverse selection."""
    markout_1s: float
    markout_5s: float
    bid_quoted_share: float
    ask_quoted_share: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def _mid_at(ts: np.ndarray, px: np.ndarray, t: np.ndarray) -> np.ndarray:
    i = np.searchsorted(ts, t, side="right") - 1
    return px[np.clip(i, 0, len(px) - 1)]


def episode_metrics(
    trace: EpisodeTrace,
    feed_seed: int,
    step_s: float,
    warmup_s: float,
    horizon_s: float,
) -> EpisodeMetrics:
    pnl_path = np.asarray(trace.pnl)
    pos = np.asarray(trace.position)
    final_mid = trace.mid[-1]

    # 1-second PnL changes for the within-episode Sharpe.
    per_s = max(1, round(1.0 / step_s))
    sampled = np.concatenate([[0.0], pnl_path[per_s - 1 :: per_s]])
    d = np.diff(sampled)
    sharpe = float(d.mean() / d.std()) if d.size > 1 and d.std() > 0 else 0.0
    path = np.concatenate([[0.0], pnl_path])
    max_dd = float((np.maximum.accumulate(path) - path).max())

    fills = trace.fills
    side = np.array([f.side for f in fills], dtype=np.float64)
    qty = np.array([f.qty for f in fills], dtype=np.float64)
    price = np.array([f.price for f in fills], dtype=np.float64)
    mid_before = np.array(
        [f.mid_before if f.mid_before is not None else f.price for f in fills], dtype=np.float64
    )
    spread_capture = float(np.sum(side * qty * (mid_before - price)))
    inventory_pnl = float(np.sum(side * qty * (final_mid - mid_before)))
    filled = int(qty.sum())

    markouts = []
    end_ns = round((warmup_s + horizon_s) * 1e9)
    mts = np.asarray(trace.mid_ts, dtype=np.int64)
    mpx = np.asarray(trace.mid_px, dtype=np.float64)
    fts = np.array([f.ts for f in fills], dtype=np.int64)
    for h in MARKOUT_HORIZONS_S:
        h_ns = round(h * 1e9)
        ok = fts + h_ns <= end_ns
        if not ok.any() or mts.size == 0:
            markouts.append(0.0)
            continue
        after = _mid_at(mts, mpx, fts[ok] + h_ns)
        move = side[ok] * (after - mid_before[ok]) * qty[ok]
        markouts.append(float(move.sum() / qty[ok].sum()))

    bid_q = np.asarray(trace.bid_offset)
    ask_q = np.asarray(trace.ask_offset)

    return EpisodeMetrics(
        feed_seed=feed_seed,
        pnl=float(pnl_path[-1] - trace.liquidation_cost),
        pnl_mtm=float(pnl_path[-1]),
        reward=float(np.sum(trace.reward)),
        sharpe=sharpe,
        max_drawdown=max_dd,
        mean_abs_inventory=float(np.abs(pos).mean()),
        max_abs_inventory=int(np.abs(pos).max()),
        final_inventory=int(pos[-1]),
        fills=len(fills),
        filled_lots=filled,
        fill_rate=filled / trace.submitted_qty if trace.submitted_qty else 0.0,
        spread_capture=spread_capture,
        inventory_pnl=inventory_pnl,
        rebates=-trace.fees,
        liquidation_cost=trace.liquidation_cost,
        markout_0_1s=markouts[0],
        markout_1s=markouts[1],
        markout_5s=markouts[2],
        bid_quoted_share=float(np.mean(bid_q != NO_QUOTE)) if bid_q.size else 0.0,
        ask_quoted_share=float(np.mean(ask_q != NO_QUOTE)) if ask_q.size else 0.0,
    )
