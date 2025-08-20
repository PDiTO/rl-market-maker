"""The step-wise engine against lob's own Rust backtester.

The same quoting logic runs twice: inside ``lob.run_backtest`` as a Python strategy
on the Rust event loop, and on this repo's ``ReplayEngine``. Every fill, and the
final cash, position, fees and realized PnL, must agree exactly.
"""

from __future__ import annotations

from typing import Any

import lob
import pytest

from rlmm.sim import AgentCtx, ReplayEngine, SimConfig, sync_quotes
from rlmm.sim.accounting import mark_to_money, money_to_ticks

STEP_NS = 100_000_000
MARKET = {"seed": 7, "duration_s": 20}


def decide(ctx: Any, size: int = 3, max_pos: int = 15) -> tuple[int | None, int | None]:
    """Join the touch, lean back one tick on the side that adds to a big position."""
    bb, ba = ctx.best_bid(), ctx.best_ask()
    if bb is None or ba is None:
        return None, None
    pos = ctx.position
    bid: int | None = bb[0] - (1 if pos > 6 else 0)
    ask: int | None = ba[0] + (1 if pos < -6 else 0)
    if pos + size > max_pos:
        bid = None
    if pos - size < -max_pos:
        ask = None
    return bid, ask


class ScriptedLobStrategy(lob.Strategy):
    def on_timer(self, ctx: lob.Ctx) -> None:
        bid, ask = decide(ctx)
        sync_quotes(ctx, bid, ask, 3)


def run_engine(order_ns: int, md_ns: int) -> ReplayEngine:
    engine = ReplayEngine(
        lob.generate_feed(**MARKET),
        SimConfig(order_latency_ns=order_ns, market_data_latency_ns=md_ns),
    )
    ctx = AgentCtx()
    engine.schedule_timer(STEP_NS)
    while (snap := engine.run_to_timer()) is not None:
        ctx.update(snap)
        bid, ask = decide(ctx)
        sync_quotes(ctx, bid, ask, 3)
        ctx.flush(engine)
        engine.schedule_timer(snap.ts + STEP_NS)
    engine.drain()
    return engine


@pytest.mark.parametrize(
    ("order_ns", "md_ns"),
    [(50_000, 50_000), (0, 0), (5_000_000, 2_000_000), (0, 30_000_000), (40_000_000, 0)],
)
def test_matches_lob_backtester_fill_for_fill(order_ns: int, md_ns: int) -> None:
    cfg = {
        "market": MARKET,
        "sim": {
            "order_latency_ns": order_ns,
            "market_data_latency_ns": md_ns,
            "timer_interval_ns": STEP_NS,
        },
    }
    ref = lob.run_backtest(cfg, strategy=ScriptedLobStrategy())
    engine = run_engine(order_ns, md_ns)

    ours = [
        (f.ts, f.order_id, "bid" if f.side > 0 else "ask", f.price, f.qty, f.mid_before)
        for f in engine.fills
    ]
    theirs = [
        (f["ts"], f["order_id"], f["side"], f["price"], f["qty"], f["mid_before"])
        for f in ref["fills"]
    ]
    assert len(theirs) > 20
    assert ours == theirs

    s = ref["summary"]
    acct = engine.acct
    assert acct.position == s["final_position"]
    assert money_to_ticks(acct.cash) == pytest.approx(s["cash"], abs=1e-9)
    assert money_to_ticks(acct.fees) == pytest.approx(s["fees"], abs=1e-9)
    assert money_to_ticks(acct.realized) == pytest.approx(s["realized_pnl"], abs=1e-9)
    assert acct.net_ticks(engine.last_mid) == pytest.approx(s["net_pnl"], abs=1e-9)
    assert engine.submitted_qty == s["submitted_qty"]


def test_accounting_identity_and_fill_log_agree() -> None:
    engine = run_engine(50_000, 50_000)
    acct = engine.acct
    mark = mark_to_money(engine.last_mid or 0.0)
    assert acct.cash + acct.position * mark == acct.realized + acct.unrealized(mark) - acct.fees
    # Position and cash rebuilt from the fill log alone.
    assert acct.position == sum(f.side * f.qty for f in engine.fills)
    cash = sum(-f.side * f.qty * f.price for f in engine.fills) + 0.1 * sum(
        f.qty for f in engine.fills
    )
    assert money_to_ticks(acct.cash) == pytest.approx(cash, abs=1e-6)


def test_queue_ahead_tracking_matches_l3_scan() -> None:
    """Incremental queue position equals a full L3 recount at every decision."""
    engine = ReplayEngine(lob.generate_feed(seed=11, duration_s=20), SimConfig())
    ctx = AgentCtx()
    engine.schedule_timer(STEP_NS)
    checked = 0
    while (snap := engine.run_to_timer()) is not None:
        ctx.update(snap)
        for o in engine.own_orders():
            assert o.ahead == engine.queue_ahead_from_l3(o.id)
            checked += 1
        bb, ba = ctx.best_bid(), ctx.best_ask()
        if bb and ba:
            # Sit one tick back most of the time so orders build up queue history.
            k = (snap.ts // STEP_NS) % 7
            sync_quotes(ctx, bb[0] - (k % 2), ba[0] + (k % 3 == 0), 2)
        ctx.flush(engine)
        engine.schedule_timer(snap.ts + STEP_NS)
    assert checked > 100


def test_order_latency_delays_arrival() -> None:
    """An order sent at t reaches the book at t + order latency, not before."""
    engine = ReplayEngine(
        lob.generate_feed(seed=3, duration_s=5),
        SimConfig(order_latency_ns=20_000_000, market_data_latency_ns=0),
    )
    ctx = AgentCtx()
    engine.schedule_timer(STEP_NS)
    snap = engine.run_to_timer()
    assert snap is not None
    ctx.update(snap)
    bb = ctx.best_bid()
    assert bb is not None
    oid = ctx.post_only("bid", bb[0] - 3, 1)
    ctx.flush(engine)
    engine.schedule_timer(STEP_NS + 10_000_000)
    engine.run_to_timer()
    assert engine.book.order(oid) is None
    engine.schedule_timer(STEP_NS + 30_000_000)
    engine.run_to_timer()
    assert engine.book.order(oid) is not None


def test_market_data_latency_hides_recent_fills() -> None:
    """The agent's position lags the exchange's by market-data latency."""
    md = 60_000_000
    engine = ReplayEngine(
        lob.generate_feed(seed=5, duration_s=30),
        SimConfig(order_latency_ns=0, market_data_latency_ns=md),
    )
    ctx = AgentCtx()
    engine.schedule_timer(STEP_NS)
    lagged = 0
    while (snap := engine.run_to_timer()) is not None:
        ctx.update(snap)
        known = sum(f.side * f.qty for f in engine.fills if f.ts < snap.ts - md)
        assert ctx.position == known
        lagged += ctx.position != engine.acct.position
        bb, ba = ctx.best_bid(), ctx.best_ask()
        if bb and ba:
            sync_quotes(ctx, bb[0], ba[0], 2)
        ctx.flush(engine)
        engine.schedule_timer(snap.ts + STEP_NS)
    assert lagged > 0
