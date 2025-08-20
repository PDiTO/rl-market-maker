"""A step-wise port of ``lob``'s event loop, driven from Python.

``lob.run_backtest`` owns its event loop: it runs the whole feed and calls the
strategy back. An RL environment needs the opposite, where the caller advances the
simulation one decision at a time. ``lob`` exposes the pieces to build that (the
matching engine as ``lob.OrderBook`` and the synthetic market as
``lob.generate_feed``), so this module re-implements the backtester's loop on top of
them with the same semantics:

* feed messages reach the exchange at their own timestamps;
* our orders and cancels reach it ``order_latency_ns`` after we send them;
* we see public and private messages ``market_data_latency_ns`` after the exchange
  produced them;
* ties go to the feed first, then everything else in the order it was scheduled.

The decision clock plays the role of ``lob``'s ``on_timer``. The equivalence with the
Rust backtester is tested fill for fill in ``tests/test_engine.py``.

Two things are added on top. The engine tracks the exact queue position of each of
our resting orders (lots ahead of it at its price), updated from the matching events
rather than from an O(book) L3 scan. And at each decision it takes a snapshot of the
book as of ``market_data_latency_ns`` earlier, which is exactly what ``lob``'s delayed
L2 view would show at that moment.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import lob

from rlmm.sim.accounting import Accounting, fee_to_money

STRATEGY_OWNER = 0
STRATEGY_ID_BASE = 1 << 60
FEED_OWNER_REMAP = 2**32 - 1

_ARRIVE = 0
_TIMER = 1

Level = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Latency and fees. Same meaning and defaults as ``lob``'s ``[sim]`` table."""

    order_latency_ns: int = 50_000
    market_data_latency_ns: int = 50_000
    maker_fee: float = -0.1
    """Ticks per lot when adding liquidity; negative is a rebate."""
    taker_fee: float = 0.3


@dataclass(slots=True)
class Fill:
    ts: int
    order_id: int
    side: int
    """+1 if we bought, -1 if we sold."""
    price: int
    qty: int
    leaves: int
    maker: bool
    mid_before: float | None


@dataclass(slots=True)
class OwnOrder:
    """Exchange-side state of one of our resting orders."""

    id: int
    side: str
    price: int
    qty: int
    seq: int
    ahead: int
    """Lots resting at the same price with higher priority."""


@dataclass(frozen=True, slots=True)
class OwnQueue:
    """Queue position of one of our resting orders, as of the snapshot."""

    id: int
    side: str
    price: int
    qty: int
    ahead: int
    level_qty: int


@dataclass(frozen=True, slots=True)
class PrivateMessage:
    """Ack, reject, cancel or fill for one of our orders, stamped with exchange time."""

    ts: int
    kind: str
    order_id: int
    request: str = ""
    reason: str = ""
    fill: Fill | None = None


@dataclass(slots=True)
class Snapshot:
    """What the strategy knows at a decision: everything the exchange did before
    ``cutoff = ts - market_data_latency_ns``."""

    ts: int
    cutoff: int
    bids: list[Level]
    asks: list[Level]
    own: list[OwnQueue]
    private: list[PrivateMessage]
    buy_volume: int
    """Public volume bought by aggressors since the previous snapshot."""
    sell_volume: int

    def best_bid(self) -> tuple[int, int] | None:
        return (self.bids[0][0], self.bids[0][1]) if self.bids else None

    def best_ask(self) -> tuple[int, int] | None:
        return (self.asks[0][0], self.asks[0][1]) if self.asks else None

    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2.0

    def microprice(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        (bp, bq, _), (ap, aq, _) = self.bids[0], self.asks[0]
        return (bp * aq + ap * bq) / (bq + aq)


@dataclass(slots=True)
class _Scheduled:
    ts: int
    seq: int
    kind: int
    cmd: tuple[Any, ...] = field(default=())

    def __lt__(self, other: _Scheduled) -> bool:
        return (self.ts, self.seq) < (other.ts, other.seq)


def _opposite(side: str) -> str:
    return "ask" if side == "bid" else "bid"


class ReplayEngine:
    """Replays a ``lob`` feed into a ``lob.OrderBook`` alongside our orders.

    Usage: ``schedule_timer(t)``, then ``run_to_timer()`` returns the snapshot at
    ``t``; send orders with ``send_new``/``send_cancel``; schedule the next timer;
    repeat. ``run_to_timer`` returns ``None`` once the feed has ended, with the same
    end-of-run rule as ``lob``: nothing after the last feed message is processed.
    """

    def __init__(self, feed: list[dict[str, Any]], sim: SimConfig, snapshot_depth: int = 5) -> None:
        self.sim = sim
        self.book = lob.OrderBook(emit_book_updates=False)
        self.acct = Accounting()
        self.fills: list[Fill] = []
        self.mid_ts: list[int] = []
        self.mid_px: list[float] = []
        self.last_mid: float | None = None
        self.now = 0
        self.feed_rejects = 0
        self.submitted_qty = 0
        """Lots of ours accepted by the exchange, as in ``lob``'s fill ratio."""

        self._feed = feed
        self._fi = 0
        self._last_feed_ts = feed[-1]["ts"] if feed else 0
        self._heap: list[_Scheduled] = []
        self._seq = 0
        self._timer_ts: int | None = None
        self._maker_fee = fee_to_money(sim.maker_fee)
        self._taker_fee = fee_to_money(sim.taker_fee)
        self._depth = snapshot_depth

        # Arrival order of every resting order, for queue-position bookkeeping.
        self._book_seq: dict[int, int] = {}
        self._next_book_seq = 0
        self._own: dict[int, OwnOrder] = {}
        self._private: deque[PrivateMessage] = deque()
        self._buy_vol = 0
        self._sell_vol = 0

    # -- scheduling -----------------------------------------------------------------

    def _schedule(self, ts: int, kind: int, cmd: tuple[Any, ...] = ()) -> None:
        self._seq += 1
        heapq.heappush(self._heap, _Scheduled(ts, self._seq, kind, cmd))

    def schedule_timer(self, ts: int) -> None:
        """Schedules the next decision. Call after sending that decision's orders."""
        if self._timer_ts is not None:
            raise RuntimeError("a timer is already scheduled")
        if ts - self.sim.market_data_latency_ns < self.now:
            raise ValueError("market-data latency must be shorter than the decision interval")
        self._timer_ts = ts
        self._schedule(ts, _TIMER)

    def send_new(self, order_id: int, side: str, price: int, qty: int, order_type: str) -> None:
        self._schedule(
            self.now + self.sim.order_latency_ns,
            _ARRIVE,
            ("new", order_id, side, price, qty, order_type),
        )

    def send_cancel(self, order_id: int) -> None:
        self._schedule(self.now + self.sim.order_latency_ns, _ARRIVE, ("cancel", order_id))

    # -- main loop ------------------------------------------------------------------

    def run_to_timer(self) -> Snapshot | None:
        """Processes everything due before the scheduled timer and returns the view
        at the timer, or None if the feed ends first."""
        if self._timer_ts is None:
            raise RuntimeError("no timer scheduled")
        cutoff = self._timer_ts - self.sim.market_data_latency_ns
        snap: Snapshot | None = None
        feed, heap = self._feed, self._heap
        n = len(feed)
        while True:
            feed_ts = feed[self._fi]["ts"] if self._fi < n else None
            if feed_ts is not None and (not heap or feed_ts <= heap[0].ts):
                if snap is None and feed_ts >= cutoff:
                    snap = self._snapshot(cutoff)
                e = feed[self._fi]
                self._fi += 1
                self.now = max(self.now, feed_ts)
                self._feed_event(e)
                continue
            if not heap or (feed_ts is None and heap[0].ts > self._last_feed_ts):
                return None
            item = heapq.heappop(heap)
            if snap is None and item.ts >= cutoff:
                snap = self._snapshot(cutoff)
            self.now = max(self.now, item.ts)
            if item.kind == _TIMER:
                self._timer_ts = None
                assert snap is not None
                snap.ts = item.ts
                return snap
            self._our_command(item.cmd)

    def drain(self) -> None:
        """Processes everything up to the last feed message, dropping the timer."""
        self._heap = [s for s in self._heap if s.kind != _TIMER]
        heapq.heapify(self._heap)
        self._timer_ts = None
        feed, heap = self._feed, self._heap
        while True:
            feed_ts = feed[self._fi]["ts"] if self._fi < len(feed) else None
            if feed_ts is not None and (not heap or feed_ts <= heap[0].ts):
                e = feed[self._fi]
                self._fi += 1
                self.now = max(self.now, feed_ts)
                self._feed_event(e)
            elif heap and heap[0].ts <= self._last_feed_ts:
                item = heapq.heappop(heap)
                self.now = max(self.now, item.ts)
                self._our_command(item.cmd)
            else:
                return

    # -- exchange -------------------------------------------------------------------

    def _feed_event(self, e: dict[str, Any]) -> None:
        action = e["action"]
        if action == "new":
            owner = e["owner"]
            if owner == STRATEGY_OWNER:
                owner = FEED_OWNER_REMAP
            events = self.book.submit(
                e["id"], e["side"], e["price"], e["qty"], owner, e["order_type"]
            )
        elif action == "cancel":
            events = self.book.cancel(e["id"])
        else:
            events = self.book.amend(e["id"], e["price"], e["qty"])
        self._route(events, None)

    def _our_command(self, cmd: tuple[Any, ...]) -> None:
        if cmd[0] == "new":
            _, oid, side, price, qty, order_type = cmd
            events = self.book.submit(oid, side, price, qty, STRATEGY_OWNER, order_type)
            self._route(events, cmd)
        else:
            self._route(self.book.cancel(cmd[1]), cmd)

    def _route(self, events: list[dict[str, Any]], ours: tuple[Any, ...] | None) -> None:
        mid_before = self.last_mid
        ts = self.now
        own = self._own
        taker_leaves = ours[4] if ours is not None and ours[0] == "new" else 0
        for ev in events:
            kind = ev["type"]
            if kind == "trade":
                price, qty = ev["price"], ev["qty"]
                taker_side = ev["taker_side"]
                maker_id = ev["maker_id"]
                maker_side = _opposite(taker_side)
                # Price-time priority: any of our orders at this level other than the
                # maker is behind the maker, so the queue ahead of it shrinks.
                for o in own.values():
                    if o.side == maker_side and o.price == price and o.id != maker_id:
                        o.ahead = max(0, o.ahead - qty)
                if ev["maker_owner"] == STRATEGY_OWNER:
                    remaining = ev["maker_remaining"]
                    self._book_fill(
                        Fill(
                            ts,
                            maker_id,
                            1 if maker_side == "bid" else -1,
                            price,
                            qty,
                            remaining,
                            True,
                            mid_before,
                        )
                    )
                    mine = own.get(maker_id)
                    if mine is not None:
                        mine.qty, mine.ahead = remaining, 0
                        if remaining == 0:
                            del own[maker_id]
                if ev["taker_owner"] == STRATEGY_OWNER:
                    taker_leaves -= qty
                    self._book_fill(
                        Fill(
                            ts,
                            ev["taker_id"],
                            1 if taker_side == "bid" else -1,
                            price,
                            qty,
                            taker_leaves,
                            False,
                            mid_before,
                        )
                    )
                if ev["maker_remaining"] == 0:
                    self._book_seq.pop(maker_id, None)
                if taker_side == "bid":
                    self._buy_vol += qty
                else:
                    self._sell_vol += qty
            elif kind == "accepted":
                self._book_seq[ev["id"]] = self._next_book_seq
                self._next_book_seq += 1
                if ev["owner"] == STRATEGY_OWNER:
                    self.submitted_qty += ev["qty"]
                    self._private.append(PrivateMessage(ts, "accepted", ev["id"]))
            elif kind == "cancelled":
                oid = ev["id"]
                seq = self._book_seq.pop(oid, None)
                if seq is not None:
                    for o in own.values():
                        if o.side == ev["side"] and o.price == ev["price"] and seq < o.seq:
                            o.ahead = max(0, o.ahead - ev["qty"])
                if ev["owner"] == STRATEGY_OWNER:
                    own.pop(oid, None)
                    self._private.append(PrivateMessage(ts, "cancelled", oid, reason=ev["reason"]))
            elif kind == "rejected":
                if ours is not None:
                    self._private.append(
                        PrivateMessage(ts, "rejected", ev["id"], ev["request"], ev["reason"])
                    )
                else:
                    self.feed_rejects += 1
            elif kind == "amended":
                self._feed_amended(ev)

        if ours is not None and ours[0] == "new":
            oid = ours[1]
            resting = self.book.order(oid)
            if resting is not None:
                side, price = resting["side"], resting["price"]
                level = self._level_qty(side, price)
                own[oid] = OwnOrder(
                    oid,
                    side,
                    price,
                    resting["qty"],
                    self._book_seq[oid],
                    level - resting["qty"],
                )
        mid = self.book.mid()
        if mid is not None and mid != self.last_mid:
            self.last_mid = mid
            self.mid_ts.append(ts)
            self.mid_px.append(mid)

    def _feed_amended(self, ev: dict[str, Any]) -> None:
        """A feed order was amended (only happens with recorded CSV feeds)."""
        oid = ev["id"]
        seq = self._book_seq.get(oid)
        for o in self._own.values():
            if seq is None or seq > o.seq or o.side != ev["side"]:
                continue
            if o.price == ev["old_price"]:
                if ev["kept_priority"]:
                    o.ahead = max(0, o.ahead - (ev["old_qty"] - ev["new_qty"]))
                else:
                    o.ahead = max(0, o.ahead - ev["old_qty"])
        if not ev["kept_priority"]:
            self._book_seq[oid] = self._next_book_seq
            self._next_book_seq += 1

    def _book_fill(self, fill: Fill) -> None:
        fee = self._maker_fee if fill.maker else self._taker_fee
        self.acct.fill(fill.side, fill.price, fill.qty, fee)
        self.fills.append(fill)
        self._private.append(PrivateMessage(fill.ts, "fill", fill.order_id, fill=fill))

    def _level_qty(self, side: str, price: int) -> int:
        for depth in (10, 1_000_000):
            levels = self.book.l2(depth)["bids" if side == "bid" else "asks"]
            for p, q, _ in levels:
                if p == price:
                    return int(q)
            if len(levels) < depth:
                break
        return 0

    def _snapshot(self, cutoff: int) -> Snapshot:
        l2 = self.book.l2(self._depth)
        own: list[OwnQueue] = []
        for o in self._own.values():
            own.append(
                OwnQueue(o.id, o.side, o.price, o.qty, o.ahead, self._level_qty(o.side, o.price))
            )
        private: list[PrivateMessage] = []
        while self._private and self._private[0].ts < cutoff:
            private.append(self._private.popleft())
        snap = Snapshot(
            ts=cutoff,
            cutoff=cutoff,
            bids=l2["bids"],
            asks=l2["asks"],
            own=own,
            private=private,
            buy_volume=self._buy_vol,
            sell_volume=self._sell_vol,
        )
        self._buy_vol = self._sell_vol = 0
        return snap

    # -- inspection -----------------------------------------------------------------

    def own_orders(self) -> list[OwnOrder]:
        """Exchange-side state of our resting orders, right now."""
        return list(self._own.values())

    def queue_ahead_from_l3(self, order_id: int) -> int | None:
        """Lots ahead of ``order_id`` from a full L3 scan. Slow; used in tests."""
        o = self.book.order(order_id)
        if o is None:
            return None
        side = self.book.l3()["bids" if o["side"] == "bid" else "asks"]
        return sum(
            int(r["qty"]) for r in side if r["price"] == o["price"] and r["seq"] < o["seq"]
        )
