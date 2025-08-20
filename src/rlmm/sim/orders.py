"""The strategy's side of the wire: what it believes about its orders, and quoting.

:class:`AgentCtx` mirrors ``lob.Ctx`` (same method names, same return shapes, same
state rules), so quoting logic written against it runs unchanged inside
``lob.run_backtest``. That is what lets the tests check this repo's step-wise engine
against the Rust backtester fill for fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rlmm.sim.engine import STRATEGY_ID_BASE, PrivateMessage, ReplayEngine, Snapshot


@dataclass(slots=True)
class OpenOrder:
    id: int
    side: str
    order_type: str
    price: int
    qty: int
    pending: str
    """One of ``none``, ``new``, ``cancel``."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "side": self.side,
            "order_type": self.order_type,
            "price": self.price,
            "qty": self.qty,
            "pending": self.pending,
        }


class QuoteCtx(Protocol):
    """The subset of ``lob.Ctx`` that quoting needs."""

    def open_orders(self) -> list[dict[str, Any]]: ...
    def post_only(self, side: str, price: int, qty: int) -> int: ...
    def cancel(self, id: int) -> bool: ...


class AgentCtx:
    """The agent's delayed view: book as of the last snapshot, and orders and
    position rebuilt only from private messages that have arrived."""

    def __init__(self) -> None:
        self.now = 0
        self.position = 0
        self.snapshot: Snapshot | None = None
        self._open: dict[int, OpenOrder] = {}
        self._next_id = STRATEGY_ID_BASE
        self._actions: list[tuple[Any, ...]] = []

    # -- market data ----------------------------------------------------------------

    def update(self, snap: Snapshot) -> list[PrivateMessage]:
        """Applies a new snapshot and the private messages that arrived with it."""
        self.snapshot = snap
        self.now = snap.ts
        for msg in snap.private:
            self._apply(msg)
        return snap.private

    def best_bid(self) -> tuple[int, int] | None:
        return self.snapshot.best_bid() if self.snapshot else None

    def best_ask(self) -> tuple[int, int] | None:
        return self.snapshot.best_ask() if self.snapshot else None

    def mid(self) -> float | None:
        return self.snapshot.mid() if self.snapshot else None

    # -- orders ---------------------------------------------------------------------

    def open_orders(self) -> list[dict[str, Any]]:
        return [o.as_dict() for o in self._open.values()]

    def open_order_list(self) -> list[OpenOrder]:
        return list(self._open.values())

    def submit(self, side: str, price: int, qty: int, order_type: str = "limit") -> int:
        oid = self._next_id
        self._next_id += 1
        self._actions.append(("new", oid, side, price, qty, order_type))
        self._open[oid] = OpenOrder(oid, side, order_type, price, qty, "new")
        return oid

    def post_only(self, side: str, price: int, qty: int) -> int:
        return self.submit(side, price, qty, "post_only")

    def cancel(self, id: int) -> bool:
        o = self._open.get(id)
        if o is None or o.pending == "cancel":
            return False
        o.pending = "cancel"
        self._actions.append(("cancel", id))
        return True

    def cancel_all(self) -> None:
        for oid in list(self._open):
            self.cancel(oid)

    def flush(self, engine: ReplayEngine) -> None:
        """Sends queued actions, in order, like ``lob``'s ``flush_actions``."""
        for a in self._actions:
            if a[0] == "new":
                engine.send_new(a[1], a[2], a[3], a[4], a[5])
            else:
                engine.send_cancel(a[1])
        self._actions.clear()

    def _apply(self, msg: PrivateMessage) -> None:
        kind = msg.kind
        if kind == "fill":
            f = msg.fill
            assert f is not None
            self.position += f.side * f.qty
            if f.leaves == 0:
                self._open.pop(f.order_id, None)
            elif (o := self._open.get(f.order_id)) is not None:
                o.qty = f.leaves
        elif kind == "accepted":
            o = self._open.get(msg.order_id)
            if o is not None and o.pending == "new":
                o.pending = "none"
        elif kind == "cancelled":
            self._open.pop(msg.order_id, None)
        elif kind == "rejected":
            if msg.request == "new" or msg.reason == "unknown_order":
                self._open.pop(msg.order_id, None)
            elif (o := self._open.get(msg.order_id)) is not None:
                o.pending = "none"


def sync_quotes(ctx: QuoteCtx, bid: int | None, ask: int | None, qty: int) -> None:
    """Moves our resting quotes to ``bid`` and ``ask`` (None means no quote).

    An order already at the target price is left alone, so it keeps its place in the
    queue even if it has been partly filled. Anything else is cancelled and a new
    post-only order goes to the back of the queue at the target.
    """
    targets = {"bid": bid, "ask": ask}
    kept = {"bid": False, "ask": False}
    for o in ctx.open_orders():
        if o["pending"] == "cancel":
            continue
        side = o["side"]
        if not kept[side] and o["price"] == targets[side]:
            kept[side] = True
        else:
            ctx.cancel(o["id"])
    for side in ("bid", "ask"):
        price = targets[side]
        if price is not None and not kept[side]:
            ctx.post_only(side, price, qty)
