"""Position, cash and PnL bookkeeping, a line-for-line port of ``lob``'s accounting.

Money is held in integer micro-ticks (one tick = 1,000,000 units per lot), so prices,
fractional fees and half-tick mids are exact and the numbers agree bit for bit with
the Rust backtester. Realized PnL uses average cost. The books balance by
construction::

    cash + position * mark == realized + unrealized - fees
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MONEY_SCALE = 1_000_000


def _round_half_away(x: float) -> int:
    """Rounds like Rust's ``f64::round``: halves go away from zero."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def fee_to_money(ticks_per_lot: float) -> int:
    """Fee in ticks per lot to money units per lot."""
    return _round_half_away(ticks_per_lot * MONEY_SCALE)


def mark_to_money(mark: float) -> int:
    """Mark price in ticks (possibly a half tick) to money units per lot."""
    return _round_half_away(mark * MONEY_SCALE)


def money_to_ticks(m: int) -> float:
    return m / MONEY_SCALE


def _trunc_div(a: int, b: int) -> int:
    """Integer division truncating toward zero, as Rust's ``/`` on integers does."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


@dataclass(slots=True)
class Accounting:
    """Average-cost accounting for one instrument. ``side`` is +1 buy, -1 sell."""

    position: int = 0
    cash: int = 0
    """Trade cash flows minus fees, in money units."""
    basis: int = 0
    """Signed cost of the open position (negative when short)."""
    realized: int = 0
    fees: int = 0

    def fill(self, side: int, price: int, qty: int, fee: int) -> None:
        """Books one fill. ``fee`` is money units per lot; negative for a rebate."""
        px = price * MONEY_SCALE
        self.cash -= side * qty * px
        fee_total = fee * qty
        self.fees += fee_total
        self.cash -= fee_total

        pos = self.position
        opening = qty
        if pos != 0 and (1 if pos > 0 else -1) != side:
            closing = min(qty, abs(pos))
            basis_removed = _trunc_div(self.basis * closing, abs(pos))
            self.realized += -side * closing * px - basis_removed
            self.basis -= basis_removed
            self.position += side * closing
            opening -= closing
        if opening > 0:
            self.basis += side * opening * px
            self.position += side * opening

    def unrealized(self, mark: int) -> int:
        return self.position * mark - self.basis

    def net(self, mark: int) -> int:
        """Net PnL in money units: realized + unrealized - fees."""
        return self.cash + self.position * mark

    def net_ticks(self, mid: float | None) -> float:
        """Net PnL in ticks x lots, marked at ``mid`` (zero mark if there is none)."""
        mark = 0 if mid is None else mark_to_money(mid)
        return money_to_ticks(self.net(mark))
