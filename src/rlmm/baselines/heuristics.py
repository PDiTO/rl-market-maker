"""Two hand-written quoters to anchor the comparison."""

from __future__ import annotations

import math

from rlmm.env.market_making import MarketView


class SymmetricQuoter:
    """Quotes ``offset`` ticks behind the best bid and best ask, whatever the
    inventory. With ``offset=0`` it simply joins the touch on both sides."""

    def __init__(self, offset: int = 0, name: str = "symmetric") -> None:
        self.offset = offset
        self.name = name

    def reset(self) -> None:
        pass

    def quote(self, view: MarketView) -> tuple[int | None, int | None]:
        if view.best_bid is None or view.best_ask is None:
            return None, None
        return view.best_bid[0] - self.offset, view.best_ask[0] + self.offset


class LinearSkewQuoter:
    """Quotes ``half_spread`` either side of a reference that shifts against
    inventory by ``kappa`` ticks per lot: ``ref = mid - kappa * q``.

    With ``half_spread = 0.5`` and no inventory it joins the touch. Long inventory
    lowers both quotes, so the bid backs off and the ask gets more aggressive.
    """

    def __init__(self, kappa: float, half_spread: float = 0.5, name: str = "linear_skew") -> None:
        self.kappa = kappa
        self.half_spread = half_spread
        self.name = name

    def reset(self) -> None:
        pass

    def quote(self, view: MarketView) -> tuple[int | None, int | None]:
        if view.best_bid is None or view.best_ask is None or view.mid is None:
            return None, None
        ref = view.mid - self.kappa * view.position
        bid = math.floor(ref - self.half_spread + 1e-9)
        ask = math.ceil(ref + self.half_spread - 1e-9)
        return min(bid, view.best_ask[0] - 1), max(ask, view.best_bid[0] + 1)
