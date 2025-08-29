"""Avellaneda and Stoikov (2008), "High-frequency trading in a limit order book".

The mid follows ``dS = sigma dW``, the market maker has CARA utility with risk
aversion ``gamma`` over terminal wealth at ``T``, and a quote at distance ``delta``
from the mid is filled at Poisson rate ``lambda(delta) = A exp(-k delta)``. The
approximate solution is a reservation price

    r = s - q gamma sigma^2 (T - t)

and a total spread around it of

    delta_a + delta_b = gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k).

Units here: prices in ticks, ``sigma`` in ticks per sqrt(second), time in seconds,
``q`` in lots, ``k`` per tick, ``gamma`` per tick per lot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rlmm.env.market_making import MarketView


@dataclass(frozen=True)
class ASParams:
    gamma: float
    sigma: float
    k: float


def reservation_price(mid: float, q: float, gamma: float, sigma: float, tau: float) -> float:
    return mid - q * gamma * sigma**2 * tau


def optimal_spread(gamma: float, sigma: float, tau: float, k: float) -> float:
    """Total bid-ask spread around the reservation price."""
    return gamma * sigma**2 * tau + (2.0 / gamma) * math.log1p(gamma / k)


def as_quotes(mid: float, q: float, p: ASParams, tau: float) -> tuple[float, float]:
    """Continuous (bid, ask) prices."""
    r = reservation_price(mid, q, p.gamma, p.sigma, tau)
    half = optimal_spread(p.gamma, p.sigma, tau, p.k) / 2.0
    return r - half, r + half


def to_ticks(bid: float, ask: float, best_bid: int, best_ask: int) -> tuple[int, int]:
    """Rounds model prices to the nearest tick (ties go to the passive side), keeps
    them from crossing the visible touch since quotes are post-only, and keeps the
    ask at least a tick above our own bid."""
    b = min(math.ceil(bid - 0.5), best_ask - 1)
    a = max(math.floor(ask + 0.5), best_bid + 1, b + 1)
    return b, a


class AvellanedaStoikov:
    """AS quoting on the session horizon. ``q`` is the agent's known position."""

    def __init__(self, params: ASParams, name: str = "avellaneda_stoikov") -> None:
        self.params = params
        self.name = name

    def reset(self) -> None:
        pass

    def quote(self, view: MarketView) -> tuple[int | None, int | None]:
        if view.best_bid is None or view.best_ask is None or view.mid is None:
            return None, None
        tau = max(view.time_left_s, 0.0)
        bid, ask = as_quotes(view.mid, view.position, self.params, tau)
        return to_ticks(bid, ask, view.best_bid[0], view.best_ask[0])


def glft_offsets(q: float, gamma: float, sigma: float, k: float, A: float) -> tuple[float, float]:
    """Asymptotic (long-horizon) optimal distances from the mid for bid and ask.

    Gueant, Lehalle and Fernandez-Tapia (2013) solve the AS problem with an
    inventory limit and show that far from the terminal time the quotes stop
    depending on ``T - t``::

        delta_b = c1 + (2q + 1) / 2 * c2        delta_a = c1 - (2q - 1) / 2 * c2
        c1 = (1 / gamma) ln(1 + gamma / k)
        c2 = sqrt(sigma^2 gamma / (2 k A) * (1 + gamma / k) ** (1 + k / gamma))

    So the skew is a constant ``c2`` ticks per lot instead of AS's
    ``gamma sigma^2 (T - t)``, which is large early in a session and vanishes at
    the end. Returns ``(delta_b, delta_a)``.
    """
    c1 = math.log1p(gamma / k) / gamma
    c2 = math.sqrt(sigma**2 * gamma / (2.0 * k * A) * (1.0 + gamma / k) ** (1.0 + k / gamma))
    return c1 + (2.0 * q + 1.0) / 2.0 * c2, c1 - (2.0 * q - 1.0) / 2.0 * c2


class GueantLehalleFernandezTapia:
    """The stationary AS quotes from :func:`glft_offsets`, with ``q`` in lots."""

    def __init__(self, params: ASParams, A: float, name: str = "as_stationary") -> None:
        self.params = params
        self.A = A
        self.name = name

    def reset(self) -> None:
        pass

    def quote(self, view: MarketView) -> tuple[int | None, int | None]:
        if view.best_bid is None or view.best_ask is None or view.mid is None:
            return None, None
        p = self.params
        db, da = glft_offsets(view.position, p.gamma, p.sigma, p.k, self.A)
        return to_ticks(view.mid - db, view.mid + da, view.best_bid[0], view.best_ask[0])
