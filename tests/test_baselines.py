"""Avellaneda-Stoikov and heuristic quoters against hand-computed values."""

from __future__ import annotations

import math

import pytest

from rlmm.baselines import (
    ASParams,
    AvellanedaStoikov,
    GueantLehalleFernandezTapia,
    LinearSkewQuoter,
    SymmetricQuoter,
)
from rlmm.baselines.avellaneda_stoikov import (
    as_quotes,
    glft_offsets,
    optimal_spread,
    reservation_price,
    to_ticks,
)
from rlmm.baselines.calibration import IntensityPoint, fit_intensity
from rlmm.env.market_making import MarketView


def view(bid: int, ask: int, q: int, time_left: float = 30.0) -> MarketView:
    return MarketView(
        t_s=60.0 - time_left,
        time_left_s=time_left,
        best_bid=(bid, 10),
        best_ask=(ask, 10),
        mid=(bid + ask) / 2,
        microprice=(bid + ask) / 2,
        position=q,
        quote_size=5,
        max_position=50,
    )


def test_reservation_price_by_hand() -> None:
    # r = s - q * gamma * sigma^2 * tau = 100 - 2 * 0.1 * 4 * 10
    assert reservation_price(100.0, 2, 0.1, 2.0, 10.0) == pytest.approx(92.0)
    assert reservation_price(100.0, 0, 0.1, 2.0, 10.0) == 100.0
    assert reservation_price(100.0, -3, 0.1, 2.0, 10.0) == pytest.approx(112.0)


def test_optimal_spread_by_hand() -> None:
    # gamma sigma^2 tau + (2 / gamma) ln(1 + gamma / k)
    # = 0.1 * 4 * 10 + 20 * ln(1 + 0.1 / 1.5) = 4 + 1.290770...
    expected = 4.0 + 20.0 * math.log(1.0 + 0.1 / 1.5)
    assert optimal_spread(0.1, 2.0, 10.0, 1.5) == pytest.approx(expected)
    assert expected == pytest.approx(5.29077, abs=1e-5)
    # As tau -> 0 only the intensity term is left; as gamma -> 0 that tends to 2 / k.
    assert optimal_spread(1e-9, 2.0, 0.0, 1.5) == pytest.approx(2.0 / 1.5, rel=1e-6)


def test_quotes_symmetric_at_zero_inventory() -> None:
    p = ASParams(gamma=0.1, sigma=2.0, k=1.5)
    bid, ask = as_quotes(100.0, 0, p, 10.0)
    assert 100.0 - bid == pytest.approx(ask - 100.0)
    assert ask - bid == pytest.approx(optimal_spread(0.1, 2.0, 10.0, 1.5))


def test_quotes_skew_against_inventory() -> None:
    p = ASParams(gamma=0.1, sigma=2.0, k=1.5)
    flat = as_quotes(100.0, 0, p, 10.0)
    long = as_quotes(100.0, 2, p, 10.0)
    short = as_quotes(100.0, -2, p, 10.0)
    # Long: both quotes move down by q * gamma * sigma^2 * tau = 8 ticks.
    assert long[0] == pytest.approx(flat[0] - 8.0)
    assert long[1] == pytest.approx(flat[1] - 8.0)
    assert short[0] == pytest.approx(flat[0] + 8.0)
    # Spread width does not depend on inventory.
    assert long[1] - long[0] == pytest.approx(flat[1] - flat[0])


def test_skew_shrinks_toward_the_end_of_the_session() -> None:
    p = ASParams(gamma=0.1, sigma=2.0, k=1.5)
    early = as_quotes(100.0, 2, p, 30.0)
    late = as_quotes(100.0, 2, p, 1.0)
    assert (100.0 - (early[0] + early[1]) / 2) > (100.0 - (late[0] + late[1]) / 2) > 0


def test_rounding_and_no_crossing() -> None:
    assert to_ticks(99.4, 100.6, 99, 101) == (99, 101)
    assert to_ticks(99.5, 100.5, 99, 101) == (99, 101)  # ties go to the passive side
    assert to_ticks(99.6, 100.4, 99, 101) == (100, 101)  # rounds to 100/100: ask moved up
    assert to_ticks(99.0, 99.2, 98, 102) == (99, 100)
    assert to_ticks(103.0, 104.0, 99, 101) == (100, 104)  # never crosses the ask


def test_as_quoter_symmetric_then_skewed() -> None:
    q = AvellanedaStoikov(ASParams(gamma=0.01, sigma=2.0, k=2.0))
    b0, a0 = q.quote(view(100, 102, 0))
    assert b0 is not None and a0 is not None
    assert 101 - b0 == a0 - 101
    b1, a1 = q.quote(view(100, 102, 20))
    assert b1 is not None and a1 is not None
    assert b1 < b0 and a1 <= a0
    b2, a2 = q.quote(view(100, 102, -20))
    assert b2 is not None and a2 is not None
    assert b2 >= b0 and a2 > a0


def test_glft_offsets_by_hand() -> None:
    gamma, sigma, k, A = 0.1, 2.0, 1.5, 3.0
    c1 = math.log(1 + gamma / k) / gamma
    c2 = math.sqrt(sigma**2 * gamma / (2 * k * A) * (1 + gamma / k) ** (1 + k / gamma))
    db, da = glft_offsets(0, gamma, sigma, k, A)
    assert db == pytest.approx(c1 + c2 / 2)
    assert da == pytest.approx(c1 + c2 / 2)  # symmetric when flat
    db3, da3 = glft_offsets(3, gamma, sigma, k, A)
    assert db3 - db == pytest.approx(3 * c2)  # bid backs off c2 per lot
    assert da - da3 == pytest.approx(3 * c2)  # ask comes in c2 per lot


def test_glft_quoter_skews_with_inventory() -> None:
    q = GueantLehalleFernandezTapia(ASParams(0.05, 2.0, 2.0), A=5.0)
    b0, a0 = q.quote(view(100, 102, 0))
    b1, a1 = q.quote(view(100, 102, 30))
    assert b0 is not None and a0 is not None and b1 is not None and a1 is not None
    assert 101 - b0 == a0 - 101
    assert b1 < b0 and a1 < a0


def test_symmetric_quoter() -> None:
    assert SymmetricQuoter(0).quote(view(100, 101, 25)) == (100, 101)
    assert SymmetricQuoter(1).quote(view(100, 101, -25)) == (99, 102)


def test_linear_skew_quoter() -> None:
    s = LinearSkewQuoter(kappa=0.1)
    assert s.quote(view(100, 101, 0)) == (100, 101)  # joins the touch when flat
    assert s.quote(view(100, 102, 0)) == (100, 102)
    # Long 10 lots: reference 1 tick lower, so the bid backs off and, where the
    # spread allows, the ask comes inside it.
    assert s.quote(view(100, 101, 10)) == (99, 101)
    assert s.quote(view(100, 103, 10)) == (100, 101)
    assert s.quote(view(100, 101, -10)) == (100, 102)


def test_intensity_fit_recovers_parameters() -> None:
    A, k = 4.0, 2.5
    pts = [
        IntensityPoint(i, d, A * math.exp(-k * d), 100.0, 1000 // (i + 1))
        for i, d in enumerate([0.5, 1.5, 2.5, 3.5])
    ]
    fa, fk = fit_intensity(pts)
    assert fa == pytest.approx(A)
    assert fk == pytest.approx(k)
