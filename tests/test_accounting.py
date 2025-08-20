"""The accounting port matches lob's Rust unit tests, number for number."""

from rlmm.sim.accounting import Accounting, fee_to_money, mark_to_money

S = 1_000_000
BUY, SELL = 1, -1


def test_round_trip_realizes_the_price_difference() -> None:
    a = Accounting()
    a.fill(BUY, 100, 3, 0)
    a.fill(SELL, 104, 3, 0)
    assert (a.position, a.realized, a.basis, a.cash) == (0, 12 * S, 0, 12 * S)


def test_short_then_cover() -> None:
    a = Accounting()
    a.fill(SELL, 100, 2, 0)
    a.fill(SELL, 102, 2, 0)
    assert a.position == -4
    a.fill(BUY, 99, 1, 0)  # average short 101, cover one at 99
    assert a.realized == 2 * S
    assert a.unrealized(99 * S) == 6 * S


def test_crossing_through_flat_splits_the_fill() -> None:
    a = Accounting()
    a.fill(BUY, 100, 2, 0)
    a.fill(SELL, 105, 5, 0)
    assert (a.position, a.realized, a.basis) == (-3, 10 * S, -315 * S)
    assert a.unrealized(103 * S) == 6 * S


def test_fees_and_rebates_hit_cash_not_realized() -> None:
    a = Accounting()
    a.fill(BUY, 100, 10, fee_to_money(0.25))
    a.fill(SELL, 100, 10, fee_to_money(-0.1))
    assert a.realized == 0
    assert a.fees == fee_to_money(1.5)
    assert a.net(100 * S) == -fee_to_money(1.5)


def test_identity_holds_with_rounding_in_the_basis() -> None:
    a = Accounting()
    a.fill(BUY, 100, 1, 0)
    a.fill(BUY, 101, 2, 0)
    a.fill(SELL, 107, 1, 7)  # basis 302 / 3 lots does not divide evenly
    mark = mark_to_money(103.5)
    assert a.cash + a.position * mark == a.realized + a.unrealized(mark) - a.fees


def test_short_basis_truncates_toward_zero_like_rust() -> None:
    a = Accounting()
    a.fill(SELL, 100, 1, 0)
    a.fill(SELL, 101, 2, 0)  # basis -302, not divisible by 3
    a.fill(BUY, 99, 1, 0)
    # Removes -302e6 / 3 = -100_666_666.67 of basis. Rust truncates that to
    # -100_666_666; Python's floor division would give -100_666_667.
    assert a.basis == -201_333_334


def test_fee_rounding_matches_rust() -> None:
    assert fee_to_money(-0.1) == -100_000
    assert fee_to_money(0.0000005) == 1  # half rounds away from zero
    assert fee_to_money(-0.0000005) == -1
