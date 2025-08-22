from __future__ import annotations

from typing import Any

import lob
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rlmm.env import EnvConfig, LobMarketMakingEnv
from rlmm.env.features import FEATURE_NAMES, OBS_DIM
from rlmm.env.market_making import level_price, liquidation_cost
from rlmm.sim.accounting import mark_to_money, money_to_ticks

SHORT = EnvConfig(horizon_s=5.0, warmup_s=2.0)


def rollout(
    env: LobMarketMakingEnv, seed: int, actions: list[tuple[int, int]] | None = None
) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]]]:
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    observations, rewards, infos = [obs], [], []
    done, k = False, 0
    while not done:
        a = np.array(actions[k % len(actions)]) if actions else rng.integers(0, 5, size=2)
        obs, r, term, trunc, info = env.step(a)
        observations.append(obs)
        rewards.append(float(r))
        infos.append(info)
        done = term or trunc
        k += 1
    return observations, rewards, infos


def test_gymnasium_env_checker() -> None:
    check_env(LobMarketMakingEnv(SHORT), skip_render_check=True)


def test_episode_length_and_termination() -> None:
    env = LobMarketMakingEnv(SHORT)
    _, rewards, infos = rollout(env, 0)
    assert len(rewards) == SHORT.n_steps == 50
    assert "episode_stats" in infos[-1]
    with pytest.raises(RuntimeError):
        env.step(np.array([2, 2]))


def test_same_seed_same_episode() -> None:
    a_obs, a_rew, _ = rollout(LobMarketMakingEnv(SHORT), 5)
    b_obs, b_rew, _ = rollout(LobMarketMakingEnv(SHORT), 5)
    assert a_rew == b_rew
    assert all(np.array_equal(x, y) for x, y in zip(a_obs, b_obs, strict=True))
    _, c_rew, _ = rollout(LobMarketMakingEnv(SHORT), 6)
    assert c_rew != a_rew


def test_feed_seed_option_pins_the_market() -> None:
    env = LobMarketMakingEnv(SHORT)
    o1, i1 = env.reset(seed=1, options={"feed_seed": 42})
    o2, i2 = env.reset(seed=2, options={"feed_seed": 42})
    assert i1["feed_seed"] == i2["feed_seed"] == 42
    assert np.array_equal(o1, o2)


def test_observations_stay_in_bounds() -> None:
    env = LobMarketMakingEnv(SHORT.replace(horizon_s=20.0))
    assert env.observation_space.shape == (OBS_DIM,) == (len(FEATURE_NAMES),)
    for seed in range(3):
        obs, _, _ = rollout(env, seed)
        for o in obs:
            assert env.observation_space.contains(o), o
            assert np.all(np.isfinite(o))


def test_reward_is_pnl_change_minus_penalties_on_scripted_episode() -> None:
    cfg = SHORT.replace(horizon_s=10.0, reward={"inventory_penalty": 0.05, "dampening": 0.3})
    env = LobMarketMakingEnv(cfg, record=True)
    # Join both sides, then lean on one side, then pull quotes.
    script = [(2, 2)] * 30 + [(4, 1)] * 30 + [(0, 0)] * 10 + [(2, 3)] * 30
    _, rewards, infos = rollout(env, 3, script)
    trace = env.trace
    dt = cfg.step_ms / 1000.0

    prev_pnl, prev_mid, prev_q = 0.0, None, 0
    for k, (r, info) in enumerate(zip(rewards, infos, strict=True)):
        q, mid, pnl = trace.position[k], trace.mid[k], trace.pnl[k]
        assert info["pnl_change"] == pytest.approx(pnl - prev_pnl, abs=1e-9)
        assert info["inventory_penalty"] == pytest.approx(0.05 * q * q * dt)
        if prev_mid is not None:
            assert info["dampening"] == pytest.approx(0.3 * max(0.0, prev_q * (mid - prev_mid)))
        expected = (
            info["pnl_change"]
            - info["inventory_penalty"]
            - info["dampening"]
            - info["liquidation_cost"]
        )
        assert r == pytest.approx(expected, abs=1e-9)
        prev_pnl, prev_mid, prev_q = pnl, mid, q

    # PnL is rebuilt from the fill log alone: cash flows, rebates, mark at mid.
    fills = trace.fills
    assert len(fills) > 10
    q = sum(f.side * f.qty for f in fills)
    cash = sum(-f.side * f.qty * f.price for f in fills) + 0.1 * sum(f.qty for f in fills)
    assert q == trace.position[-1]
    assert cash + q * trace.mid[-1] == pytest.approx(trace.pnl[-1], abs=1e-6)
    assert sum(i["pnl_change"] for i in infos) == pytest.approx(trace.pnl[-1], abs=1e-6)
    assert infos[-1]["liquidation_cost"] == pytest.approx(trace.liquidation_cost)
    if q != 0:
        assert trace.liquidation_cost > 0


def test_inventory_and_cash_identity_against_simulator_accounting() -> None:
    env = LobMarketMakingEnv(SHORT.replace(horizon_s=20.0))
    rollout(env, 9, [(2, 2), (1, 2), (2, 1), (3, 3)])
    acct = env.engine.acct
    mark = mark_to_money(env.engine.last_mid or 0.0)
    assert acct.cash + acct.position * mark == acct.realized + acct.unrealized(mark) - acct.fees
    assert acct.position == sum(f.side * f.qty for f in env.engine.fills)
    assert money_to_ticks(acct.fees) == pytest.approx(-0.1 * sum(f.qty for f in env.engine.fills))


def test_position_limit_is_respected() -> None:
    cfg = SHORT.replace(horizon_s=30.0, max_position=10, quote_size=5)
    env = LobMarketMakingEnv(cfg, record=True)
    rollout(env, 1, [(1, 0)])  # only ever bid, aggressively
    # Known position gates new bids; in-flight fills can overshoot by one quote.
    assert max(env.trace.position) <= cfg.max_position + cfg.quote_size
    assert max(env.trace.position) >= cfg.max_position


def test_level_prices() -> None:
    # spread of 2: improve steps inside, the rest are at or behind the touch
    assert level_price("bid", 0, 100, 102) is None
    assert level_price("bid", 1, 100, 102) == 101
    assert level_price("bid", 2, 100, 102) == 100
    assert level_price("bid", 4, 100, 102) == 98
    assert level_price("ask", 1, 100, 102) == 101
    assert level_price("ask", 3, 100, 102) == 103
    # spread of 1: improve would cross, so it joins
    assert level_price("bid", 1, 100, 101) == 100
    assert level_price("ask", 1, 100, 101) == 101


def test_liquidation_cost_walks_the_book() -> None:
    book = lob.OrderBook()
    book.limit(1, "bid", 99, 3, owner=1)
    book.limit(2, "bid", 98, 4, owner=1)
    book.limit(3, "ask", 101, 5, owner=1)
    # mid 100: sell 5 = 3 lots at 1 tick + 2 lots at 2 ticks, plus 0.3 per lot
    assert liquidation_cost(book, 5, 0.3) == pytest.approx(3 * 1 + 2 * 2 + 1.5)
    # buy 6: 5 at 1 tick, 1 beyond the visible book at 2 ticks
    assert liquidation_cost(book, -6, 0.0) == pytest.approx(5 + 2)
    assert liquidation_cost(book, 0, 0.3) == 0.0
