from __future__ import annotations

import numpy as np
import pytest

from rlmm.baselines import LinearSkewQuoter, SymmetricQuoter
from rlmm.env import EnvConfig, LobMarketMakingEnv
from rlmm.evaluation.metrics import episode_metrics
from rlmm.evaluation.report import paired_difference, summary_rows
from rlmm.evaluation.runner import QuoterPolicy, evaluate, run_episode
from rlmm.evaluation.stats import bootstrap_ci, summarize

CFG = EnvConfig(horizon_s=20.0, warmup_s=2.0)


def test_pnl_decomposition_adds_up() -> None:
    env = LobMarketMakingEnv(CFG, record=True)
    for seed in (1, 2, 3):
        trace = run_episode(env, QuoterPolicy(SymmetricQuoter(0)), seed)
        m = episode_metrics(trace, seed, 0.1, CFG.warmup_s, CFG.horizon_s)
        assert m.fills > 0
        assert m.spread_capture + m.inventory_pnl + m.rebates == pytest.approx(m.pnl_mtm, abs=1e-6)
        assert m.pnl == pytest.approx(m.pnl_mtm - m.liquidation_cost)
        assert m.rebates == pytest.approx(0.1 * m.filled_lots)
        assert 0 < m.fill_rate <= 1
        assert m.bid_quoted_share == 1.0 and m.ask_quoted_share == 1.0
        # Passive fills at the touch are at or better than mid.
        assert m.spread_capture > 0


def test_evaluation_is_deterministic() -> None:
    a = evaluate(QuoterPolicy(LinearSkewQuoter(0.05)), CFG, [10, 11])
    b = evaluate(QuoterPolicy(LinearSkewQuoter(0.05)), CFG, [10, 11])
    assert a == b


def test_bootstrap_ci() -> None:
    x = np.random.default_rng(0).normal(5.0, 2.0, size=400)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi
    # Close to the normal-theory interval for this sample size.
    half = 1.96 * x.std(ddof=1) / np.sqrt(x.size)
    assert (hi - lo) / 2 == pytest.approx(half, rel=0.1)
    assert bootstrap_ci(x) == (lo, hi)
    s = summarize(x)
    assert s.n == 400 and s.lo == lo


def test_paired_difference_and_summary() -> None:
    res = {
        "a": evaluate(QuoterPolicy(SymmetricQuoter(0)), CFG, [1, 2, 3]),
        "b": evaluate(QuoterPolicy(SymmetricQuoter(1)), CFG, [1, 2, 3]),
    }
    d = paired_difference(res, "a", "b")
    expected = np.mean([x.pnl - y.pnl for x, y in zip(res["a"], res["b"], strict=True)])
    assert d.mean == pytest.approx(expected)
    rows = summary_rows(res)
    assert [r["policy"] for r in rows] == ["a", "b"]
    assert rows[0]["pnl_mean"] == pytest.approx(np.mean([m.pnl for m in res["a"]]))
