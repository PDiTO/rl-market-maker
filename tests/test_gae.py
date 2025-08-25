"""GAE against hand-computed values."""

import numpy as np
import pytest

from rlmm.ppo.gae import compute_gae

GAMMA, LAM = 0.9, 0.8
R = np.array([[1.0], [2.0], [3.0]])
V = np.array([[0.5], [1.0], [1.5]])
NEXT_V = np.array([[1.0], [1.5], [2.0]])
NONE = np.zeros((3, 1), dtype=bool)


def test_no_episode_boundary() -> None:
    adv, ret = compute_gae(R, V, NEXT_V, NONE, NONE, GAMMA, LAM)
    # delta = 1.4, 2.35, 3.3; A_t = delta_t + 0.72 * A_{t+1}
    assert adv[:, 0] == pytest.approx([4.80272, 4.726, 3.3])
    assert ret[:, 0] == pytest.approx(adv[:, 0] + V[:, 0])


def test_termination_drops_bootstrap_and_stops_propagation() -> None:
    term = np.array([[False], [True], [False]])
    next_v = NEXT_V.copy()
    next_v[1] = 99.0  # value of a terminal state must be ignored
    adv, _ = compute_gae(R, V, next_v, term, term, GAMMA, LAM)
    # delta_1 = 2 - 1 = 1; A_0 = 1.4 + 0.72 * 1
    assert adv[:, 0] == pytest.approx([2.12, 1.0, 3.3])


def test_truncation_bootstraps_from_final_observation() -> None:
    term = np.zeros((3, 1), dtype=bool)
    end = np.array([[False], [True], [False]])
    next_v = NEXT_V.copy()
    next_v[1] = 5.0  # V(final obs) of the truncated episode
    adv, _ = compute_gae(R, V, next_v, term, end, GAMMA, LAM)
    # delta_1 = 2 + 0.9 * 5 - 1 = 5.5; no carry from the next episode
    assert adv[:, 0] == pytest.approx([1.4 + 0.72 * 5.5, 5.5, 3.3])


def test_lambda_one_gives_discounted_returns() -> None:
    _, ret = compute_gae(R, V, NEXT_V, NONE, NONE, gamma=0.5, lam=1.0)
    # Monte Carlo with bootstrap from V(s_3) = 2: 1 + .5*2 + .25*3 + .125*2
    assert ret[0, 0] == pytest.approx(1 + 1 + 0.75 + 0.25)
    assert ret[2, 0] == pytest.approx(3 + 0.5 * 2)


def test_lambda_zero_is_one_step_td() -> None:
    adv, _ = compute_gae(R, V, NEXT_V, NONE, NONE, GAMMA, lam=0.0)
    assert adv[:, 0] == pytest.approx([1.4, 2.35, 3.3])


def test_envs_are_independent() -> None:
    r = np.hstack([R, R[::-1]])
    v = np.hstack([V, V])
    nv = np.hstack([NEXT_V, NEXT_V])
    none = np.zeros((3, 2), dtype=bool)
    adv, _ = compute_gae(r, v, nv, none, none, GAMMA, LAM)
    single, _ = compute_gae(R, V, NEXT_V, NONE, NONE, GAMMA, LAM)
    assert adv[:, 0] == pytest.approx(single[:, 0])
    assert not np.allclose(adv[:, 1], adv[:, 0])
