import numpy as np
import pytest

from rlmm.ppo.normalize import ObsNormalizer, RewardNormalizer, RunningMeanStd


def test_running_stats_equal_full_batch_stats() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(1000, 4)) * np.array([1, 10, 0.1, 5])
    rms = RunningMeanStd((4,), epsilon=1e-12)
    for chunk in np.array_split(data, [1, 7, 100, 101, 555]):
        rms.update(chunk)
    assert rms.mean == pytest.approx(data.mean(axis=0), rel=1e-9)
    assert rms.var == pytest.approx(data.var(axis=0), rel=1e-9)
    assert rms.count == pytest.approx(1000)


def test_state_round_trip() -> None:
    rms = RunningMeanStd((2,))
    rms.update(np.array([[1.0, 2.0], [3.0, 5.0]]))
    other = RunningMeanStd((2,))
    other.load_state_dict(rms.state_dict())
    assert np.array_equal(other.mean, rms.mean)
    assert other.count == rms.count


def test_obs_normalizer_standardises_and_clips() -> None:
    rng = np.random.default_rng(1)
    norm = ObsNormalizer((3,), clip=5.0)
    x = rng.normal([10.0, -4.0, 0.0], [2.0, 0.5, 1.0], size=(5000, 3))
    out = norm(x)
    assert out.mean(axis=0) == pytest.approx([0, 0, 0], abs=0.02)
    assert out.std(axis=0) == pytest.approx([1, 1, 1], abs=0.02)
    assert np.abs(norm(np.array([[1e6, 0.0, 0.0]]), update=False)).max() == 5.0


def test_reward_normalizer_scales_by_discounted_return_std() -> None:
    gamma = 0.9
    norm = RewardNormalizer(num_envs=1, gamma=gamma, clip=100.0)
    rewards = [1.0, 2.0, -1.0, 0.5]
    dones = [False, True, False, False]
    g, returns, outs = 0.0, [], []
    for r, d in zip(rewards, dones, strict=True):
        outs.append(norm(np.array([r]), np.array([d]))[0])
        g = g * gamma + r
        returns.append(g)
        rms = RunningMeanStd(())
        rms.update(np.array(returns))
        assert outs[-1] == pytest.approx(r / np.sqrt(rms.var + 1e-8), rel=1e-3)
        if d:
            g = 0.0
    assert norm.returns[0] == pytest.approx(0.5 - 1.0 * gamma)
