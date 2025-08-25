"""Gradient behaviour of the clipped surrogate and the clipped value loss."""

import torch

from rlmm.ppo.losses import clipped_policy_loss, explained_variance, value_loss

EPS = 0.2


def grad_wrt_logp(ratio: float, adv: float) -> float:
    new = torch.tensor([float(torch.log(torch.tensor(ratio)))], requires_grad=True)
    loss, _ = clipped_policy_loss(new, torch.zeros(1), torch.tensor([adv]), EPS)
    loss.backward()
    assert new.grad is not None
    return float(new.grad[0])


def test_ratio_one_is_the_policy_gradient() -> None:
    new = torch.zeros(4, requires_grad=True)
    adv = torch.tensor([1.0, -2.0, 0.5, 3.0])
    loss, stats = clipped_policy_loss(new, torch.zeros(4), adv, EPS)
    assert torch.isclose(loss, -adv.mean())
    loss.backward()
    assert new.grad is not None
    assert torch.allclose(new.grad, -adv / 4)
    assert stats.approx_kl == 0.0
    assert stats.clip_fraction == 0.0


def test_gradient_vanishes_once_the_ratio_has_moved_far_enough() -> None:
    # Good action already made much more likely: no further push.
    assert grad_wrt_logp(1.5, +1.0) == 0.0
    # Bad action already made much less likely: no further push.
    assert grad_wrt_logp(0.5, -1.0) == 0.0


def test_gradient_survives_when_the_ratio_moved_the_wrong_way() -> None:
    # The min keeps the pessimistic unclipped term, so these still get corrected.
    assert grad_wrt_logp(0.5, +1.0) < 0.0  # d/dlogp of -r*A = -r*A
    assert grad_wrt_logp(1.5, -1.0) > 0.0


def test_gradient_inside_the_trust_region() -> None:
    g = grad_wrt_logp(1.1, 2.0)
    assert abs(g - (-1.1 * 2.0)) < 1e-5


def test_stats() -> None:
    new = torch.log(torch.tensor([1.0, 1.5, 0.7, 1.1]))
    _, stats = clipped_policy_loss(new, torch.zeros(4), torch.ones(4), EPS)
    assert stats.clip_fraction == 0.5
    r = torch.tensor([1.0, 1.5, 0.7, 1.1])
    assert abs(stats.approx_kl - float(((r - 1) - r.log()).mean())) < 1e-6
    assert stats.approx_kl > 0


def test_value_clipping_blocks_large_improving_moves() -> None:
    old, ret = torch.zeros(1), torch.ones(1)
    new = torch.tensor([0.5], requires_grad=True)
    loss = value_loss(new, old, ret, clip_coef=0.2)
    # unclipped 0.25, clipped (0.2 - 1)^2 = 0.64, max = 0.64
    assert torch.isclose(loss, torch.tensor(0.32))
    loss.backward()
    assert new.grad is not None and new.grad.item() == 0.0


def test_value_clipping_keeps_gradient_when_moving_away() -> None:
    old, ret = torch.zeros(1), torch.ones(1)
    new = torch.tensor([-0.5], requires_grad=True)
    loss = value_loss(new, old, ret, clip_coef=0.2)
    assert torch.isclose(loss, torch.tensor(0.5 * 2.25))
    loss.backward()
    assert new.grad is not None and abs(new.grad.item() - (-1.5)) < 1e-6


def test_unclipped_value_loss() -> None:
    new = torch.tensor([0.5], requires_grad=True)
    loss = value_loss(new, torch.zeros(1), torch.ones(1), clip_coef=None)
    assert torch.isclose(loss, torch.tensor(0.125))


def test_explained_variance() -> None:
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(y, y) == 1.0
    assert abs(explained_variance(torch.full((4,), 2.5), y)) < 1e-6
