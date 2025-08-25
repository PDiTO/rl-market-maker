"""The PPO objective, split into pieces that can be tested on their own."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolicyLossStats:
    approx_kl: float
    """``E[(r - 1) - log r]``, an unbiased, non-negative estimator of KL(old||new)."""
    clip_fraction: float


def clipped_policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, PolicyLossStats]:
    """Negative clipped surrogate, to be minimised::

        L = -E[min(r A, clip(r, 1 - eps, 1 + eps) A)],  r = pi_new / pi_old

    Once the ratio has moved more than ``eps`` in the direction that helps the
    advantage, the clipped term is the minimum and is constant, so that sample stops
    contributing gradient.
    """
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
        clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
    return loss, PolicyLossStats(approx_kl, clip_fraction)


def value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float | None,
) -> torch.Tensor:
    """Half mean squared error, optionally with PPO's value clipping::

        V_clip = V_old + clip(V_new - V_old, -eps, eps)
        L = 0.5 * E[max((V_new - R)^2, (V_clip - R)^2)]

    The max makes clipping pessimistic: it only bites when the unclipped update
    would have moved the value further than ``eps`` and lowered the loss by doing so.
    """
    unclipped = (new_values - returns) ** 2
    if clip_coef is None:
        return 0.5 * unclipped.mean()
    v_clipped = old_values + (new_values - old_values).clamp(-clip_coef, clip_coef)
    clipped = (v_clipped - returns) ** 2
    return 0.5 * torch.max(unclipped, clipped).mean()


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """``1 - Var[y - y_pred] / Var[y]``: 1 is perfect, 0 is no better than the mean."""
    var_y = y_true.var()
    if var_y.item() == 0.0:
        return float("nan")
    return float(1.0 - (y_true - y_pred).var() / var_y)
