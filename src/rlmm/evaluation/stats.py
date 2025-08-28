"""Summary statistics with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Summary:
    mean: float
    std: float
    lo: float
    """Lower end of the 95% bootstrap CI for the mean."""
    hi: float
    n: int

    def fmt(self, digits: int = 1) -> str:
        d = digits
        return f"{self.mean:.{d}f} ± {self.std:.{d}f} [{self.lo:.{d}f}, {self.hi:.{d}f}]"


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Deterministic for a given seed."""
    a = np.asarray(x, dtype=np.float64)
    if a.size < 2:
        m = float(a.mean()) if a.size else float("nan")
        return m, m
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def summarize(x: Sequence[float] | np.ndarray, seed: int = 0) -> Summary:
    a = np.asarray(x, dtype=np.float64)
    lo, hi = bootstrap_ci(a, seed=seed)
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return Summary(float(a.mean()), std, lo, hi, int(a.size))
