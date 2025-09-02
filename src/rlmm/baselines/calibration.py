"""Fitting the Avellaneda-Stoikov inputs to the simulator, and tuning baselines.

Everything here runs on calibration seeds, a range disjoint from training and
evaluation markets.

* ``sigma``: realised volatility of the exchange mid from non-overlapping increments
  at several horizons. At short horizons the mid is noisier than the latent fair
  value (bid-ask bounce, discrete ticks); the AS model wants the diffusive part, so
  the default is the 5-second estimate.
* ``A`` and ``k``: the model's fill intensity ``lambda(delta) = A exp(-k delta)``.
  Rather than infer it from trade prints, which would ignore queue position, a
  probe strategy quotes a fixed number of ticks behind the touch on both sides in
  the full simulator and we measure how fast its quotes actually fill. ``delta`` is
  the time-averaged distance of the quote from the mid, the rate is filled lots per
  lot quoted per second, and ``log(rate) = log(A) - k delta`` is fitted by least
  squares.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import lob
import numpy as np

from rlmm.env.config import CALIBRATION_SEED_BASE, EnvConfig
from rlmm.env.market_making import LobMarketMakingEnv
from rlmm.evaluation.runner import Job, Quoter, QuoterPolicy, run_jobs
from rlmm.evaluation.stats import summarize
from rlmm.sim.engine import ReplayEngine

VOL_HORIZONS_S = (0.1, 1.0, 5.0, 10.0)


def calibration_seeds(n: int, offset: int = 0) -> list[int]:
    return [CALIBRATION_SEED_BASE + offset + i for i in range(n)]


def mid_series(cfg: EnvConfig, feed_seed: int, dt_s: float = 0.1) -> np.ndarray:
    """Exchange mid sampled every ``dt_s`` over the session, with no orders of ours."""
    duration = cfg.warmup_s + cfg.horizon_s + 1.0
    engine = ReplayEngine(lob.generate_feed(**cfg.market.to_lob(feed_seed, duration)), cfg.sim)
    step = round(dt_s * 1e9)
    mids = []
    t = cfg.warmup_ns
    for _ in range(round(cfg.horizon_s / dt_s) + 1):
        engine.schedule_timer(t)
        if engine.run_to_timer() is None:
            break
        mids.append(engine.last_mid if engine.last_mid is not None else np.nan)
        t += step
    return np.asarray(mids, dtype=np.float64)


def estimate_volatility(
    cfg: EnvConfig, seeds: Sequence[int], dt_s: float = 0.1
) -> dict[float, float]:
    """Volatility in ticks per sqrt(second) at each horizon in ``VOL_HORIZONS_S``."""
    sq: dict[float, list[float]] = {h: [] for h in VOL_HORIZONS_S}
    for s in seeds:
        m = mid_series(cfg, s, dt_s)
        for h in VOL_HORIZONS_S:
            k = round(h / dt_s)
            d = m[::k]
            d = np.diff(d[np.isfinite(d)])
            sq[h].extend((d * d).tolist())
    return {h: float(np.sqrt(np.mean(v) / h)) for h, v in sq.items()}


@dataclass(frozen=True)
class IntensityPoint:
    ticks_behind: int
    distance: float
    """Time-averaged distance of the quote from the mid, ticks."""
    rate: float
    """Filled lots per quoted lot per second."""
    exposure_s: float
    filled_lots: int


def probe_intensity(
    cfg: EnvConfig, seeds: Sequence[int], ticks_behind: Sequence[int] = (0, 1, 2, 3, 4)
) -> list[IntensityPoint]:
    probe_cfg = cfg.replace(max_position=10**9)
    env = LobMarketMakingEnv(probe_cfg)
    step_s = cfg.step_ms / 1000.0
    points = []
    for L in ticks_behind:
        dist_sum = 0.0
        exposure = 0.0
        lots = 0
        for s in seeds:
            env.reset(seed=s, options={"feed_seed": s})
            done = False
            while not done:
                v = env.market_view()
                bid = ask = None
                if v.best_bid is not None and v.best_ask is not None and v.mid is not None:
                    bid, ask = v.best_bid[0] - L, v.best_ask[0] + L
                    dist_sum += (v.mid - bid) + (ask - v.mid)
                    exposure += 2 * step_s
                _, _, term, trunc, _ = env.step_quotes(bid, ask)
                done = term or trunc
            lots += sum(f.qty for f in env.engine.fills)
        points.append(
            IntensityPoint(
                ticks_behind=L,
                distance=dist_sum / (exposure / step_s),
                rate=lots / (cfg.quote_size * exposure),
                exposure_s=exposure,
                filled_lots=lots,
            )
        )
    return points


def fit_intensity(points: Sequence[IntensityPoint]) -> tuple[float, float]:
    """Weighted least-squares fit of ``log(rate) = log(A) - k * distance``.

    Fill counts are roughly Poisson, so the variance of ``log(rate)`` is about
    ``1 / filled_lots``; each point is weighted by its count. Without the weights a
    far-away level with five fills counts as much as the touch with thousands.
    Returns ``(A, k)``.
    """
    pts = [p for p in points if p.filled_lots > 0]
    x = np.array([p.distance for p in pts])
    y = np.log([p.rate for p in pts])
    w = np.sqrt([p.filled_lots for p in pts])  # polyfit squares the weights
    slope, intercept = np.polyfit(x, y, 1, w=w)
    return float(np.exp(intercept)), float(-slope)


@dataclass(frozen=True)
class GridPoint:
    param: float
    reward: float
    pnl: float
    pnl_std: float
    mean_abs_inventory: float


def tune(
    make: Callable[[float], Quoter],
    grid: Sequence[float],
    cfg: EnvConfig,
    seeds: Sequence[int],
    workers: int | None = None,
) -> list[GridPoint]:
    """Evaluates a one-parameter family of quoters on calibration seeds."""
    jobs = [Job(str(p), QuoterPolicy(make(p)), cfg, tuple(seeds)) for p in grid]
    res = run_jobs(jobs, workers=workers)
    out = []
    for p in grid:
        ms = res[str(p)]
        pnl = summarize([m.pnl for m in ms])
        out.append(
            GridPoint(
                param=p,
                reward=float(np.mean([m.reward for m in ms])),
                pnl=pnl.mean,
                pnl_std=pnl.std,
                mean_abs_inventory=float(np.mean([m.mean_abs_inventory for m in ms])),
            )
        )
    return out


def best(points: Sequence[GridPoint]) -> GridPoint:
    """Picks by mean training reward: the same objective the PPO agent maximises."""
    return max(points, key=lambda g: g.reward)


AS_GAMMA_GRID = (0.0003, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05)
GLFT_GAMMA_GRID = (0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
SKEW_KAPPA_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2)
SYMMETRIC_OFFSET_GRID = (0.0, 1.0, 2.0)
SIGMA_HORIZON_S = 5.0


@dataclass
class Calibration:
    sigma: float
    """Volatility used by the AS quoters, ticks per sqrt(second)."""
    sigma_by_horizon: dict[str, float]
    A: float
    k: float
    intensity: list[IntensityPoint]
    chosen: dict[str, float]
    """Selected parameter per baseline family."""
    grids: dict[str, list[GridPoint]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Calibration:
        return Calibration(
            sigma=d["sigma"],
            sigma_by_horizon=d["sigma_by_horizon"],
            A=d["A"],
            k=d["k"],
            intensity=[IntensityPoint(**p) for p in d["intensity"]],
            chosen=d["chosen"],
            grids={k: [GridPoint(**g) for g in v] for k, v in d["grids"].items()},
        )


def run_calibration(
    cfg: EnvConfig,
    vol_episodes: int = 50,
    probe_episodes: int = 30,
    tune_episodes: int = 200,
    workers: int | None = None,
    log: Callable[[str], None] = print,
) -> Calibration:
    """Estimates sigma, A and k, then tunes each baseline's free parameter."""
    from rlmm.baselines.avellaneda_stoikov import (
        ASParams,
        AvellanedaStoikov,
        GueantLehalleFernandezTapia,
    )
    from rlmm.baselines.heuristics import LinearSkewQuoter, SymmetricQuoter

    vol = estimate_volatility(cfg, calibration_seeds(vol_episodes))
    sigma = vol[SIGMA_HORIZON_S]
    log(f"sigma by horizon: {vol}; using {sigma:.3f}")
    points = probe_intensity(cfg, calibration_seeds(probe_episodes, offset=1_000))
    A, k = fit_intensity(points)
    log(f"intensity fit: A={A:.3f}, k={k:.3f}")

    seeds = calibration_seeds(tune_episodes, offset=2_000)
    families: dict[str, tuple[Callable[[float], Quoter], Sequence[float]]] = {
        "symmetric": (lambda o: SymmetricQuoter(int(o)), SYMMETRIC_OFFSET_GRID),
        "linear_skew": (lambda kap: LinearSkewQuoter(kap), SKEW_KAPPA_GRID),
        "microprice_skew": (
            lambda kap: LinearSkewQuoter(kap, reference="microprice", name="microprice_skew"),
            SKEW_KAPPA_GRID,
        ),
        "avellaneda_stoikov": (
            lambda g: AvellanedaStoikov(ASParams(g, sigma, k)),
            AS_GAMMA_GRID,
        ),
        "as_stationary": (
            lambda g: GueantLehalleFernandezTapia(ASParams(g, sigma, k), A),
            GLFT_GAMMA_GRID,
        ),
    }
    grids: dict[str, list[GridPoint]] = {}
    chosen: dict[str, float] = {}
    for name, (make, grid) in families.items():
        grids[name] = tune(make, grid, cfg, seeds, workers)
        chosen[name] = best(grids[name]).param
        log(f"{name}: chose {chosen[name]}")
    return Calibration(
        sigma=sigma,
        sigma_by_horizon={str(h): v for h, v in vol.items()},
        A=A,
        k=k,
        intensity=points,
        chosen=chosen,
        grids=grids,
    )


def baselines_from(cal: Calibration) -> list[Quoter]:
    """The tuned baselines."""
    from rlmm.baselines.avellaneda_stoikov import (
        ASParams,
        AvellanedaStoikov,
        GueantLehalleFernandezTapia,
    )
    from rlmm.baselines.heuristics import LinearSkewQuoter, SymmetricQuoter

    c = cal.chosen
    return [
        SymmetricQuoter(int(c["symmetric"])),
        LinearSkewQuoter(c["linear_skew"]),
        LinearSkewQuoter(c["microprice_skew"], reference="microprice", name="microprice_skew"),
        AvellanedaStoikov(ASParams(c["avellaneda_stoikov"], cal.sigma, cal.k)),
        GueantLehalleFernandezTapia(ASParams(c["as_stationary"], cal.sigma, cal.k), cal.A),
    ]
