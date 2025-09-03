"""Figures for the README: learning curves, diagnostics, evaluation and stress tests."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

# Fixed colour per entity, never by rank.
COLORS = {
    "ppo": "#2a78d6",
    "linear_skew": "#eb6834",
    "as_stationary": "#1baf7a",
    "avellaneda_stoikov": "#eda100",
    "symmetric": "#e87ba4",
    "microprice_skew": "#008300",
    "ppo_ablation": "#4a3aa7",
}
SEED_STYLES = ("-", "--", ":", "-.")


def color_for(name: str) -> str:
    if name.startswith("ppo_ablation"):
        return COLORS["ppo_ablation"]
    if name.startswith("ppo"):
        return COLORS["ppo"]
    return COLORS.get(name, TEXT_2)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "font.size": 9,
        }
    )


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _col(rows: list[dict[str, str]], key: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for r in rows:
        if r.get(key, "") != "":
            x.append(float(r["global_step"]))
            y.append(float(r[key]))
    return np.asarray(x), np.asarray(y)


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    if y.size < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    head = c[1:k] / np.arange(1, k)
    return np.concatenate([head, (c[k:] - c[:-k]) / k])


def load_runs(run_dirs: Sequence[tuple[str, Path]]) -> dict[str, list[dict[str, str]]]:
    """``(label, directory)`` pairs to ``{label: metrics rows}``."""
    return {label: read_csv(d / "metrics.csv") for label, d in run_dirs}


def learning_curves(
    runs: dict[str, list[dict[str, str]]],
    baselines: dict[str, float],
    path: Path,
    title: str = "PPO learning curves",
) -> None:
    """Training episode PnL (stochastic policy) and validation PnL (greedy policy)
    against environment steps, with baseline means as reference lines."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = (
        ("episode_pnl", "Training episodes (sampled actions)", 20),
        ("val_pnl", "Validation markets (greedy actions)", 3),
    )
    for ax, (key, sub, k) in zip(axes, panels, strict=True):
        for i, (label, rows) in enumerate(runs.items()):
            x, y = _col(rows, key)
            if x.size == 0:
                continue
            ax.plot(
                x / 1e6,
                _smooth(y, k),
                color=color_for(label),
                linestyle=SEED_STYLES[i % len(SEED_STYLES)],
                label=label,
            )
        for name, v in baselines.items():
            ax.axhline(v, color=color_for(name), linewidth=1.2, alpha=0.9)
            ax.annotate(
                f"{name} {v:.0f}",
                xy=(1.0, v),
                xycoords=("axes fraction", "data"),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=TEXT_2,
            )
        ax.set_title(sub, loc="left")
        ax.set_xlabel("environment steps (millions)")
    axes[0].set_ylabel("PnL per 60 s episode (ticks x lots)")
    axes[0].legend(loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    _save(fig, path)


def diagnostics(runs: dict[str, list[dict[str, str]]], path: Path) -> None:
    _style()
    keys = (
        ("approx_kl", "approx KL"),
        ("clip_fraction", "clip fraction"),
        ("explained_variance", "explained variance"),
        ("entropy", "policy entropy (nats, two heads)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (key, title) in zip(axes.ravel(), keys, strict=True):
        for i, (label, rows) in enumerate(runs.items()):
            x, y = _col(rows, key)
            ax.plot(
                x / 1e6,
                _smooth(y, 10),
                color=color_for(label),
                linestyle=SEED_STYLES[i % len(SEED_STYLES)],
                linewidth=1.5,
                label=label,
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel("environment steps (millions)")
    axes[0, 0].legend()
    fig.tight_layout()
    _save(fig, path)


def _episodes(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows = read_csv(path)
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r.items():
            if k != "policy":
                out[r["policy"]][k].append(float(v))
    return {p: {k: np.asarray(v) for k, v in d.items()} for p, d in out.items()}


def pnl_distribution(episodes_csv: Path, path: Path) -> None:
    eps = _episodes(episodes_csv)
    names = list(eps)
    _style()
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.5))
    data = [eps[n]["pnl"] for n in names]
    bp = ax.boxplot(
        data,
        vert=False,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": TEXT, "linewidth": 1.5},
        whiskerprops={"color": TEXT_2, "linewidth": 1},
        capprops={"color": TEXT_2, "linewidth": 1},
    )
    for patch, n in zip(bp["boxes"], names, strict=True):
        patch.set_facecolor(color_for(n))
        patch.set_alpha(0.35)
        patch.set_edgecolor(color_for(n))
    for i, n in enumerate(names):
        ax.plot(
            eps[n]["pnl"].mean(),
            i + 1,
            "o",
            color=color_for(n),
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
    ax.set_yticks(range(1, len(names) + 1), names)
    ax.invert_yaxis()
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_xlabel("episode PnL (ticks x lots); box = quartiles, whiskers = 1.5 IQR, dot = mean")
    ax.set_title("PnL per held-out episode", loc="left")
    fig.tight_layout()
    _save(fig, path)


def pnl_decomposition(summary_csv: Path, path: Path) -> None:
    rows = read_csv(summary_csv)
    names = [r["policy"] for r in rows]
    parts = (
        ("spread_capture_mean", "spread capture", "#2a78d6"),
        ("inventory_pnl_mean", "inventory PnL", "#eb6834"),
        ("rebates_mean", "rebates", "#1baf7a"),
        ("liquidation_cost_mean", "liquidation (negated)", "#eda100"),
    )
    _style()
    fig, ax = plt.subplots(figsize=(9, 0.8 * len(names) + 1.5))
    h = 0.18
    y = np.arange(len(names))
    for j, (key, label, c) in enumerate(parts):
        vals = np.array([float(r[key]) for r in rows])
        if key.startswith("liquidation"):
            vals = -vals
        ax.barh(y + (j - 1.5) * (h + 0.02), vals, height=h, color=c, label=label)
    net = np.array([float(r["pnl_mean"]) for r in rows])
    ax.scatter(net, y, marker="D", color=TEXT, s=30, zorder=3, label="net PnL")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_xlabel("mean per episode (ticks x lots)")
    ax.set_title("Where the PnL comes from", loc="left")
    ax.legend(loc="lower right", ncols=5, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _save(fig, path)


def stress(stress_csv: Path, path: Path) -> None:
    rows = read_csv(stress_csv)
    groups = list(dict.fromkeys(r["group"] for r in rows))
    names = list(dict.fromkeys(r["policy"] for r in rows))
    titles = {
        "latency": "Latency each way (trained at 0.05 ms)",
        "volatility": "Fair-value volatility (trained at 2)",
        "informed": "Informed share of market orders (trained at 0.5)",
        "fair_value": "Fair-value process (trained on random walk)",
    }
    _style()
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4), sharey=False)
    axes_list: list[Axes] = list(np.atleast_1d(axes))
    for ax, g in zip(axes_list, groups, strict=True):
        labels = list(dict.fromkeys(r["regime"] for r in rows if r["group"] == g))
        x = np.arange(len(labels))
        for i, n in enumerate(names):
            sel = [
                next(r for r in rows if r["group"] == g and r["regime"] == lab and r["policy"] == n)
                for lab in labels
            ]
            m = np.array([float(r["pnl_mean"]) for r in sel])
            lo = np.array([float(r["pnl_ci_lo"]) for r in sel])
            hi = np.array([float(r["pnl_ci_hi"]) for r in sel])
            off = (i - (len(names) - 1) / 2) * 0.06
            ax.errorbar(
                x + off,
                m,
                yerr=[m - lo, hi - m],
                color=color_for(n),
                marker="o",
                markersize=5,
                linewidth=1.5,
                capsize=0,
                label=n,
                linestyle=SEED_STYLES[i % len(SEED_STYLES)] if n.startswith("ppo") else "-",
            )
        ax.set_xticks(x, labels)
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.set_title(titles.get(g, g), loc="left", fontsize=10)
    axes_list[0].set_ylabel("mean PnL per episode, 95% CI")
    axes_list[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    _save(fig, path)


def inventory_and_behaviour(
    traces: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    inv_path: Path,
    behaviour_path: Path,
    max_position: int,
) -> None:
    """``traces[name]`` holds ``(position, bid_offset, ask_offset)`` arrays per episode."""
    from rlmm.env.market_making import NO_QUOTE

    names = list(traces)
    _style()
    cols = min(3, len(names))
    rws = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(3.6 * cols, 2.6 * rws), sharex=True)
    bins = np.arange(-max_position - 7.5, max_position + 8.5, 5)
    for ax, n in zip(np.atleast_1d(axes).ravel(), names, strict=False):
        pos = np.concatenate([t[0] for t in traces[n]])
        ax.hist(
            pos,
            bins=bins,
            color=color_for(n),
            alpha=0.85,
            rwidth=0.85,
            weights=np.full(pos.size, 1.0 / pos.size),
        )
        ax.set_title(f"{n}  (sd {pos.std():.1f})", loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    for ax in np.atleast_1d(axes).ravel()[len(names) :]:
        ax.set_visible(False)
    fig.supxlabel("inventory, lots (share of decisions)", fontsize=9, color=TEXT_2)
    fig.tight_layout()
    _save(fig, inv_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    edges = np.arange(-40, 45, 5)
    centers = (edges[:-1] + edges[1:]) / 2
    for ax, idx, side in ((axes[0], 1, "bid"), (axes[1], 2, "ask")):
        for i, n in enumerate(names):
            pos = np.concatenate([t[0] for t in traces[n]])
            off = np.concatenate([t[idx] for t in traces[n]]).astype(float)
            # "No quote" plotted as 3 ticks back, the edge of the chart.
            off = np.where(off == NO_QUOTE, 3.0, off)
            which = np.digitize(pos, edges) - 1
            mean = np.array(
                [
                    off[which == b].mean() if np.sum(which == b) >= 30 else np.nan
                    for b in range(len(centers))
                ]
            )
            ax.plot(
                centers,
                mean,
                color=color_for(n),
                marker="o",
                markersize=4,
                linestyle=SEED_STYLES[i % len(SEED_STYLES)] if n.startswith("ppo") else "-",
                label=n,
            )
        ax.set_title(f"{side}: ticks behind the touch (3 = not quoting)", loc="left")
        ax.set_xlabel("inventory before the decision, lots")
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
    axes[0].set_ylabel("mean quote offset (negative = inside the spread)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    _save(fig, behaviour_path)


def calibration_plot(cal: dict[str, object], path: Path) -> None:
    """Probe fill intensity against distance, with the fitted exponential."""
    pts = cal["intensity"]
    assert isinstance(pts, list)
    A = float(cal["A"])  # type: ignore[arg-type]
    k = float(cal["k"])  # type: ignore[arg-type]
    d = np.array([p["distance"] for p in pts])
    r = np.array([p["rate"] for p in pts])
    n = np.array([p["filled_lots"] for p in pts])
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ok = n > 0
    ax.semilogy(
        d[ok],
        r[ok],
        "o",
        color=COLORS["ppo"],
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label="probe quotes",
    )
    xs = np.linspace(d.min() - 0.1, d.max() + 0.1, 50)
    ax.semilogy(
        xs, A * np.exp(-k * xs), color=COLORS["linear_skew"], label=f"fit: {A:.2f} exp(-{k:.2f} d)"
    )
    for di, ri, ni in zip(d[ok], r[ok], n[ok], strict=True):
        ax.annotate(
            f"{int(ni)} lots",
            (di, ri),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7,
            color=TEXT_2,
        )
    ax.set_xlabel("mean distance of the quote from mid (ticks)")
    ax.set_ylabel("fill rate (lots per quoted lot per second)")
    ax.set_title("Fill intensity measured in the simulator", loc="left")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
