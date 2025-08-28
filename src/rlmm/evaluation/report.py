"""Tables from evaluation results: CSV for machines, markdown for the README."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rlmm.evaluation.metrics import EpisodeMetrics
from rlmm.evaluation.stats import Summary, summarize

Results = dict[str, list[EpisodeMetrics]]

SUMMARY_FIELDS = (
    "pnl",
    "sharpe",
    "max_drawdown",
    "mean_abs_inventory",
    "max_abs_inventory",
    "fills",
    "fill_rate",
    "markout_0_1s",
    "markout_1s",
    "markout_5s",
    "spread_capture",
    "inventory_pnl",
    "rebates",
    "liquidation_cost",
    "reward",
    "bid_quoted_share",
    "ask_quoted_share",
)


def summary_rows(results: Results) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, ms in results.items():
        row: dict[str, float | str] = {"policy": name, "episodes": float(len(ms))}
        for f in SUMMARY_FIELDS:
            s = summarize([getattr(m, f) for m in ms])
            row[f"{f}_mean"] = s.mean
            row[f"{f}_std"] = s.std
            row[f"{f}_ci_lo"] = s.lo
            row[f"{f}_ci_hi"] = s.hi
        pnl = np.array([m.pnl for m in ms])
        row["pnl_sharpe_across_episodes"] = float(pnl.mean() / pnl.std(ddof=1))
        filled = sum(m.filled_lots for m in ms)
        for h in ("markout_0_1s", "markout_1s", "markout_5s"):
            # Lot-weighted across episodes rather than a mean of per-episode ratios.
            row[f"{h}_per_lot"] = (
                float(sum(getattr(m, h) * m.filled_lots for m in ms) / filled) if filled else 0.0
            )
        rows.append(row)
    return rows


def paired_difference(results: Results, a: str, b: str, field: str = "pnl") -> Summary:
    """Mean of ``a - b`` over the same markets, with a bootstrap CI."""
    xa = {m.feed_seed: getattr(m, field) for m in results[a]}
    xb = {m.feed_seed: getattr(m, field) for m in results[b]}
    common = sorted(set(xa) & set(xb))
    return summarize([xa[s] - xb[s] for s in common])


def write_csv(path: str | Path, rows: Sequence[dict[str, float | str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def write_episodes_csv(path: str | Path, results: Results) -> None:
    rows: list[dict[str, float | str]] = []
    for name, ms in results.items():
        for m in ms:
            rows.append({"policy": name, **m.as_dict()})
    write_csv(path, rows)


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _ci(row: dict[str, float | str], f: str, d: int = 1) -> str:
    return f"{row[f + '_mean']:.{d}f} [{row[f + '_ci_lo']:.{d}f}, {row[f + '_ci_hi']:.{d}f}]"


def _ms(row: dict[str, float | str], f: str, d: int = 1) -> str:
    return f"{row[f + '_mean']:.{d}f} ± {row[f + '_std']:.{d}f}"


def main_tables(rows: Sequence[dict[str, float | str]]) -> str:
    perf = markdown_table(
        [
            "policy",
            "PnL mean [95% CI]",
            "PnL sd",
            "PnL mean/sd",
            "Sharpe, 1 s",
            "max drawdown",
            "mean abs inventory",
            "reward [95% CI]",
        ],
        [
            [
                str(r["policy"]),
                _ci(r, "pnl"),
                f"{r['pnl_std']:.1f}",
                f"{r['pnl_sharpe_across_episodes']:.2f}",
                _ms(r, "sharpe", 3),
                _ms(r, "max_drawdown"),
                _ms(r, "mean_abs_inventory"),
                _ci(r, "reward"),
            ]
            for r in rows
        ],
    )
    flow = markdown_table(
        [
            "policy",
            "fills",
            "fill rate",
            "bid / ask quoted",
            "markout 0.1 s",
            "markout 1 s",
            "markout 5 s",
        ],
        [
            [
                str(r["policy"]),
                f"{r['fills_mean']:.0f}",
                f"{r['fill_rate_mean']:.3f}",
                f"{r['bid_quoted_share_mean']:.2f} / {r['ask_quoted_share_mean']:.2f}",
                f"{r['markout_0_1s_per_lot']:+.3f}",
                f"{r['markout_1s_per_lot']:+.3f}",
                f"{r['markout_5s_per_lot']:+.3f}",
            ]
            for r in rows
        ],
    )
    decomp = markdown_table(
        ["policy", "spread capture", "inventory PnL", "rebates", "liquidation", "net PnL"],
        [
            [
                str(r["policy"]),
                _ci(r, "spread_capture"),
                _ci(r, "inventory_pnl"),
                f"{r['rebates_mean']:.1f}",
                f"{-float(r['liquidation_cost_mean']):.1f}",
                f"{r['pnl_mean']:.1f}",
            ]
            for r in rows
        ],
    )
    return "\n\n".join([perf, flow, decomp])
