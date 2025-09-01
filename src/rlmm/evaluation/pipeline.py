"""``rlmm eval``: the held-out comparison and the robustness sweep."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rlmm.baselines.calibration import Calibration, baselines_from
from rlmm.config import RunConfig
from rlmm.env.config import EnvConfig
from rlmm.evaluation.report import (
    Results,
    main_tables,
    markdown_table,
    paired_difference,
    summary_rows,
    write_csv,
    write_episodes_csv,
)
from rlmm.evaluation.runner import AgentPolicy, Job, Policy, QuoterPolicy, eval_seeds, run_jobs
from rlmm.evaluation.stats import summarize
from rlmm.ppo.agent import Agent

ROBUSTNESS_SEED_OFFSET = 1_000_000


def load_calibration(path: str | Path) -> Calibration:
    return Calibration.from_dict(json.loads(Path(path).read_text()))


def load_policies(run: RunConfig) -> list[tuple[Policy, EnvConfig]]:
    """Tuned baselines on the eval env, then each PPO checkpoint on its own training
    env (step, horizon, size, reward) with the eval market and latency swapped in."""
    env = run.env
    cal = load_calibration(run.eval.calibration)
    out: list[tuple[Policy, EnvConfig]] = [(QuoterPolicy(q), env) for q in baselines_from(cal)]
    for name, path in run.eval.policies.items():
        agent, meta = Agent.load(path)
        cfg = EnvConfig.from_dict(meta["env"]).replace(
            market=asdict(env.market), sim=asdict(env.sim)
        )
        out.append((AgentPolicy(agent, name), cfg))
    return out


def regimes(run: RunConfig) -> list[tuple[str, str, dict[str, dict[str, float]]]]:
    ev = run.eval
    out: list[tuple[str, str, dict[str, dict[str, float]]]] = []
    for lat in ev.latency_ns:
        label = f"{lat / 1e6:g} ms"
        out.append(
            ("latency", label, {"sim": {"order_latency_ns": lat, "market_data_latency_ns": lat}})
        )
    for v in ev.volatility:
        out.append(("volatility", f"{v:g}", {"market": {"volatility": v}}))
    for i in ev.informed:
        out.append(("informed", f"{i:g}", {"market": {"informed": i}}))
    return out


def _workers(run: RunConfig) -> int | None:
    return run.eval.workers or None


def _seed_spread(results: Results, prefix: str = "ppo") -> str:
    names = [n for n in results if n.startswith(prefix)]
    if len(names) < 2:
        return ""
    means = [float(np.mean([m.pnl for m in results[n]])) for n in names]
    return (
        f"Across the {len(names)} training seeds, mean PnL per seed is "
        f"{np.mean(means):.1f} with a between-seed sd of {np.std(means, ddof=1):.1f}."
    )


def run_main(run: RunConfig, log: bool = True) -> Results:
    policies = load_policies(run)
    seeds = tuple(eval_seeds(run.eval.episodes))
    jobs = [Job(p.name, p, cfg, seeds) for p, cfg in policies]
    results = run_jobs(jobs, workers=_workers(run))
    out = Path(run.eval.out_dir)
    rows = summary_rows(results)
    write_csv(out / "eval_summary.csv", rows)
    write_episodes_csv(out / "eval_episodes.csv", results)

    baselines = [p.name for p, _ in policies if isinstance(p, QuoterPolicy)]
    agents = [p.name for p, _ in policies if isinstance(p, AgentPolicy)]
    paired_rows = []
    for a in agents:
        for b in baselines:
            d = paired_difference(results, a, b)
            paired_rows.append([a, b, f"{d.mean:+.1f}", f"[{d.lo:+.1f}, {d.hi:+.1f}]"])
    md = [
        f"Evaluated on {len(seeds)} held-out markets, {run.env.horizon_s:g} s each. "
        "PnL in ticks x lots, net of fees and of flattening the final position. "
        "Brackets are 95% bootstrap confidence intervals for the mean.",
        main_tables(rows),
    ]
    if paired_rows:
        md.append(
            "Paired PnL differences on the same markets:\n\n"
            + markdown_table(["agent", "baseline", "mean difference", "95% CI"], paired_rows)
        )
    spread = _seed_spread(results)
    if spread:
        md.append(spread)
    (out / "eval_summary.md").write_text("\n\n".join(md) + "\n")
    if log:
        print((out / "eval_summary.md").read_text())
    return results


def run_robustness(run: RunConfig, log: bool = True) -> Results:
    policies = load_policies(run)
    seeds = tuple(eval_seeds(run.eval.robustness_episodes, offset=ROBUSTNESS_SEED_OFFSET))
    jobs = []
    for group, label, overrides in regimes(run):
        for p, cfg in policies:
            jobs.append(Job(f"{group}|{label}|{p.name}", p, cfg.replace(**overrides), seeds))
    results = run_jobs(jobs, workers=_workers(run))

    rows: list[dict[str, float | str]] = []
    for key, ms in results.items():
        group, label, name = key.split("|")
        pnl = summarize([m.pnl for m in ms])
        rows.append(
            {
                "group": group,
                "regime": label,
                "policy": name,
                "episodes": float(len(ms)),
                "pnl_mean": pnl.mean,
                "pnl_std": pnl.std,
                "pnl_ci_lo": pnl.lo,
                "pnl_ci_hi": pnl.hi,
                "sharpe_mean": float(np.mean([m.sharpe for m in ms])),
                "mean_abs_inventory": float(np.mean([m.mean_abs_inventory for m in ms])),
                "fills_mean": float(np.mean([m.fills for m in ms])),
                "markout_1s_per_lot": float(
                    sum(m.markout_1s * m.filled_lots for m in ms)
                    / max(1, sum(m.filled_lots for m in ms))
                ),
            }
        )
    out = Path(run.eval.out_dir)
    write_csv(out / "robustness.csv", rows)

    names = [p.name for p, _ in policies]
    sections = []
    for group in dict.fromkeys(str(r["group"]) for r in rows):
        labels = list(dict.fromkeys(str(r["regime"]) for r in rows if r["group"] == group))
        table = []
        for label in labels:
            cells = [str(label)]
            for n in names:
                r = next(
                    x
                    for x in rows
                    if x["group"] == group and x["regime"] == label and x["policy"] == n
                )
                cells.append(f"{r['pnl_mean']:.1f} [{r['pnl_ci_lo']:.1f}, {r['pnl_ci_hi']:.1f}]")
            table.append(cells)
        sections.append(f"**{group}**\n\n" + markdown_table([group, *names], table))
    md = (
        f"Mean PnL [95% CI] over {len(seeds)} markets per cell. Training conditions are "
        "50 us latency each way, volatility 2, informed share 0.5.\n\n" + "\n\n".join(sections)
    )
    (out / "robustness.md").write_text(md + "\n")
    if log:
        print(md)
    return results
