"""Command line: ``rlmm train``, ``rlmm calibrate``, ``rlmm eval``, ``rlmm plot``."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from rlmm.config import load_config


def _train(args: argparse.Namespace) -> None:
    from rlmm.train import train_seed

    run = load_config(args.config)
    if args.total_timesteps:
        run = replace(run, ppo=replace(run.ppo, total_timesteps=args.total_timesteps))
    seeds = [args.seed] if args.seed is not None else list(run.train.seeds)
    for s in seeds:
        out = train_seed(run, s, args.out)
        print(f"seed {s} done: {out}")


def _calibrate(args: argparse.Namespace) -> None:
    from rlmm.baselines.calibration import run_calibration

    run = load_config(args.config)
    cal = run_calibration(
        run.env,
        tune_episodes=args.episodes,
        workers=run.eval.workers or None,
    )
    out = Path(args.out or run.eval.calibration)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal.to_dict(), indent=2))
    print(f"wrote {out}")


def _eval(args: argparse.Namespace) -> None:
    from rlmm.evaluation.pipeline import run_main, run_stress

    run = load_config(args.config)
    if args.workers:
        run = replace(run, eval=replace(run.eval, workers=args.workers))
    if not args.skip_main:
        run_main(run)
    if not args.skip_stress:
        run_stress(run)


def _plot(args: argparse.Namespace) -> None:
    import numpy as np

    from rlmm import plotting
    from rlmm.env import LobMarketMakingEnv
    from rlmm.evaluation.pipeline import load_policies
    from rlmm.evaluation.runner import eval_seeds, run_episode

    run = load_config(args.config)
    res = Path(run.eval.out_dir)
    fig_dir = res / "figures"
    runs = {name: Path(path).parent for name, path in run.eval.policies.items()}
    metrics = plotting.load_runs([(n, d) for n, d in runs.items() if (d / "metrics.csv").exists()])
    # Keep a copy of each training log next to the results it produced.
    for n, d in runs.items():
        if (d / "metrics.csv").exists():
            (res / "training").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(d / "metrics.csv", res / "training" / f"{n}.csv")
    baselines: dict[str, float] = {}
    if (res / "eval_summary.csv").exists():
        for r in plotting.read_csv(res / "eval_summary.csv"):
            if not r["policy"].startswith("ppo"):
                baselines[r["policy"]] = float(r["pnl_mean"])
    groups: dict[str, dict[str, list[dict[str, str]]]] = {}
    for name, rows in metrics.items():
        groups.setdefault(name.rsplit("_seed", 1)[0], {})[name] = rows
    for prefix, g in groups.items():
        plotting.learning_curves(
            g, baselines, fig_dir / f"learning_curves_{prefix}.png", f"{prefix}: learning curves"
        )
        plotting.diagnostics(g, fig_dir / f"diagnostics_{prefix}.png")
    if (res / "eval_episodes.csv").exists():
        plotting.pnl_distribution(res / "eval_episodes.csv", fig_dir / "pnl_distribution.png")
        plotting.pnl_decomposition(res / "eval_summary.csv", fig_dir / "pnl_decomposition.png")
    if (res / "stress.csv").exists():
        plotting.stress(res / "stress.csv", fig_dir / "stress.png")
    cal_path = Path(run.eval.calibration)
    if cal_path.exists():
        plotting.calibration_plot(json.loads(cal_path.read_text()), fig_dir / "calibration.png")

    traces: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for policy, cfg in load_policies(run):
        env = LobMarketMakingEnv(cfg, record=True)
        eps = []
        for s in eval_seeds(args.trace_episodes):
            tr = run_episode(env, policy, s)
            pos_before = np.concatenate([[0], np.asarray(tr.position[:-1])])
            eps.append((pos_before, np.asarray(tr.bid_offset), np.asarray(tr.ask_offset)))
        # Seeds of the same agent are pooled so the figure stays readable.
        name = policy.name.rsplit("_seed", 1)[0] if "_seed" in policy.name else policy.name
        traces.setdefault(name, []).extend(eps)
    plotting.inventory_and_behaviour(
        traces,
        fig_dir / "inventory_distribution.png",
        fig_dir / "policy_behaviour.png",
        run.env.max_position,
    )
    print(f"figures in {fig_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rlmm", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train PPO, one run per seed")
    t.add_argument("--config", required=True)
    t.add_argument("--seed", type=int, help="train only this seed")
    t.add_argument("--out", help="output directory (default: [train] out_dir)")
    t.add_argument("--total-timesteps", type=int, help="override [ppo] total_timesteps")
    t.set_defaults(func=_train)

    c = sub.add_parser("calibrate", help="fit AS inputs and tune the baselines")
    c.add_argument("--config", required=True)
    c.add_argument("--out", help="output JSON (default: [eval] calibration)")
    c.add_argument("--episodes", type=int, default=200, help="markets per grid point")
    c.set_defaults(func=_calibrate)

    e = sub.add_parser("eval", help="held-out comparison and out-of-regime stress tests")
    e.add_argument("--config", required=True)
    e.add_argument("--workers", type=int, help="processes (default: CPUs - 1)")
    e.add_argument("--skip-main", action="store_true")
    e.add_argument("--skip-stress", action="store_true")
    e.set_defaults(func=_eval)

    pl = sub.add_parser("plot", help="figures from training logs and eval results")
    pl.add_argument("--config", required=True)
    pl.add_argument("--trace-episodes", type=int, default=40)
    pl.set_defaults(func=_plot)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
