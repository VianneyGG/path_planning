"""Run pipeline module."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

from src.benchmark.core.config import DEFAULTS

ROOT = Path(__file__).resolve().parents[3]

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports � each step main() is imported only when called so dry-run
# stays fast and import errors surface clearly.
# ---------------------------------------------------------------------------

def _tune_main() -> Callable[[list[str]], None]:
    from src.benchmark.jobs.tune_algo_bayes import main
    return main


def _benchmark_main() -> Callable[[list[str]], None]:
    from src.benchmark.jobs.benchmark_algo_vs_basic import main
    return main


def _tuning_plot_main() -> Callable[[list[str]], None]:
    from src.benchmark.viz.plot_tuning_curves import main
    return main


# ---------------------------------------------------------------------------
# Step dispatcher
# ---------------------------------------------------------------------------

def _run_step(label: str, argv: list[str], fn: Callable[[list[str]], None], dry_run: bool) -> None:
    """Call fn(argv) in-process, or log the call for --dry-run."""
    _LOG.info("[step] %s", label)
    _LOG.debug("  argv: %s", " ".join(argv))
    if dry_run:
        _LOG.info("  [dry-run] skipped -- would call: python -m ... %s", " ".join(argv))
        return
    fn(argv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run full benchmarking pipeline: tune -> benchmark -> table -> plots.",
    )
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--scenarios", type=int, nargs="+", default=list(DEFAULTS.scenarios))
    parser.add_argument("--runs", type=int, default=DEFAULTS.runs)
    parser.add_argument("--no-tuning", dest="no_tuning", action="store_true", help="Skip tuning and reuse a previous tuning summary.")
    parser.add_argument(
        "--tuning-exp-id", dest="tuning_exp_id", type=str, default=None,
        help="Exp-id whose tuning/ dir holds the tuning_summary.json to reuse with --no-tuning. "
             "Defaults to --algo when --no-tuning is set, or --exp-id otherwise.",
    )
    parser.add_argument("--init-points", type=int, default=DEFAULTS.init_points)
    parser.add_argument("--n-iter", type=int, default=DEFAULTS.n_iter)
    parser.add_argument("--eval-repeats", type=int, default=DEFAULTS.eval_repeats)
    parser.add_argument("--collision-penalty", type=float, default=DEFAULTS.collision_penalty)
    parser.add_argument("--non-collision-free-penalty", type=float, default=DEFAULTS.non_collision_free_penalty)
    parser.add_argument("--collision-free-weight", type=float, default=DEFAULTS.collision_free_weight)
    parser.add_argument("--no-feasible-penalty", type=float, default=DEFAULTS.no_feasible_penalty)
    parser.add_argument("--n-jobs", type=int, default=DEFAULTS.n_jobs)
    parser.add_argument("--chunk-size", type=int, default=DEFAULTS.chunk_size)
    parser.add_argument("--adaptive-chunking", dest="adaptive_chunking", action="store_true")
    parser.add_argument("--no-adaptive-chunking", dest="adaptive_chunking", action="store_false")
    parser.set_defaults(adaptive_chunking=DEFAULTS.adaptive_chunking)
    parser.add_argument("--grid-warmstart-points", type=int, default=DEFAULTS.grid_warmstart_points)
    parser.add_argument("--grid-focus-params", type=int, default=DEFAULTS.grid_focus_params)
    parser.add_argument("--hpo-backend", type=str, choices=["bayes_opt", "optuna"], default=DEFAULTS.hpo_backend)
    parser.add_argument(
        "--hpo-sampler", dest="hpo_sampler",
        type=str, choices=["tpe", "cmaes"], default="tpe",
        help="Optuna sampler. 'cmaes' uses CmaEsSampler (HPO-B).",
    )
    parser.add_argument(
        "--enable-pruning", dest="enable_pruning", action="store_true",
        help="Enable Optuna MedianPruner to cut bad trials early (HPO-A).",
    )
    parser.add_argument("--disable-auto-penalties", action="store_true")
    parser.add_argument("--penalty-calibration-runs", type=int, default=DEFAULTS.penalty_calibration_runs)
    parser.add_argument("--time-weight", type=float, default=DEFAULTS.time_weight)

    parser.add_argument("--confidence-level", type=float, default=0.95,
                        help="CI confidence level for iteration-loss plots.")
    parser.add_argument("--plot-curves", dest="plot_curves", action="store_true",
                        help="Run loss-curve plot steps after tune and benchmark (default: on).")
    parser.add_argument("--no-plot-curves", dest="plot_curves", action="store_false")
    parser.set_defaults(plot_curves=True)
    parser.add_argument(
        "--vanilla-params-summary",
        type=str,
        default=DEFAULTS.vanilla_params_summary,
        help="Canonical baseline tuning summary used for vanilla in compare mode.",
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULTS.seed_base)
    parser.add_argument("--mode", type=str, choices=["compare", "vanilla_only"], default=DEFAULTS.mode)
    parser.add_argument("--exp-id", type=str, default=DEFAULTS.exp_id)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be executed without running anything.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        _LOG.info("[dry-run mode] No steps will be executed.")

    eval_repeats = int(args.eval_repeats)
    if any(int(s) >= 4 for s in args.scenarios) and eval_repeats < 6:
        _LOG.info(
            "Scenario 4 detected with low --eval-repeats=%d; "
            "tuning will be faster but less robust to seed noise.",
            eval_repeats,
        )

    artifacts = ROOT / "src" / "benchmark" / "artifacts" / args.exp_id

    # When --no-tuning, resolve the tuning dir from a previous run.
    # Priority: explicit --tuning-exp-id > algo name > exp_id (fallback).
    if args.no_tuning:
        tuning_exp_id = args.tuning_exp_id or args.algo
    else:
        tuning_exp_id = args.tuning_exp_id or args.exp_id
    tuning_dir = ROOT / "src" / "benchmark" / "artifacts" / tuning_exp_id / "tuning"

    benchmark_dir = artifacts / "benchmark"
    logplots_dir = artifacts / "logplots"

    for d in (benchmark_dir, logplots_dir):
        d.mkdir(parents=True, exist_ok=True)
    if not args.no_tuning:
        tuning_dir.mkdir(parents=True, exist_ok=True)

    log_level_argv = ["--log-level", args.log_level]

    # step 1: tune
    if not args.no_tuning:
        
        tune_argv = [
            "--algo", args.algo,
            "--scenarios", *[str(s) for s in args.scenarios],
            "--init-points", str(args.init_points),
            "--n-iter", str(args.n_iter),
            "--eval-repeats", str(eval_repeats),
            "--collision-penalty", str(args.collision_penalty),
            "--non-collision-free-penalty", str(args.non_collision_free_penalty),
            "--collision-free-weight", str(args.collision_free_weight),
            "--no-feasible-penalty", str(args.no_feasible_penalty),
            "--time-weight", str(args.time_weight),
            "--n-jobs", str(args.n_jobs),
            "--grid-warmstart-points", str(args.grid_warmstart_points),
            "--grid-focus-params", str(args.grid_focus_params),
            "--hpo-backend", str(args.hpo_backend),
            "--penalty-calibration-runs", str(args.penalty_calibration_runs),
            "--seed-base", str(args.seed_base),
            "--out-dir", str(tuning_dir.relative_to(ROOT)),
            "--skip-global",
            "--hpo-sampler", str(args.hpo_sampler),
            *(("--enable-pruning",) if args.enable_pruning else ()),
            *log_level_argv,
        ]
        if args.disable_auto_penalties:
            tune_argv.append("--disable-auto-penalties")
        _run_step("tune", tune_argv, _tune_main(), args.dry_run)

        # plot tuning PSO curves (collision-free iterations)
        pso_curves_path = tuning_dir / "tuning_pso_curves.parquet"
        if args.plot_curves and (pso_curves_path.exists() or args.dry_run):
            tplot_argv = [
                "--pso-curves", str(pso_curves_path.relative_to(ROOT)),
                "--confidence-level", str(args.confidence_level),
                "--out-dir", str(logplots_dir.relative_to(ROOT)),
                *log_level_argv,
            ]
            _run_step("plot_tuning_curves", tplot_argv, _tuning_plot_main(), args.dry_run)

    # step 2: benchmark

    if args.no_tuning:
        _LOG.info(
            "Skipping tuning step (--no-tuning); using tuning summary from: %s",
            tuning_dir / "tuning_summary.json",
        )
    
    bench_argv = [
        "--algo", args.algo,
        "--scenarios", *[str(s) for s in args.scenarios],
        "--runs", str(args.runs),
        "--seed-base", str(args.seed_base),
        "--n-jobs", str(args.n_jobs),
        "--chunk-size", str(args.chunk_size),
        "--adaptive-chunking" if args.adaptive_chunking else "--no-adaptive-chunking",
        "--mode", args.mode,
        "--out-dir", str(benchmark_dir.relative_to(ROOT)),
        *log_level_argv,
    ]
    if args.mode == "compare":
        bench_argv += [
            "--params-summary",
            str((tuning_dir / "tuning_summary.json").relative_to(ROOT)),
            "--vanilla-params-summary",
            str(Path(args.vanilla_params_summary).as_posix()),
        ]
    _run_step("benchmark", bench_argv, _benchmark_main(), args.dry_run)

    # step 2b: plot benchmark collision-free iteration loss curves
    bench_curves_path = benchmark_dir / "benchmark_loss_curves.parquet"
    if args.plot_curves and (bench_curves_path.exists() or args.dry_run):
        bplot_argv = [
            "--benchmark-curves", str(bench_curves_path.relative_to(ROOT)),
            "--confidence-level", str(args.confidence_level),
            "--out-dir", str(logplots_dir.relative_to(ROOT)),
            *log_level_argv,
        ]
        _run_step("plot_benchmark_curves", bplot_argv, _tuning_plot_main(), args.dry_run)

    _LOG.info("=== Pipeline %s ===", "dry-run complete" if args.dry_run else "complete")
    _LOG.info("Tuning:    %s", tuning_dir)
    _LOG.info("Benchmark: %s", benchmark_dir)
    _LOG.info("Plots:     %s", logplots_dir)


if __name__ == "__main__":
    main()
