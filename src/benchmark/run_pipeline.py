from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run_step(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmarking pipeline: tune -> benchmark -> table -> plots.")
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--init-points", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=80)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=["compare", "vanilla_only"], default="compare")
    parser.add_argument("--exp-id", type=str, default="exp01")
    args = parser.parse_args()

    artifacts = ROOT / "src" / "benchmark" / "artifacts" / args.exp_id
    tuning_dir = artifacts / "tuning"
    benchmark_dir = artifacts / "benchmark"
    plots_dir = artifacts / "plots"

    tuning_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    py = [sys.executable]

    tune_cmd = py + [
        "-m",
        "src.benchmark.tune_algo_bayes",
        "--algo",
        args.algo,
        "--scenarios",
        *[str(s) for s in args.scenarios],
        "--init-points",
        str(args.init_points),
        "--n-iter",
        str(args.n_iter),
        "--eval-repeats",
        str(args.eval_repeats),
        "--n-jobs",
        str(args.n_jobs),
        "--skip-global",
         # "--skip_local",
        "--seed-base",
        str(args.seed_base),
        "--out-dir",
        str(tuning_dir.relative_to(ROOT)),
    ]
    _run_step(tune_cmd, ROOT)

    benchmark_cmd = py + [
        "-m",
        "src.benchmark.benchmark_algo_vs_basic",
        "--algo",
        args.algo,
        "--scenarios",
        *[str(s) for s in args.scenarios],
        "--runs",
        str(args.runs),
        "--seed-base",
        str(args.seed_base),
        "--n-jobs",
        str(args.n_jobs),
        "--mode",
        args.mode,
        "--out-dir",
        str(benchmark_dir.relative_to(ROOT)),
    ]
    if args.mode == "compare":
        benchmark_cmd.extend(["--params-summary", str((tuning_dir / "tuning_summary.json").relative_to(ROOT))])

    _run_step(benchmark_cmd, ROOT)

    report_cmd = py + [
        "-m",
        "src.benchmark.performance",
        "--benchmark-dir",
        str(benchmark_dir.relative_to(ROOT)),
        "--runs-file",
        "benchmark_runs.parquet",
        "--out-csv",
        "comparison_table.csv",
    ]
    _run_step(report_cmd, ROOT)

    plot_cmd = py + [
        "-m",
        "src.benchmark.plot_all_algos",
        "--runs",
        str((benchmark_dir / "benchmark_runs.parquet").relative_to(ROOT)),
        "--out-dir",
        str(plots_dir.relative_to(ROOT)),
        "--mode",
        args.mode,
    ]
    _run_step(plot_cmd, ROOT)

    print("\n=== Pipeline complete ===")
    print(f"Tuning: {tuning_dir}")
    print(f"Benchmark: {benchmark_dir}")
    print(f"Reports CSV: {benchmark_dir / 'comparison_table.csv'}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
