"""Unified CLI entry point for the benchmark package.

Usage examples
--------------
Tune only:
    python -m src.benchmark tune --algo RS_SA_PH --n-iter 50

Benchmark only (needs an existing tuning_summary.json):
    python -m src.benchmark benchmark --algo RS_SA_PH --runs 50

Full pipeline (tune → benchmark → table → plots):
    python -m src.benchmark run --algo RS_SA_PH --runs 100 --exp-id my_exp

Dry-run the full pipeline (no computation):
    python -m src.benchmark run --algo RS_SA_PH --dry-run
"""

from __future__ import annotations

import argparse
import sys


def _cmd_run(rest: list[str]) -> None:
    from src.benchmark.jobs.run_pipeline import main
    main(rest)


def _cmd_tune(rest: list[str]) -> None:
    from src.benchmark.jobs.tune_algo_bayes import main
    main(rest)


def _cmd_benchmark(rest: list[str]) -> None:
    from src.benchmark.jobs.benchmark_algo_vs_basic import main
    main(rest)


def _cmd_report(rest: list[str]) -> None:
    from src.benchmark.viz.performance import main
    main(rest)


def _cmd_plot(rest: list[str]) -> None:
    from src.benchmark.viz.plot_all_algos import main
    main(rest)


_COMMANDS: dict[str, tuple[str, object]] = {
    "run":       ("Full pipeline: tune → benchmark → table → plots.", _cmd_run),
    "tune":      ("Bayesian hyper-parameter tuning only.",             _cmd_tune),
    "benchmark": ("Benchmark a tuned algorithm vs vanilla baseline.",  _cmd_benchmark),
    "report":    ("Build per-scenario performance table from runs.",   _cmd_report),
    "plot":      ("Generate comparison boxplots from runs.",           _cmd_plot),
}


def main() -> None:
    top = argparse.ArgumentParser(
        prog="python -m src.benchmark",
        description="Path-planning benchmark suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {cmd:<11} {desc}" for cmd, (desc, _) in _COMMANDS.items()
        ),
    )
    top.add_argument("command", choices=list(_COMMANDS), metavar="command",
                     help="{" + ", ".join(_COMMANDS) + "}")
    top.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the sub-command.")

    # Show help if called without arguments
    if len(sys.argv) == 1:
        top.print_help()
        raise SystemExit(0)

    parsed = top.parse_args(sys.argv[1:2])
    _, fn = _COMMANDS[parsed.command]
    fn(sys.argv[2:])  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
