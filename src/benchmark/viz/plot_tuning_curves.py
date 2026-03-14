"""Plot tuning curves module."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark.core.algo_profiles import ALGO_LABELS

_LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def _z_value(confidence_level: float) -> float:
    level = float(confidence_level)
    if level >= 0.99:
        return 2.576
    if level >= 0.98:
        return 2.326
    if level >= 0.95:
        return 1.96
    if level >= 0.90:
        return 1.645
    return 1.0


def _plot_loss_pso(pso_df: pd.DataFrame, confidence_level: float, out_path: Path) -> None:
    """Plot collision-free best_fitness vs PSO iteration (tuning curves)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install with `uv pip install matplotlib`.") from exc

    if pso_df.empty:
        return

    # Filter to CF rows only (all iterations are stored since the schema change;
    # non-CF iterations carry is_collision_free=False and should be excluded here).
    if "is_collision_free" in pso_df.columns:
        pso_df = pso_df[pso_df["is_collision_free"]]
    if pso_df.empty:
        return

    grouped = (
        pso_df.groupby(["scope", "scope_id", "algo", "scenario", "pso_iteration"], as_index=False)
        .agg(mean_best_fitness=("best_fitness", "mean"), std_best_fitness=("best_fitness", "std"), n=("best_fitness", "count"))
        .sort_values(["scope", "scope_id", "scenario", "pso_iteration"])
    )

    z = _z_value(confidence_level)
    grouped["std_best_fitness"] = grouped["std_best_fitness"].fillna(0.0)
    grouped["ci95"] = z * grouped["std_best_fitness"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["low"] = grouped["mean_best_fitness"] - grouped["ci95"]
    grouped["high"] = grouped["mean_best_fitness"] + grouped["ci95"]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    for (scope, scope_id, scenario, algo), frame in grouped.groupby(["scope", "scope_id", "scenario", "algo"]):
        frame = frame.sort_values("pso_iteration")
        display = ALGO_LABELS.get(str(algo), str(algo))
        if int(scenario) >= 0:
            label = f"{display} | s{int(scenario)}"
        else:
            label = display
        x = frame["pso_iteration"].to_numpy(dtype=float)
        y = frame["mean_best_fitness"].to_numpy(dtype=float)
        low = frame["low"].to_numpy(dtype=float)
        high = frame["high"].to_numpy(dtype=float)
        ax.plot(x, y, linewidth=1.5, label=label)
        ax.fill_between(x, low, high, alpha=0.2)

    ax.set_title(f"Tuning — collision-free best fitness by PSO iteration (CI {int(confidence_level * 100)}%)")
    ax.set_xlabel("PSO iteration")
    ax.set_ylabel("Best fitness (collision-free only)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_benchmark_loss(bench_df: pd.DataFrame, confidence_level: float, out_path: Path) -> None:
    """Plot best_fitness vs PSO iteration for runs that ended collision-free."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install with `uv pip install matplotlib`.") from exc

    if bench_df.empty:
        return

    # Sentinel rows (iteration = -1) represent non-CF runs and must be excluded:
    # they would appear at x = -1, pulling the axis left of zero.
    bench_df = bench_df[bench_df["iteration"] >= 0]
    if bench_df.empty:
        return

    grouped = (
        bench_df.groupby(["algo", "scenario", "iteration"], as_index=False)
        .agg(mean_best_fitness=("best_fitness", "mean"), std_best_fitness=("best_fitness", "std"), n=("best_fitness", "count"))
        .sort_values(["scenario", "algo", "iteration"])
    )

    z = _z_value(confidence_level)
    grouped["std_best_fitness"] = grouped["std_best_fitness"].fillna(0.0)
    grouped["ci95"] = z * grouped["std_best_fitness"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["low"] = grouped["mean_best_fitness"] - grouped["ci95"]
    grouped["high"] = grouped["mean_best_fitness"] + grouped["ci95"]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    for (scenario, algo), frame in grouped.groupby(["scenario", "algo"]):
        frame = frame.sort_values("iteration")
        display = ALGO_LABELS.get(str(algo), str(algo))
        label = f"{display} | s{int(scenario)}"
        x = frame["iteration"].to_numpy(dtype=float)
        y = frame["mean_best_fitness"].to_numpy(dtype=float)
        low = frame["low"].to_numpy(dtype=float)
        high = frame["high"].to_numpy(dtype=float)
        ax.plot(x, y, linewidth=1.5, label=label)
        ax.fill_between(x, low, high, alpha=0.2)

    ax.set_title(f"Benchmark — best fitness by iteration for CF-ending runs (CI {int(confidence_level * 100)}%)")
    ax.set_xlabel("PSO iteration")
    ax.set_ylabel("Best fitness")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_first_cf_boxplot(bench_df: pd.DataFrame, out_path: Path) -> None:
    """Boxplot of first collision-free iteration per run, grouped by (scenario, algo)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install with `uv pip install matplotlib`.") from exc

    if bench_df.empty or "is_collision_free" not in bench_df.columns:
        return

    cf_rows = bench_df[bench_df["is_collision_free"]]
    if cf_rows.empty:
        return

    first_cf = (
        cf_rows.groupby("run_id")["iteration"].min().reset_index(name="first_cf_iter")
    )
    meta = bench_df[["run_id", "scenario", "algo"]].drop_duplicates("run_id")
    first_cf = first_cf.merge(meta, on="run_id", how="left")

    scenarios = sorted(first_cf["scenario"].unique())
    n_scenarios = len(scenarios)
    if n_scenarios == 0:
        return

    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 6), dpi=160, sharey=False)
    if n_scenarios == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios):
        sub = first_cf[first_cf["scenario"] == scen]
        algos = sorted(sub["algo"].unique())
        data = [sub[sub["algo"] == a]["first_cf_iter"].to_numpy() for a in algos]
        labels = [ALGO_LABELS.get(str(a), str(a)) for a in algos]
        ax.boxplot(data, labels=labels, patch_artist=True)
        ax.set_title(f"Scenario {int(scen)}")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("First CF iteration")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Iteration at which each run first achieves a collision-free path", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_benchmark_perf(bench_df: pd.DataFrame, out_path: Path) -> None:
    """Three-panel performance overview: CPU time, path length, CF proportion.

    Derives per-run summaries via ``drop_duplicates("run_id")`` so the
    run-level constants (``elapsed_s``, ``path_length_m``, ``is_collision_free``)
    stored in every iteration row are only counted once per run.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install with `uv pip install matplotlib`.") from exc

    required = {"run_id", "scenario", "algo", "elapsed_s", "path_length_m", "is_collision_free"}
    if bench_df.empty or not required.issubset(bench_df.columns):
        return

    # CF runs have normal iteration rows (iteration >= 0); non-CF runs only have
    # a sentinel with iteration = -1.  Using iteration >= 0 is the reliable
    # run-level CF indicator — drop_duplicates on is_collision_free is wrong
    # because it picks the first row (usually iter 0, not yet CF) even for
    # runs that eventually converge.
    all_runs = bench_df[["run_id", "scenario", "algo", "elapsed_s"]].drop_duplicates("run_id")
    cf_runs = (
        bench_df.loc[bench_df["iteration"] >= 0, ["run_id", "scenario", "algo", "path_length_m"]]
        .drop_duplicates("run_id")
    )

    scenarios = sorted(all_runs["scenario"].unique())
    n_scen = len(scenarios)
    if n_scen == 0:
        return

    fig, axes = plt.subplots(3, n_scen, figsize=(5 * n_scen, 13), dpi=160, squeeze=False)

    for col, scen in enumerate(scenarios):
        sub_all = all_runs[all_runs["scenario"] == scen]
        sub_cf  = cf_runs[cf_runs["scenario"] == scen]
        algos = sorted(sub_all["algo"].unique())
        labels = [ALGO_LABELS.get(str(a), str(a)) for a in algos]

        # ── row 0: CPU time (all runs) ───────────────────────────────────────
        ax0 = axes[0][col]
        data_time = [sub_all[sub_all["algo"] == a]["elapsed_s"].dropna().to_numpy() for a in algos]
        ax0.boxplot(data_time, labels=labels, patch_artist=True)
        ax0.set_title(f"Scenario {int(scen)}")
        ax0.set_ylabel("CPU time (s)" if col == 0 else "")
        ax0.grid(True, axis="y", alpha=0.3)
        ax0.tick_params(axis="x", rotation=30)

        # ── row 1: path length (CF runs only) ────────────────────────────────
        ax1 = axes[1][col]
        data_len = [sub_cf[sub_cf["algo"] == a]["path_length_m"].dropna().to_numpy() for a in algos]
        ax1.boxplot(data_len, labels=labels, patch_artist=True)
        ax1.set_ylabel("Path length (CF runs)" if col == 0 else "")
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.tick_params(axis="x", rotation=30)

        # ── row 2: CF proportion ─────────────────────────────────────────────
        ax2 = axes[2][col]
        n_total = sub_all.groupby("algo")["run_id"].nunique()
        n_cf    = sub_cf.groupby("algo")["run_id"].nunique()
        cf_rate = [
            float(n_cf.get(a, 0)) / max(1, int(n_total.get(a, 1)))
            for a in algos
        ]
        bars = ax2.bar(labels, cf_rate, color="steelblue", edgecolor="white")
        ax2.set_ylim(0.0, 1.05)
        ax2.set_ylabel("CF proportion" if col == 0 else "")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.tick_params(axis="x", rotation=30)
        for bar, rate in zip(bars, cf_rate):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rate:.0%}",
                ha="center", va="bottom", fontsize=8,
            )

    axes[0][0].figure.suptitle("Benchmark performance overview", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot collision-free iteration loss curves with confidence intervals."
    )
    parser.add_argument("--pso-curves", type=str, default=None,
                        help="Path to tuning_pso_curves.parquet (tuning phase).")
    parser.add_argument("--benchmark-curves", type=str, default=None,
                        help="Path to benchmark_loss_curves.parquet (benchmark phase).")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--out-dir", type=str, default="artifacts/exp/logplots")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if args.pso_curves is None and args.benchmark_curves is None:
        parser.error("At least one of --pso-curves or --benchmark-curves is required.")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pso_curves:
        pso_path = Path(args.pso_curves)
        if not pso_path.is_absolute():
            pso_path = ROOT / pso_path
        if pso_path.exists():
            pso_df = pd.read_parquet(pso_path)
            _plot_loss_pso(
                pso_df=pso_df,
                confidence_level=float(args.confidence_level),
                out_path=out_dir / "tuning_loss_by_pso_iteration_ci.png",
            )
            _LOG.info("Saved: %s", out_dir / "tuning_loss_by_pso_iteration_ci.png")
        else:
            _LOG.warning("PSO curves file not found: %s", pso_path)

    if args.benchmark_curves:
        bench_path = Path(args.benchmark_curves)
        if not bench_path.is_absolute():
            bench_path = ROOT / bench_path
        if bench_path.exists():
            bench_df = pd.read_parquet(bench_path)
            _plot_benchmark_loss(
                bench_df=bench_df,
                confidence_level=float(args.confidence_level),
                out_path=out_dir / "benchmark_loss_by_iteration_ci.png",
            )
            _LOG.info("Saved: %s", out_dir / "benchmark_loss_by_iteration_ci.png")
            _plot_first_cf_boxplot(
                bench_df=bench_df,
                out_path=out_dir / "benchmark_first_cf_iter_boxplot.png",
            )
            _LOG.info("Saved: %s", out_dir / "benchmark_first_cf_iter_boxplot.png")
            _plot_benchmark_perf(
                bench_df=bench_df,
                out_path=out_dir / "benchmark_perf_overview.png",
            )
            _LOG.info("Saved: %s", out_dir / "benchmark_perf_overview.png")
        else:
            _LOG.warning("Benchmark curves file not found: %s", bench_path)


if __name__ == "__main__":
    main()
