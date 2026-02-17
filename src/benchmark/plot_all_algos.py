from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.benchmark.performance import (
    _plot_metric_boxplot_by_scenario,
    _prepare_algorithm_labels,
    _build_algo_palette_and_order,
    PLOT_DPI,
    PLOT_FIGSIZE_WIDTH_FACTOR,
    PLOT_FIGSIZE_HEIGHT,
    PLOT_FONTSIZE_TITLE,
    PLOT_FONTSIZE_LABEL,
    PLOT_FONTSIZE_LEGEND,
    PLOT_LINEWIDTH,
    COLOR_GRADIENT_START,
    COLOR_GRADIENT_END,
)


ROOT = Path(__file__).resolve().parents[2]


def _plot_collision_free_barplot(runs_df: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "Plotting requires seaborn and matplotlib. Install with `uv pip install seaborn matplotlib`."
        ) from exc

    if "is_collision_free" not in runs_df.columns:
        raise ValueError("Column 'is_collision_free' is required for collision-free proportion barplot.")

    df = _prepare_algorithm_labels(runs_df)
    if "scenario" not in df.columns:
        raise ValueError("Column 'scenario' is required for scenario-wise barplot.")

    df = df.dropna(subset=["scenario", "algorithm", "is_collision_free"]).copy()
    if df.empty:
        raise ValueError("No rows available to plot collision-free proportion.")

    df["scenario"] = pd.to_numeric(df["scenario"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["scenario"])
    df["scenario"] = df["scenario"].astype(int)
    df["is_collision_free"] = df["is_collision_free"].astype("boolean").astype(float)

    grouped = (
        df.groupby(["scenario", "algorithm"], as_index=False)
        .agg(collision_free_proportion=("is_collision_free", "mean"))
        .sort_values(["scenario", "algorithm"])
    )

    fig_width = max(10, PLOT_FIGSIZE_WIDTH_FACTOR * grouped["scenario"].nunique())
    plt.figure(figsize=(fig_width, PLOT_FIGSIZE_HEIGHT), dpi=PLOT_DPI)
    sns.set_theme(style="whitegrid")

    # get ordered algos and colors
    present_algos = list(grouped["algorithm"].unique())
    ordered_algos, colors_list = _build_algo_palette_and_order(present_algos)

    ax = sns.barplot(
        data=grouped,
        x="scenario",
        y="collision_free_proportion",
        hue="algorithm",
        hue_order=ordered_algos,
        palette=colors_list,
        linewidth=PLOT_LINEWIDTH,
    )
    ax.set_xlabel("Scenario", fontsize=PLOT_FONTSIZE_LABEL, fontweight="bold")
    ax.set_ylabel("Collision-free proportion", fontsize=PLOT_FONTSIZE_LABEL, fontweight="bold")
    ax.tick_params(labelsize=PLOT_FONTSIZE_LABEL - 1)
    ax.set_ylim(0.0, 1.0)

    # add run counts to title
    total_runs = int(df.shape[0])
    algo_counts = df["algorithm"].value_counts().to_dict()
    counts_suffix = f" ({total_runs//(5 *len(algo_counts))} runs per scenario)"

    ax.set_title("Collision-free proportion by scenario" + counts_suffix, fontsize=PLOT_FONTSIZE_TITLE, fontweight="bold", pad=15)
    ax.legend(title="Algorithm", fontsize=PLOT_FONTSIZE_LEGEND, title_fontsize=PLOT_FONTSIZE_LEGEND)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate standard comparison plots: path length boxplot, CPU time boxplot, collision-free proportion barplot."
    )
    parser.add_argument("--runs", type=str, required=True, help="Path to runs parquet file.")
    parser.add_argument("--out-dir", type=str, default="artifacts/exp/plots")
    parser.add_argument("--mode", type=str, choices=["compare", "vanilla_only"], default="compare")
    args = parser.parse_args()

    runs_path = Path(args.runs)
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs file not found: {runs_path}")

    runs_df = pd.read_parquet(runs_path)
    runs_df = _prepare_algorithm_labels(runs_df)

    if args.mode == "vanilla_only":
        runs_df = runs_df[runs_df["algorithm"] == "Basic"].copy()
    elif args.mode == "compare":
        # keep everything present in the runs file (Basic + any tuned algos)
        runs_df = runs_df.copy()

    if runs_df.empty:
        raise ValueError("No rows available for plotting after mode filtering.")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    length_plot = out_dir / "boxplot_length.png"
    _plot_metric_boxplot_by_scenario(
        runs_df=runs_df,
        metric="length",
        output_path=length_plot,
        valid_only=False,
        title="Path length comparison by scenario",
    )
    print(f"Saved boxplot: {length_plot}")

    time_plot = out_dir / "boxplot_time_sec.png"
    _plot_metric_boxplot_by_scenario(
        runs_df=runs_df,
        metric="time_sec",
        output_path=time_plot,
        valid_only=False,
        title="CPU time comparison by scenario",
    )
    print(f"Saved boxplot: {time_plot}")

    collision_bar_plot = out_dir / "barplot_collision_free_proportion.png"
    _plot_collision_free_barplot(runs_df=runs_df, output_path=collision_bar_plot)
    print(f"Saved barplot: {collision_bar_plot}")


if __name__ == "__main__":
    main()
