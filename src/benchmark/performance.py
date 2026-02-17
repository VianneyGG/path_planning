from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = ROOT / "src" / "benchmarking" / "benchmark_basic"
CASE_LABEL_MAP = {
	"vanilla": "Basic",
	"RS": "RS",
	"dim_only": "DL",
	"sa_no_cc": "SA",
	"dim_plus_sa_no_cc": "DL+SA",
}
# Metrics to use log scale on boxplots (empty set to disable)
LOG_SCALE_METRICS = {"time_sec", "length", "fitness"}
# Plot quality settings
PLOT_DPI = 300
PLOT_FIGSIZE_WIDTH_FACTOR = 2.2  # width = max(10, factor * num_scenarios)
PLOT_FIGSIZE_HEIGHT = 6.5
PLOT_FONTSIZE_TITLE = 18
PLOT_FONTSIZE_LABEL = 12
PLOT_FONTSIZE_LEGEND = 11
PLOT_LINEWIDTH = 1.5
# Gradient endpoints for algorithm palette (blue -> green)
# make Basic a lighter blue by default
COLOR_GRADIENT_START = "#B3D9FF"  # lighter blue (Basic)
COLOR_GRADIENT_END = "#66C2A5"  # green/teal (last algo)
# Backwards-compat colours (fallbacks)
BASIC_COLOR = COLOR_GRADIENT_START
DEFAULT_ALGO_COLOR = COLOR_GRADIENT_END


def _format_mean_var(mean_value: float, var_value: float, decimals: int = 4) -> str:
	return f"{mean_value:.{decimals}f} ± {var_value:.{decimals}f}"


def _build_table(runs_df: pd.DataFrame) -> pd.DataFrame:
	required_columns = {"scenario", "fitness", "time_sec", "is_collision_free"}
	missing = sorted(required_columns - set(runs_df.columns))
	if missing:
		raise ValueError(f"Missing required columns in runs DataFrame: {missing}")

	df = _prepare_algorithm_labels(runs_df)

	if "algorithm" not in df.columns:
		raise ValueError("Could not infer algorithm label from runs data.")

	aggregated = (
		df.groupby(["scenario", "algorithm"], as_index=False)
		.agg(
			length_mean=("length", "mean"),
			length_var=("length", "var"),
			time_mean=("time_sec", "mean"),
			time_var=("time_sec", "var"),
			collision_free_mean=("is_collision_free", "mean"),
			collision_free_var=("is_collision_free", "var"),
		)
		.sort_values(["scenario", "algorithm"])
		.reset_index(drop=True)
	)

	table_df = pd.DataFrame(
		{
			"scenario": aggregated["scenario"].astype(int),
			"algorithm": aggregated["algorithm"].astype(str),
			"length (mean ± variance)": [
				_format_mean_var(mean_val, var_val)
				for mean_val, var_val in zip(aggregated["length_mean"], aggregated["length_var"], strict=True)
			],
			"CPU time(s) (mean ± variance)": [
				_format_mean_var(mean_val, var_val)
				for mean_val, var_val in zip(aggregated["time_mean"], aggregated["time_var"], strict=True)
			],
			"collision-free proportion": aggregated["collision_free_mean"].astype(float).tolist(),
		}
	)
	return table_df


def _algorithm_label_from_row(row: pd.Series) -> str:
	dl = bool(row.get("dimensional_learning", False))
	sa = bool(row.get("simulated_annealing", False))
	if dl and sa:
		return "DL+SA"
	if dl:
		return "DL"
	if sa:
		return "SA"
	return "RS"


def _prepare_algorithm_labels(runs_df: pd.DataFrame) -> pd.DataFrame:
	df = runs_df.copy()
	if "case" in df.columns:
		df["algorithm"] = df["case"].astype(str).map(CASE_LABEL_MAP).fillna(df["case"].astype(str))
		return df
	if "algo" in df.columns:
		df["algorithm"] = df["algo"].astype(str).map(CASE_LABEL_MAP).fillna(df["algo"].astype(str))
		return df

	df["algorithm"] = df.apply(_algorithm_label_from_row, axis=1)
	return df


def _build_algo_palette_and_order(present_algo_labels: list[str]) -> tuple[list[str], list[str]]:
	"""
	Build an ordered list of algorithms and their corresponding discrete colors.
	
	Returns:
		(ordered_algos, colors_list): ordered list of algo labels and hex colors.
	"""
	from src.benchmark.algo_profiles import ALGO_PLOT_ORDER, ALGO_LABELS
	
	# Map ALGO_PLOT_ORDER keys to display labels (those that appear in present_algo_labels)
	all_labels = [ALGO_LABELS.get(k, k) for k in ALGO_PLOT_ORDER]
	ordered = [a for a in all_labels if a in present_algo_labels]
	
	# Append any remaining algos not in ALGO_PLOT_ORDER
	for a in present_algo_labels:
		if a not in ordered:
			ordered.append(a)
	
	# Discrete color palette: each algo has a fixed color (not indexed by position)
	algo_color_map = {
		"Basic": "#B3D9FF",           # light blue
		"RS": "#2E8B9E",              # teal/dark cyan
		"RS_SA_noCC": "#52A55A",      # medium green
		"RS_SA_noCC_DL": "#DAA520",   # goldenrod
		"RS_SA_CC": "#E74C3C",        # coral/red
	}
	colors = [algo_color_map.get(a, "#808080") for a in ordered]
	
	return ordered, colors


def _plot_metric_boxplot_by_scenario(
	runs_df: pd.DataFrame,
	metric: str,
	output_path: Path,
	valid_only: bool,
	title: str | None = None,
) -> None:
	try:
		import matplotlib.pyplot as plt
		import seaborn as sns
		from matplotlib import colors as mcolors
	except ImportError as exc:
		raise ImportError(
			"Plotting requires seaborn and matplotlib. Install with `uv pip install seaborn matplotlib`."
		) from exc

	if metric not in runs_df.columns:
		raise ValueError(f"Metric '{metric}' not found in runs DataFrame columns.")

	use_log_scale = metric in LOG_SCALE_METRICS

	df = _prepare_algorithm_labels(runs_df)

	if valid_only:
		if "is_collision_free" not in df.columns:
			raise ValueError("Column 'is_collision_free' is required when --valid-only is enabled.")
		df = df[df["is_collision_free"].astype("boolean") == True]  # noqa: E712

	if "scenario" not in df.columns:
		raise ValueError("Column 'scenario' is required for scenario-wise boxplots.")

	df = df.dropna(subset=[metric, "scenario", "algorithm"]).copy()
	if df.empty:
		raise ValueError("No rows available to plot after filtering. Check metric and valid-only settings.")

	df["scenario"] = pd.to_numeric(df["scenario"], errors="coerce").astype("Int64")
	df = df.dropna(subset=["scenario"])
	df["scenario"] = df["scenario"].astype(int)

	fig_width = max(10, PLOT_FIGSIZE_WIDTH_FACTOR * df["scenario"].nunique())
	plt.figure(figsize=(fig_width, PLOT_FIGSIZE_HEIGHT), dpi=PLOT_DPI)
	sns.set_theme(style="whitegrid")
	
	# get ordered algos and gradient colors
	present_algo_labels = list(df["algorithm"].unique())
	ordered_algos, colors_list = _build_algo_palette_and_order(present_algo_labels)

	# apply boxplot with ordered hue and palette
	ax = sns.boxplot(
		data=df,
		x="scenario",
		y=metric,
		hue="algorithm",
		hue_order=ordered_algos,
		palette=colors_list,
		linewidth=PLOT_LINEWIDTH,
	)
	ax.set_xlabel("Scenario", fontsize=PLOT_FONTSIZE_LABEL, fontweight="bold")
	# label mentions log scale when applicable
	if use_log_scale:
		ax.set_ylabel(f"{metric} (log scale)", fontsize=PLOT_FONTSIZE_LABEL, fontweight="bold")
	else:
		ax.set_ylabel(metric, fontsize=PLOT_FONTSIZE_LABEL, fontweight="bold")
	ax.tick_params(labelsize=PLOT_FONTSIZE_LABEL - 1)
	if use_log_scale:
		ax.set_yscale("log")

	# append run counts (total + per-algo) to the title for clarity
	total_runs = int(df.shape[0])
	algo_counts = df["algorithm"].value_counts().to_dict()
	counts_suffix = f" ({total_runs//(5 *len(algo_counts))} runs per scenario)"

	if title is None:
		valid_suffix = " (Collision-Free Only)" if valid_only else " (All Runs)"
		title = f"{metric} Distribution by Scenario{valid_suffix}"
	# always append counts
	title = title + counts_suffix
	ax.set_title(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight="bold", pad=18)
	ax.legend(title="Algorithm", fontsize=PLOT_FONTSIZE_LEGEND, title_fontsize=PLOT_FONTSIZE_LEGEND)
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
	plt.close()


def _default_plot_filename(metric: str, valid_only: bool) -> str:
	mode = "valid_only" if valid_only else "all_runs"
	return f"{metric}_boxplot_by_scenario_{mode}.png"


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Create per-scenario performance table and optional boxplots from benchmark runs data."
	)
	parser.add_argument(
		"--benchmark-dir",
		type=str,
		default=str(DEFAULT_BENCHMARK_DIR),
		help="Directory containing pso_runs.parquet (default: src/benchmarking/benchmark_basic).",
	)
	parser.add_argument(
		"--runs-file",
		type=str,
		default="pso_runs.parquet",
		help="Runs file name inside benchmark directory.",
	)
	parser.add_argument(
		"--out-csv",
		type=str,
		default="performance_table.csv",
		help="Output CSV file name for the formatted table.",
	)
	parser.add_argument(
		"--plot",
		action="store_true",
		help="Generate scenario-wise algorithm comparison boxplot.",
	)
	parser.add_argument(
		"--plot-metric",
		type=str,
		default="fitness",
		help="Metric to plot (recommended: fitness).",
	)
	parser.add_argument(
		"--plot-file",
		type=str,
		default=None,
		help="Output image file name for the boxplot.",
	)
	parser.add_argument(
		"--valid-only",
		action="store_true",
		help="For plotting, keep only collision-free runs (recommended for fitness comparison).",
	)
	parser.add_argument(
		"--plot-title",
		type=str,
		default=None,
		help="Optional custom title for the plot.",
	)
	args = parser.parse_args()

	benchmark_dir = Path(args.benchmark_dir)
	runs_path = benchmark_dir / args.runs_file
	if not runs_path.exists():
		raise FileNotFoundError(f"Runs file not found: {runs_path}")

	runs_df = pd.read_parquet(runs_path)
	table_df = _build_table(runs_df)

	out_csv_path = benchmark_dir / args.out_csv
	table_df.to_csv(out_csv_path, index=False)

	print("\nPer-scenario performance table:\n")
	print(table_df.to_string(index=False))
	print(f"\nSaved CSV: {out_csv_path}")

	if args.plot:
		plot_filename = args.plot_file or _default_plot_filename(args.plot_metric, bool(args.valid_only))
		plot_path = benchmark_dir / plot_filename
		_plot_metric_boxplot_by_scenario(
			runs_df=runs_df,
			metric=str(args.plot_metric),
			output_path=plot_path,
			valid_only=bool(args.valid_only),
			title=args.plot_title,
		)
		print(f"Saved boxplot: {plot_path}")


if __name__ == "__main__":
	main()
