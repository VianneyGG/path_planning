"""Plot algo comparisons module."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark.core.algo_profiles import ALGO_LABELS, ALGO_PLOT_ORDER

_LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]

# Default artifacts root used for auto-discovery
_ARTIFACTS_ROOT = ROOT / "src" / "benchmark" / "artifacts"

# Color palette: generate a smooth gradient between two pastel greens
def _hex_to_rgb(hex_col: str) -> tuple[int, int, int]:
    h = hex_col.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _make_gradient(start_hex: str, end_hex: str, n: int) -> list[str]:
    start = np.array(_hex_to_rgb(start_hex), dtype=float)
    end = np.array(_hex_to_rgb(end_hex), dtype=float)
    steps = [tuple(np.round(start + (end - start) * t).astype(int)) for t in np.linspace(0.0, 1.0, n)]
    return [_rgb_to_hex(s) for s in steps]


# gentle pastel start -> lemon-ish end
# use a slightly darker start so `Basic` is clearly visible
_PALETTE = _make_gradient("#8DEFC0", "#CFEA6A", len(ALGO_PLOT_ORDER))


def _algo_color(algo: str, all_algos: list[str]) -> str:
    """Return a stable colour for *algo* based on ALGO_PLOT_ORDER position."""
    try:
        idx = ALGO_PLOT_ORDER.index(algo)
        return _PALETTE[idx % len(_PALETTE)]
    except ValueError:
        idx = all_algos.index(algo) if algo in all_algos else 0
        return _PALETTE[idx % len(_PALETTE)]


def _load_combined(artifacts_dir: Path) -> pd.DataFrame:
    """Auto-discover and concatenate all benchmark_loss_curves.parquet files."""
    paths = sorted(artifacts_dir.rglob("benchmark/benchmark_loss_curves.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No benchmark_loss_curves.parquet found under {artifacts_dir}"
        )
    _LOG.info("Loading %d parquet file(s)…", len(paths))
    dfs = [pd.read_parquet(p) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)
    _LOG.info(
        "Combined: %d rows, algos=%s",
        len(combined),
        sorted(combined["algo"].unique()),
    )
    return combined


def _ordered_algos(algos: list[str]) -> list[str]:
    """Return algos sorted by ALGO_PLOT_ORDER, then alphabetically for unknowns."""
    ordered = [a for a in ALGO_PLOT_ORDER if a in algos]
    ordered += sorted(a for a in algos if a not in ALGO_PLOT_ORDER)
    return ordered


def _run_summaries(bench_df: pd.DataFrame):
    """Return (all_runs, cf_runs) per-run summary DataFrames."""
    all_runs = bench_df[["run_id", "scenario", "algo", "elapsed_s"]].drop_duplicates("run_id")
    cf_runs = (
        bench_df.loc[bench_df["iteration"] >= 0, ["run_id", "scenario", "algo", "path_length_m"]]
        .drop_duplicates("run_id")
    )
    return all_runs, cf_runs


# ──────────────────────────────────────────────────────────────────────────────
# Individual plots
# ──────────────────────────────────────────────────────────────────────────────

def _plot_cf_proportion(
    all_runs: pd.DataFrame,
    cf_runs: pd.DataFrame,
    algos: list[str],
    scenarios: list,
    n_runs_per_scenario: int,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n_scen = len(scenarios)
    n_algo = len(algos)
    width = 0.8 / n_algo
    x = np.arange(n_scen)

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_scen), 6), dpi=160)

    for i, algo in enumerate(algos):
        label = ALGO_LABELS.get(algo, algo)
        color = _algo_color(algo, algos)
        n_tot = all_runs[all_runs["algo"] == algo].groupby("scenario")["run_id"].nunique()
        n_cf  = cf_runs[cf_runs["algo"] == algo].groupby("scenario")["run_id"].nunique()
        rates = [float(n_cf.get(s, 0)) / max(1, int(n_tot.get(s, 1))) for s in scenarios]
        offset = (i - (n_algo - 1) / 2) * width
        ax.bar(x + offset, rates, width=width, label=label, color=color, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in scenarios])
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Collision-free proportion")
    ax.set_ylim(0.0, 1.09)
    ax.set_title(f"Collision-free proportion by scenario ({n_runs_per_scenario} runs per scenario)")
    ax.legend(title="Algorithm", loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("Saved: %s", out_path)


def _grouped_boxplot(
    data_by_algo_scenario: dict[str, dict],
    algos: list[str],
    scenarios: list,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Generic grouped boxplot: x = scenario, groups = algorithm."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_scen = len(scenarios)
    n_algo = len(algos)
    width = 0.8 / n_algo

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_scen), 6), dpi=160)

    patches = []
    for i, algo in enumerate(algos):
        label = ALGO_LABELS.get(algo, algo)
        color = _algo_color(algo, algos)
        for j, scen in enumerate(scenarios):
            pos = j + (i - (n_algo - 1) / 2) * width
            data = data_by_algo_scenario.get(algo, {}).get(scen, np.array([]))
            if len(data) == 0:
                continue
            bp = ax.boxplot(
                [data],
                positions=[pos],
                widths=width * 0.85,
                patch_artist=True,
                manage_ticks=False,
                flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0),
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(facecolor=color, alpha=0.85),
                whiskerprops=dict(linewidth=1.0),
                capprops=dict(linewidth=1.0),
            )
            _ = bp  # already styled

        patches.append(mpatches.Patch(facecolor=color, label=label))

    ax.set_xticks(np.arange(n_scen))
    ax.set_xticklabels([str(int(s)) for s in scenarios])
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(handles=patches, title="Algorithm", loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("Saved: %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined_benchmark(
    bench_df: pd.DataFrame,
    out_dir: Path,
    n_runs_label: int | None = None,
    algos_override: list[str] | None = None,
) -> None:
    """Generate the three side-by-side comparison PNGs from *bench_df*."""
    required = {"run_id", "scenario", "algo", "elapsed_s", "path_length_m", "iteration"}
    if bench_df.empty or not required.issubset(bench_df.columns):
        _LOG.warning("DataFrame is empty or missing required columns.")
        return

    all_runs, cf_runs = _run_summaries(bench_df)
    algos_all = _ordered_algos(sorted(all_runs["algo"].unique()))
    if algos_override:
        # keep only algorithms that exist in the data, preserve override order
        algos = [a for a in algos_override if a in algos_all]
        if not algos:
            _LOG.warning("--algos provided but none matched available algorithms; falling back to all algos")
            algos = algos_all
    else:
        algos = algos_all
    scenarios = sorted(all_runs["scenario"].unique())

    # infer run count label
    if n_runs_label is None:
        counts = all_runs.groupby("scenario")["run_id"].nunique()
        n_runs_label = int(counts.max()) if not counts.empty else 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── CF proportion bar chart ──────────────────────────────────────────────
    _plot_cf_proportion(
        all_runs=all_runs,
        cf_runs=cf_runs,
        algos=algos,
        scenarios=scenarios,
        n_runs_per_scenario=n_runs_label,
        out_path=out_dir / "cf_proportion_by_scenario.png",
    )

    # ── Build per-algo/scenario data dicts for boxplots ─────────────────────
    time_data: dict[str, dict] = {}
    len_data:  dict[str, dict] = {}
    for algo in algos:
        time_data[algo] = {}
        len_data[algo]  = {}
        for scen in scenarios:
            sub_all = all_runs[(all_runs["algo"] == algo) & (all_runs["scenario"] == scen)]
            sub_cf  = cf_runs[(cf_runs["algo"] == algo)  & (cf_runs["scenario"] == scen)]
            time_data[algo][scen] = sub_all["elapsed_s"].dropna().to_numpy()
            len_data[algo][scen]  = sub_cf["path_length_m"].dropna().to_numpy()

    # ── Path length grouped boxplot ─────────────────────────────────────────
    _grouped_boxplot(
        data_by_algo_scenario=len_data,
        algos=algos,
        scenarios=scenarios,
        ylabel="length",
        title=f"Path length comparison by scenario ({n_runs_label} runs per scenario)",
        out_path=out_dir / "path_length_by_scenario.png",
    )

    # ── CPU time grouped boxplot ─────────────────────────────────────────────
    _grouped_boxplot(
        data_by_algo_scenario=time_data,
        algos=algos,
        scenarios=scenarios,
        ylabel="time_sec",
        title=f"CPU time comparison by scenario ({n_runs_label} runs per scenario)",
        out_path=out_dir / "cpu_time_by_scenario.png",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine all benchmark_loss_curves.parquet files found under "
            "--artifacts-dir and produce three side-by-side comparison PNGs."
        )
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help=(
            "Root folder containing per-algo artifact sub-folders "
            "(default: src/benchmark/artifacts)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="src/benchmark/artifacts/plots",
        help="Output directory for the generated PNGs.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--algos",
        type=str,
        default=None,
        help=(
            "Comma-separated list of algorithm keys to include (e.g. 'RS_SA_noCC,RS_SA_PH,RS_SA_CC'). "
            "If omitted all discovered algorithms are plotted."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else _ARTIFACTS_ROOT
    if not artifacts_dir.is_absolute():
        artifacts_dir = ROOT / artifacts_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    bench_df = _load_combined(artifacts_dir)
    algos_override = [a.strip() for a in args.algos.split(",")] if args.algos else None
    plot_combined_benchmark(bench_df=bench_df, out_dir=out_dir, algos_override=algos_override)


if __name__ == "__main__":
    main()
