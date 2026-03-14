"""Generate a GIF of PSO evolution for a collision-free ending run.

Usage
-----
    python make_pso_gif.py --scenario 0 --algo RS_SA_PH
    python make_pso_gif.py -s 2 -a RS --fps 12 --every 2
    python make_pso_gif.py -s 0 -a RS_SA_PH --max-runs 30 --output my.gif

The script attempts up to ``--max-runs`` PSO runs and picks the first one that
ends collision-free (CF).  Each GIF frame shows the full particle swarm
(all paths, low alpha) plus the current global best path highlighted.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive back-end for GIF export
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from src.environment import Environment
from src.PSO.pso_solver import PSO
from src.PSO.pso_config import PSOConfig
from src.benchmark.core.algo_profiles import ALGO_FLAGS, ALGO_LABELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLORS = {
    "fig_bg": "#0f1218",
    "ax_bg": "#121826",
    "board": "#8b95a7",
    "grid": "#2a3344",
    "obstacle_fill": "#2a1b23",
    "obstacle_edge": "#6b2d45",
    "particle": "#9aa6b2",
    "best_path": "#4da3ff",
    "best_wp": "#cfe8ff",
    "start": "#34d399",
    "goal": "#60a5fa",
    "text": "#d6deeb",
}

ROOT = Path(__file__).resolve().parent
ARCHIVE_TUNING_PATH = ROOT / "src" / "benchmark" / "archive"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tuned_params(algo: str, scenario: int) -> PSOConfig:
    """Try to load tuned params from archive; fall back to PSOConfig defaults."""
    json_path = ARCHIVE_TUNING_PATH / algo / "tuning" / "tuning_summary.json"
    if json_path.is_file():
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for entry in data.get("per_scenario", []):
                if entry.get("scenario") == scenario:
                    best = entry.get("best_params")
                    if isinstance(best, dict):
                        print(f"[params] Loaded tuned params for {algo} / scenario {scenario}")
                        return PSOConfig.from_dict(best)
        except Exception as exc:
            print(f"[params] Warning: could not parse {json_path}: {exc}")
    else:
        print(f"[params] Warning: no tuning file at {json_path}")
    print("[params] Falling back to PSOConfig defaults")
    return PSOConfig()


def _find_cf_run(
    env: Environment,
    config: PSOConfig,
    max_runs: int,
    every_n: int,
) -> tuple[list[dict], float, float, int, int]:
    """Run all max_runs attempts and return the best CF run (shortest path).

    Returns (snapshots, cpu_time, path_length, cf_count, best_attempt).
    If no CF run is found, returns empty snapshots and zeros.
    """
    best_snapshots: list[dict] = []
    best_cpu_time: float = 0.0
    best_path_length: float = float("inf")
    best_attempt: int = 0
    cf_count: int = 0

    for attempt in range(1, max_runs + 1):
        print(f"\n--- Run {attempt}/{max_runs} ---")
        pso = PSO(env, config)
        t0 = time.perf_counter()
        final_coords, snapshots, is_cf = pso.run_with_snapshots(every_n=every_n, progress=True, verbose=False)
        cpu_time = time.perf_counter() - t0
        if is_cf:
            cf_count += 1
            path_length = float(np.linalg.norm(np.diff(final_coords, axis=0), axis=1).sum())
            print(f"[result] CF OK  length={path_length:.1f}  time={cpu_time:.2f}s")
            if path_length < best_path_length:
                best_path_length = path_length
                best_snapshots = snapshots
                best_cpu_time = cpu_time
                best_attempt = attempt
                print(f"[result] New best (CF #{cf_count}, attempt {attempt})")
        else:
            print("[result] Not CF")

    if not best_snapshots:
        return [], 0.0, 0.0, 0, max_runs

    print(f"\n[best] Best CF run: attempt {best_attempt}  "
          f"length={best_path_length:.1f}  ({cf_count}/{max_runs} runs were CF)")
    return best_snapshots, best_cpu_time, best_path_length, cf_count, best_attempt


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def _print_run_summary(
    algo: str,
    scenario: int,
    config: PSOConfig,
    cpu_time: float,
    path_length: float,
    cf_count: int,
    total_runs: int,
    best_attempt: int,
) -> None:
    """Print a formatted summary of the best CF run to stdout."""
    sep = "\u2501" * 52
    print(f"\n{sep}")
    print(f"  Run Summary  \u2014  {algo} / Scenario {scenario}")
    print(sep)
    print(f"  Best CF run  : attempt {best_attempt} / {total_runs}")
    print(f"  CF rate      : {cf_count}/{total_runs}  ({100*cf_count/total_runs:.0f}%)")
    print(f"  CPU time     : {cpu_time:>10.3f} s")
    print(f"  Path length  : {path_length:>10.1f}")
    print()
    print("  Hyperparameters")
    print("  " + "\u2500" * 48)
    params = dataclasses.asdict(config)
    # Group: core, SA, cooling, fitness weights
    _GROUPS = [
        ("Core", [
            "number_of_particules", "number_of_iterations", "number_of_waypoints",
            "inertia_weight", "inertia_weight_end",
            "best_position_acceleration", "global_best_position_acceleration",
            "reset_waypoints", "reset_number",
        ]),
        ("Simulated Annealing", [
            "simulated_annealing", "initial_temperature", "temperature_decay",
            "pre_heat", "pre_heat_target_acceptance_rate", "pre_heat_max_iterations",
            "pre_heat_learning_rate",
            "controlled_cooling", "acceptance_probability_decay",
        ]),
        ("Extensions", [
            "dimensional_learning", "max_number_of_iterations_without_improvement",
            "adaptive_waypoint_growth", "early_stopping_patience",
            "prune_straight_angles", "vectorized_fitness",
        ]),
        ("Fitness weights", [
            "length_weight", "smoothness_weight", "collision_weight",
            "corner_weight", "corner_radius",
        ]),
    ]
    for group_name, keys in _GROUPS:
        print(f"\n  [{group_name}]")
        for k in keys:
            if k in params:
                v = params[k]
                # Skip None decay when controlled_cooling is off
                if v is None:
                    v = "-"
                elif isinstance(v, float):
                    v = f"{v:.6g}"
                else:
                    v = str(v)
                print(f"    {k:<44} {v}")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# GIF rendering
# ---------------------------------------------------------------------------

def _render_gif(
    snapshots: list[dict],
    env: Environment,
    algo: str,
    scenario: int,
    output_path: str,
    fps: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(COLORS["fig_bg"])
    ax.set_facecolor(COLORS["ax_bg"])

    def _update(frame_idx: int) -> None:
        snap = snapshots[frame_idx]
        ax.clear()
        ax.set_facecolor(COLORS["ax_bg"])
        ax.set_xlim(0, env.xmax)
        ax.set_ylim(0, env.ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors=COLORS["text"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(True, lw=0.5, alpha=0.25, color=COLORS["grid"])

        # Board border
        ax.plot(
            [0, env.xmax, env.xmax, 0, 0],
            [0, 0, env.ymax, env.ymax, 0],
            color=COLORS["board"], lw=2.0, alpha=0.9, zorder=2,
        )

        # Obstacles
        for obs in env.obstacles:
            patch = mpatches.Rectangle(
                (obs.x, obs.y), obs.lx, obs.ly,
                fc=COLORS["obstacle_fill"], ec=COLORS["obstacle_edge"],
                lw=1.5, alpha=0.95, zorder=2,
            )
            ax.add_patch(patch)

        # All particle paths (thin, low alpha)
        for pos in snap["particle_positions"]:
            if len(pos) >= 2:
                ax.plot(
                    pos[:, 0], pos[:, 1],
                    color=COLORS["particle"], lw=0.9, alpha=0.18, zorder=3,
                )

        # Global best path
        best = snap["best_path_coords"]
        ax.plot(
            best[:, 0], best[:, 1],
            color=COLORS["best_path"], lw=2.4, alpha=0.95,
            solid_capstyle="round", zorder=4,
        )
        if best.shape[0] > 2:
            ax.scatter(
                best[1:-1, 0], best[1:-1, 1],
                s=35, c=COLORS["best_wp"],
                edgecolors=COLORS["fig_bg"], linewidths=0.7,
                alpha=0.95, zorder=5,
            )

        # Start & Goal
        ax.scatter(
            env.u1s[0], env.u1s[1],
            s=100, color=COLORS["start"], zorder=6,
            edgecolors=COLORS["fig_bg"], linewidths=1.0,
            label="Start",
        )
        ax.scatter(
            env.u1d[0], env.u1d[1],
            s=160, color=COLORS["goal"], marker="*", zorder=6,
            label="Goal",
        )

        # Text overlay
        n_particles = len(snap["particle_positions"])
        ax.text(
            0.02, 0.98,
            f"Iter {snap['iteration']}  |  Fitness: {snap['best_fitness']:.1f}"
            f"  |  Particles: {n_particles}",
            transform=ax.transAxes, color=COLORS["text"], fontsize=9,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc=COLORS["fig_bg"], alpha=0.80),
            zorder=7,
        )

        # Legend
        leg = ax.legend(loc="upper right", frameon=True, fontsize=8)
        leg.get_frame().set_facecolor(COLORS["fig_bg"])
        leg.get_frame().set_edgecolor(COLORS["grid"])
        leg.get_frame().set_alpha(0.85)
        for txt in leg.get_texts():
            txt.set_color(COLORS["text"])

        ax.set_title(
            f"{ALGO_LABELS[algo]}  -  Scenario {scenario}  -  CF run",
            color=COLORS["text"], fontsize=11, fontweight="bold",
        )
        fig.patch.set_facecolor(COLORS["fig_bg"])

    # Append 2 s of hold frames at the end showing the final path
    hold_frames = fps * 2
    total_frames = len(snapshots) + hold_frames

    def _update_with_hold(frame_idx: int) -> None:
        _update(min(frame_idx, len(snapshots) - 1))

    print(f"\n[gif] Rendering {len(snapshots)} run frames + {hold_frames} hold frames at {fps} fps...")
    ani = FuncAnimation(
        fig, _update_with_hold,
        frames=total_frames,
        interval=1000 // fps,
        repeat=False,
    )

    Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    try:
        ani.save(output_path, writer=PillowWriter(fps=fps))
    except Exception as exc:
        print(f"[gif] Error saving GIF: {exc}")
        print("[gif] Make sure Pillow is installed: pip install pillow")
        sys.exit(1)

    plt.close(fig)
    print(f"[gif] Saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    valid_algos = sorted(ALGO_FLAGS.keys())
    p = argparse.ArgumentParser(
        description="Generate a PSO swarm-evolution GIF for a collision-free run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-s", "--scenario",
        type=int, required=True,
        metavar="ID",
        help="Scenario ID (0-4)",
    )
    p.add_argument(
        "-a", "--algo",
        type=str, required=True,
        metavar="ALGO",
        help=f"Algorithm key. Valid: {', '.join(valid_algos)}",
    )
    p.add_argument(
        "-o", "--output",
        type=str, default=None,
        metavar="PATH",
        help="Output GIF path (default: artifacts/exp/gifs/pso_<algo>_scenario<id>.gif)",
    )
    p.add_argument(
        "--max-runs",
        type=int, default=20,
        metavar="N",
        help="Maximum PSO runs to attempt before giving up",
    )
    p.add_argument(
        "--fps",
        type=int, default=10,
        metavar="N",
        help="GIF playback frames per second",
    )
    p.add_argument(
        "--every",
        type=int, default=1,
        metavar="N",
        help="Capture a snapshot every N iterations (reduces GIF size for long runs)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Validate algo
    valid_algos = sorted(ALGO_FLAGS.keys())
    if args.algo not in ALGO_FLAGS:
        print(f"[error] Unknown algo '{args.algo}'. Valid options: {', '.join(valid_algos)}")
        sys.exit(1)

    # Validate scenario
    if not 0 <= args.scenario <= 4:
        print(f"[error] scenario must be 0-4, got {args.scenario}")
        sys.exit(1)

    # Default output path
    output = args.output or (
        f"artifacts/exp/gifs/pso_{args.algo}_scenario{args.scenario}.gif"
    )

    print(f"[config] algo={args.algo}  scenario={args.scenario}  "
          f"max_runs={args.max_runs}  fps={args.fps}  every={args.every}")
    print(f"[config] output -> {output}")

    # Load environment
    scenario_file = ROOT / "scenarios" / f"scenario{args.scenario}.txt"
    if not scenario_file.is_file():
        print(f"[error] Scenario file not found: {scenario_file}")
        sys.exit(1)
    env = Environment()
    env.from_file(str(scenario_file))
    print(f"[env] Loaded scenario {args.scenario}: "
          f"{env.xmax}x{env.ymax}, {len(env.obstacles)} obstacles")

    # Load config (tuned or default)
    config = _load_tuned_params(args.algo, args.scenario)

    # Find best CF run across all max_runs attempts
    snapshots, cpu_time, path_length, cf_count, best_attempt = _find_cf_run(
        env, config, max_runs=args.max_runs, every_n=args.every
    )
    if not snapshots:
        print(
            f"\n[error] No collision-free run found after {args.max_runs} attempts. "
            "Try --max-runs with a larger value, or pick an easier scenario/algo."
        )
        sys.exit(1)

    # Print summary
    _print_run_summary(
        algo=args.algo,
        scenario=args.scenario,
        config=config,
        cpu_time=cpu_time,
        path_length=path_length,
        cf_count=cf_count,
        total_runs=args.max_runs,
        best_attempt=best_attempt,
    )

    # Render GIF
    _render_gif(
        snapshots=snapshots,
        env=env,
        algo=args.algo,
        scenario=args.scenario,
        output_path=output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

