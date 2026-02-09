"""
Compare RRT (basic, with path optimization, with intelligent sampling, and combined)
vs PSO in terms of path length and convergence speed (Q20, Q22).

Usage (from project root):
  python -m src.benchmarking.compare
  python -m src.benchmarking.compare --scenarios 0 1 2
  python -m src.benchmarking.compare --runs 5 --no-pso
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import Environment
from src.RRT.RRT import RRT, _distance


def path_length_and_collisions(path, env):
    """Return (total_length, collision_count) for a path (list of (x,y))."""
    if len(path) <= 1:
        return 0.0, 0
    length = sum(_distance(path[i - 1], path[i]) for i in range(1, len(path)))
    collisions = sum(
        env.check_line_collision(np.array(path[i - 1]), np.array(path[i]))
        for i in range(1, len(path))
    )
    return length, collisions


def run_rrt_bench(env, config: dict, seed: int):
    """Run RRT once with given config and seed. Return (path_length, time_sec, collisions, n_waypoints)."""
    np.random.seed(seed)
    rrt = RRT(
        env.u1s,
        env.u1d,
        env,
        delta_s=config.get("delta_s", 40.0),
        delta_r=config.get("delta_r", 120.0),
        n_iter=config.get("n_iter", 2000),
        p=config.get("p", 0.0),
        smooth=config.get("smooth", False),
    )
    t0 = time.perf_counter()
    path = rrt.run_algorithm(progress_bar=False)
    t1 = time.perf_counter()
    length, collisions = path_length_and_collisions(path, env)
    return length, t1 - t0, collisions, len(path)


def run_pso_bench(env, seed: int):
    """Run PSO once with given seed (core algorithm only, no progress bar/plotting). Return (path_length, time_sec, collisions, n_waypoints)."""
    from src.PSO.PSO import PSO

    np.random.seed(seed)
    pso = PSO(env)
    t0 = time.perf_counter()
    pso.run()
    t1 = time.perf_counter()
    if pso.solution is None:
        return float("nan"), t1 - t0, -1, 0
    length = pso.solution.total_length()
    collisions, _ = pso.solution.collisions_and_corners(pso.environment, pso.hyperparameters["corner_radius"])
    return length, t1 - t0, collisions, len(pso.solution.get_waypoints())


RRT_CONFIGS = {
    "RRT (basic)": {"p": 0.0, "smooth": False},
    "RRT + path optimization": {"p": 0.0, "smooth": True},
    "RRT + intelligent sampling": {"p": 0.2, "smooth": False},
    "RRT + both": {"p": 0.2, "smooth": True},
}


def run_benchmark(scenario_id: int, n_runs: int, include_pso: bool, seeds: list[int] | None = None):
    """Run all methods on one scenario. Return list of (method_name, results_per_run)."""
    env = Environment()
    path_file = ROOT / "scenarios" / f"scenario{scenario_id}.txt"
    env.from_file(path_file)

    if seeds is None:
        seeds = [42 + i for i in range(n_runs)]

    results = []

    for name, config in RRT_CONFIGS.items():
        runs = []
        for seed in seeds:
            length, time_sec, collisions, n_wp = run_rrt_bench(env, config, seed)
            runs.append({"length": length, "time": time_sec, "collisions": collisions, "waypoints": n_wp})
        results.append((name, runs))

    if include_pso:
        runs = []
        for seed in seeds:
            length, time_sec, collisions, n_wp = run_pso_bench(env, seed)
            runs.append({"length": length, "time": time_sec, "collisions": collisions, "waypoints": n_wp})
        results.append(("PSO", runs))

    return results


def print_results_table(rows: list[tuple], headers: list[str], title: str | None = None) -> None:
    """Print a Rich table with the given rows and headers."""
    if not rows:
        return
    table = Table(show_header=True, header_style="bold cyan", title=title)
    for h in headers:
        table.add_column(h)
    for row in rows:
        table.add_row(*[str(c) for c in row])
    Console().print(table)


def main():
    parser = argparse.ArgumentParser(description="Compare RRT variants and PSO (path length, time).")
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Scenario IDs")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per method (for averaging)")
    parser.add_argument("--no-pso", action="store_true", help="Skip PSO (faster)")
    parser.add_argument("--csv", type=str, default="", help="Write summary to CSV file")
    args = parser.parse_args()

    console = Console()
    all_summaries = []

    for scenario_id in args.scenarios:
        path_file = ROOT / "scenarios" / f"scenario{scenario_id}.txt"
        if not path_file.exists():
            console.print(f"[yellow]Scenario {scenario_id} not found, skipping.[/yellow]")
            continue

        console.print()
        console.print(Panel(f"Scenario [bold]{scenario_id}[/bold]", title="Benchmark", border_style="blue"))

        results = run_benchmark(scenario_id, args.runs, include_pso=not args.no_pso)

        # Per-method summary: mean length, mean time, mean collisions
        rows = []
        for method_name, runs in results:
            lengths = [r["length"] for r in runs]
            times = [r["time"] for r in runs]
            colls = [r["collisions"] for r in runs]
            mean_len = np.mean(lengths)
            std_len = np.std(lengths) if len(lengths) > 1 else 0.0
            mean_time = np.mean(times)
            mean_coll = np.mean(colls)
            rows.append((
                method_name,
                f"{mean_len:.1f} ± {std_len:.1f}" if std_len > 0 else f"{mean_len:.1f}",
                f"{mean_time:.2f}s",
                f"{mean_coll:.0f}",
            ))
            all_summaries.append({
                "scenario": scenario_id,
                "method": method_name,
                "path_length_mean": mean_len,
                "path_length_std": std_len,
                "time_mean": mean_time,
                "collisions_mean": mean_coll,
            })

        print_results_table(rows, ["Method", "Path length", "Time", "Collisions"], title=f"Scenario {scenario_id}")

    if args.csv:
        import csv
        with open(ROOT / args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scenario", "method", "path_length_mean", "path_length_std", "time_mean", "collisions_mean"])
            w.writeheader()
            w.writerows(all_summaries)
        console.print(f"\n[green]Wrote summary to {args.csv}[/green]")


if __name__ == "__main__":
    main()
