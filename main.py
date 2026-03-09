"""Path planning demo — RRT or PSO.

Examples:
  python main.py rrt
  python main.py rrt --scenario 2 --animate rrt.html
  python main.py multi --scenario 1
  python main.py pso
  python main.py pso --heuristic sa_cc --runs 400 --scenario 0
  python main.py pso -H sa_ph -r 300 -s 2 --animate
"""

import argparse
import sys
import numpy as np

from src.environment import Environment
from src.RRT.RRT import RRT, _distance, multi_robot_planner, export_rrt_animation_html

# ---------------------------------------------------------------------------
# PSO heuristic aliases → internal ALGO_FLAGS key
# ---------------------------------------------------------------------------

_HEURISTIC_ALIASES: dict[str, str] = {
    "vanilla":       "vanilla",
    "basic":         "vanilla",
    "rs":            "RS",
    "sa":            "RS_SA_noCC",
    "sa_nocc":       "RS_SA_noCC",
    "rs_sa_nocc":    "RS_SA_noCC",
    "sa_dl":         "RS_SA_noCC_DL",
    "sa_nocc_dl":    "RS_SA_noCC_DL",
    "rs_sa_nocc_dl": "RS_SA_noCC_DL",
    "sa_cc":         "RS_SA_CC",
    "rs_sa_cc":      "RS_SA_CC",
    "sa_ph":         "RS_SA_PH",
    "rs_sa_ph":      "RS_SA_PH",
    "sa_cc_dl":      "RS_SA_CC_DL",
    "rs_sa_cc_dl":   "RS_SA_CC_DL",
}


def _resolve_heuristic(name: str) -> str:
    key = name.lower()
    if key not in _HEURISTIC_ALIASES:
        choices = ", ".join(sorted(_HEURISTIC_ALIASES))
        raise argparse.ArgumentTypeError(f"Unknown heuristic '{name}'. Available: {choices}")
    return _HEURISTIC_ALIASES[key]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_rrt(scenario: int = 0, show: bool = True, animation_path: str | None = None):
    env = Environment()
    env.from_file(f"scenarios/scenario{scenario}.txt")
    rrt = RRT(env.u1s, env.u1d, env, delta_s=40.0, delta_r=120.0, n_iter=2000, p=0.2, smooth=True)
    path = rrt.run_algorithm()
    length = sum(_distance(path[i - 1], path[i]) for i in range(1, len(path)))
    collisions = sum(env.check_line_collision(np.array(path[i - 1]), np.array(path[i])) for i in range(1, len(path)))
    print(f"RRT: {len(path)} waypoints, length={length:.1f}, collisions={collisions}")
    if animation_path:
        export_rrt_animation_html(env, path, path2=None, html_path=animation_path, title=f"RRT (scenario {scenario})")
        print(f"Animation saved to {animation_path}")
    if show:
        env.render(path)


def run_multi_robot(scenario: int = 0, show: bool = True, animation_path: str | None = None):
    """Prioritized planning: RRT for Robot 1, then RRT with Robot 1 as dynamic obstacle for Robot 2."""
    env = Environment()
    env.from_file(f"scenarios/scenario{scenario}.txt")
    path1, path2 = multi_robot_planner(env, progress_bar=True)
    len1 = sum(_distance(path1[i - 1], path1[i]) for i in range(1, len(path1)))
    len2 = sum(_distance(path2[i - 1], path2[i]) for i in range(1, len(path2)))
    print(f"Robot 1: {len(path1)} waypoints, length={len1:.1f}")
    print(f"Robot 2: {len(path2)} waypoints, length={len2:.1f}")
    if animation_path:
        export_rrt_animation_html(env, path1, path2=path2, html_path=animation_path, title=f"Multi-robot (scenario {scenario})")
        print(f"Animation saved to {animation_path}")
    if show:
        env.render(path1, path2=path2, title=f"Multi-robot (scenario {scenario})")


def run_pso(
    scenario: int = 0,
    heuristic: str = "RS_SA_noCC",
    n_runs: int = 1,
    animation_path: str | None = None,
):
    from src.PSO.PSO import PSO
    from src.benchmark.core.algo_profiles import get_algo_flags, ALGO_LABELS

    label = ALGO_LABELS.get(heuristic, heuristic)
    print(f"PSO — heuristic: {label}, runs: {n_runs}, scenario: {scenario}, workers: 12")

    env = Environment()
    env.from_file(f"scenarios/scenario{scenario}.txt")

    config = get_algo_flags(heuristic)
    config["parallel_fitness_workers"] = 12

    best_pso: PSO | None = None
    best_fitness = float("inf")

    for run in range(n_runs):
        if n_runs > 1:
            print(f"\nRun {run + 1}/{n_runs}")
        pso = PSO(env, config=config)
        pso.plan_path(plot_steps=False, animation_html_path=animation_path if run == n_runs - 1 else None)
        fitness = pso.path_fitness(pso.solution) if pso.solution is not None else float("inf")
        if fitness < best_fitness:
            best_fitness = fitness
            best_pso = pso

    if best_pso is not None:
        if n_runs > 1:
            print(f"\nBest run fitness: {best_fitness:.4f}")
        if animation_path:
            print(f"Animation saved to {animation_path}")
        best_pso.plot_solution()
        best_pso.statistics()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Path planning demo — RRT or PSO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py rrt
  python main.py rrt --scenario 2 --animate rrt.html
  python main.py multi --scenario 1 --animate
  python main.py pso
  python main.py pso --heuristic sa_cc --iterations 400 --scenario 0
  python main.py pso -H sa_ph -i 300 -s 2 --animate
        """,
    )
    sub = parser.add_subparsers(dest="method", metavar="METHOD")
    sub.required = True

    # --- rrt ---
    p_rrt = sub.add_parser("rrt", help="Single-robot RRT")
    p_rrt.add_argument("--scenario", "-s", type=int, default=0, metavar="N",
                       help="Scenario index (default: 0)")
    p_rrt.add_argument("--animate", "-a", nargs="?", const="rrt_animation.html", metavar="FILE",
                       help="Save animation HTML (default: rrt_animation.html)")

    # --- multi ---
    p_multi = sub.add_parser("multi", aliases=["tworobot", "multi_robot"],
                              help="Two-robot prioritized RRT")
    p_multi.add_argument("--scenario", "-s", type=int, default=0, metavar="N",
                         help="Scenario index (default: 0)")
    p_multi.add_argument("--animate", "-a", nargs="?", const="rrt_animation.html", metavar="FILE",
                         help="Save animation HTML (default: rrt_animation.html)")

    # --- pso ---
    p_pso = sub.add_parser(
        "pso",
        help="PSO path planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "PSO path planning with configurable heuristic.\n\n"
            "heuristics:\n"
            "  vanilla / basic     Basic PSO\n"
            "  rs                  Reset-waypoints\n"
            "  sa          (default) SA without controlled cooling\n"
            "  sa_dl               SA + Dimensional Learning\n"
            "  sa_cc               SA + Controlled Cooling\n"
            "  sa_ph               SA + Pre-Heat\n"
            "  sa_cc_dl            SA + CC + DL\n"
        ),
    )
    p_pso.add_argument("--heuristic", "-H", type=_resolve_heuristic, default="RS_SA_noCC",
                       metavar="NAME", help="PSO heuristic (default: sa)")
    p_pso.add_argument("--runs", "-r", type=int, default=1, metavar="N",
                       help="Number of PSO runs (default: 1); best result is kept")
    p_pso.add_argument("--scenario", "-s", type=int, default=0, metavar="N",
                       help="Scenario index (default: 0)")
    p_pso.add_argument("--animate", "-a", nargs="?", const="pso_animation.html", metavar="FILE",
                       help="Save animation HTML (default: pso_animation.html)")

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.method == "rrt":
        run_rrt(args.scenario, show=True, animation_path=args.animate)
    elif args.method in ("multi", "tworobot", "multi_robot"):
        run_multi_robot(args.scenario, show=True, animation_path=args.animate)
    elif args.method == "pso":
        run_pso(args.scenario, args.heuristic, args.runs, animation_path=args.animate)
