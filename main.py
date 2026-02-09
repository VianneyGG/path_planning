"""Run RRT or PSO path planning on a scenario. Usage: python main.py [rrt|pso] [scenario_id]"""

import sys
import numpy as np

from src.environment import Environment
from src.RRT.RRT import RRT, _distance, multi_robot_planner, export_rrt_animation_html


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


def run_pso(scenario: int = 0, animation_path: str | None = None):
    from src.PSO.PSO import PSO
    env = Environment()
    env.from_file(f"scenarios/scenario{scenario}.txt")
    pso = PSO(env)
    pso.plan_path(plot_steps=False, animation_html_path=animation_path)
    if animation_path:
        print(f"Animation saved to {animation_path}")
    pso.plot_solution()
    pso.statistics()


if __name__ == "__main__":
    method = (sys.argv[1] if len(sys.argv) > 1 else "rrt").lower()
    scenario = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    animate = "--animate" in sys.argv or "-a" in sys.argv
    animation_file = "rrt_animation.html" if method in ("rrt", "multi", "tworobot", "multi_robot") else "pso_animation.html"
    if animate:
        try:
            i = sys.argv.index("--animate" if "--animate" in sys.argv else "-a")
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
                animation_file = sys.argv[i + 1]
        except ValueError:
            pass

    if method == "rrt":
        run_rrt(scenario, show=True, animation_path=animation_file if animate else None)
    elif method == "pso":
        run_pso(scenario, animation_path=animation_file if animate else None)
    elif method in ("multi", "tworobot", "multi_robot"):
        run_multi_robot(scenario, show=True, animation_path=animation_file if animate else None)
    else:
        print("Usage: python main.py [rrt|pso|multi] [scenario_id] [--animate [output.html]]")
        print("  rrt   - single robot RRT")
        print("  pso   - single robot PSO")
        print("  multi - two robots (prioritized RRT: Robot 1 then Robot 2 with dynamic obstacle)")
        print("  e.g. python main.py pso 0 --animate           # PSO animation")
        print("       python main.py rrt 0 --animate           # RRT single-robot animation")
        print("       python main.py multi 0 --animate out.html # two-robot RRT animation")
        sys.exit(1)
