"""Run RRT or PSO path planning on a scenario. Usage: python main.py [rrt|pso] [scenario_id]"""

import sys
import numpy as np

from src.environment import Environment
from src.RRT.RRT import RRT, _distance


def run_rrt(scenario: int = 0, show: bool = True):
    env = Environment()
    env.from_file(f"scenarios/scenario{scenario}.txt")
    rrt = RRT(env.u1s, env.u1d, env, delta_s=40.0, delta_r=120.0, n_iter=2000, p=0.2, smooth=True)
    path = rrt.run_algorithm()
    length = sum(_distance(path[i - 1], path[i]) for i in range(1, len(path)))
    collisions = sum(env.check_line_collision(np.array(path[i - 1]), np.array(path[i])) for i in range(1, len(path)))
    print(f"RRT: {len(path)} waypoints, length={length:.1f}, collisions={collisions}")
    if show:
        env.render(path)


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
    # Optional: python main.py pso 0 --animate [pso_animation.html]
    animate = "--animate" in sys.argv or "-a" in sys.argv
    animation_file = "pso_animation.html"
    if animate:
        try:
            i = sys.argv.index("--animate" if "--animate" in sys.argv else "-a")
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
                animation_file = sys.argv[i + 1]
        except ValueError:
            pass

    if method == "rrt":
        run_rrt(scenario, show=True)
    elif method == "pso":
        run_pso(scenario, animation_path=animation_file if animate else None)
    else:
        print("Usage: python main.py [rrt|pso] [scenario_id] [--animate [output.html]]")
        print("  e.g. python main.py pso 0 --animate           # run PSO, save animation to pso_animation.html")
        print("       python main.py pso 2 --animate out.html  # run PSO on scenario 2, save to out.html")
        sys.exit(1)
