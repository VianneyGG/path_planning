import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.environment import PathPlanning, Environment
from src.PSO.Swarm import Swarm
from src.PSO.Path import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#==============================================================================#
#                           Hyperparameters for PSO                            #
#==============================================================================#

number_of_particules = 500
number_of_iterations = 200
number_of_waypoints = 3
waypoints_reset_interval = 50
initial_temperature = 1.0
temperature_decay = 0.99

inertia_weight = 0.5    
best_position_acceleration = 1.5
global_best_position_acceleration = 1.5
length_weight = 1.0
smoothness_weight = 100.0
collision_weight = 4000.0
corner_weight = -100.0
corner_radius = 5.0

#==============================================================================#
#                           PSO Class                                          #
#==============================================================================#

class PSO(PathPlanning):
    def __init__(self, env: Environment)-> None:
        self.environment = env
        self.hyperparameters = {
            'inertia_weight': inertia_weight,
            'best_position_acceleration': best_position_acceleration,
            'global_best_position_acceleration': global_best_position_acceleration,
            'length_weight': length_weight,
            'smoothness_weight': smoothness_weight,
            'collision_weight': collision_weight,
            'corner_weight': corner_weight,
            'corner_radius': corner_radius,
            'prune_straight_angles': False,
            'straight_angle_tolerance': 1e-2,
        }
        self.solution = None
        self._fig = None
        self._ax = None
        
    def plan_path(
        self,
        plot_steps : bool = False,
        reset_waypoints : bool = False,
        simulated_annealing : bool = False,
        dimensional_learning : bool = False,
        *,
        animation_html_path: str | None = None,
        animation_every: int = 1,
        animation_include_plotlyjs: bool | str = True,
        animation_title: str | None = None,
    )-> np.ndarray:
        if animation_every < 1:
            raise ValueError("animation_every must be >= 1")
        plot_interval = 5
        if plot_steps:
            plt.ion()
            if self._fig is None or self._ax is None:
                self._fig, self._ax = plt.subplots(figsize=(8, 6), num='PSO - Path planning')
                # Make sure the window is created without blocking
                plt.show(block=False)
            # Draw once before the heavy initialization so the window is not blank/white
            self.environment.render(
                path=None,
                ax=self._ax,
                clear=True,
                show=False,
                pause=0.05,
                title='Initializing swarm...'
            )
        swarm = Swarm.initialize_swarm(number_of_particules, self.environment, self.hyperparameters, number_of_waypoints)
        # saved best paths across resets
        saved_bests: list[Path] = []
        path_history: list[np.ndarray] | None = [] if animation_html_path else None
        if path_history is not None:
            path_history.append(swarm.get_best_path().get_array_coords().copy())
        temperature = initial_temperature
        
        for iteration in tqdm(range(number_of_iterations), desc="PSO Progress"):
            temperature *= temperature_decay
            if plot_steps and iteration % plot_interval == 0:
                self.environment.render(
                    swarm.get_best_path(),
                    ax=self._ax,
                    clear=False,
                    show=False,
                    pause=0.01,
                    label_waypoints=True,
                    title=f'Iteration: {iteration}/{number_of_iterations}',
                )
            
            if dimensional_learning and iteration % 10 == 0:
                swarm.forward(self.environment, self.hyperparameters, temperature, simulated_annealing, True)
            else:
                swarm.forward(self.environment, self.hyperparameters, temperature, simulated_annealing, False)

            if path_history is not None and iteration % animation_every == 0:
                path_history.append(swarm.get_best_path().get_array_coords().copy())
            
            if reset_waypoints and iteration % waypoints_reset_interval == 0 and iteration >= waypoints_reset_interval:
                temperature = initial_temperature
                # save current best path, then reset swarm to a fresh state
                try:
                    current_best = swarm.get_best_path().copy()
                    saved_bests.append(current_best)
                except Exception:
                    pass
                swarm.reset_waypoints(self.environment, number_of_waypoints, self.hyperparameters)
                print(f'Waypoints reset at iteration {iteration}; saved bests: {len(saved_bests)}')
                
        # Choose best among saved bests and current swarm best
        candidates = [p.copy() for p in saved_bests]
        candidates.append(swarm.get_best_path().copy())

        def path_fitness(path: Path) -> float:
            pcopy = path.copy()
            drop = self.hyperparameters.get('prune_straight_angles', False)
            tol = self.hyperparameters.get('straight_angle_tolerance', 1e-2)
            length = pcopy.total_length()
            smooth = pcopy.smoothness(drop, tol)
            collisions, corners = pcopy.collisions_and_corners(self.environment, self.hyperparameters['corner_radius'])
            return (
                self.hyperparameters['length_weight'] * length +
                self.hyperparameters['smoothness_weight'] * smooth +
                self.hyperparameters['collision_weight'] * collisions +
                self.hyperparameters['corner_weight'] * corners
            )

        best = min(candidates, key=path_fitness)
        self.solution = best

        if animation_html_path is not None:
            if not path_history:
                path_history = [best.get_array_coords().copy()]
            try:
                import plotly.graph_objects as go
            except Exception as exc:
                raise ImportError(
                    "plotly is required for animation export. Install with `pip install plotly`."
                ) from exc

            frames: list[np.ndarray] = []
            for coords in path_history:
                arr = np.asarray(coords, dtype=float)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError("Each path frame must be a (n, 2) array of coordinates")
                frames.append(arr)

            first = frames[0]
            title = animation_title or f"PSO ({number_of_particules}x{number_of_iterations})"

            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=first[:, 0],
                        y=first[:, 1],
                        mode="lines+markers",
                        line=dict(color="#4da3ff", width=3),
                        marker=dict(color="#cfe8ff", size=6),
                        name="Best path",
                    )
                ]
            )

            for obs in self.environment.get_obstacles():
                fig.add_shape(
                    type="rect",
                    x0=obs.x,
                    y0=obs.y,
                    x1=obs.x + obs.lx,
                    y1=obs.y + obs.ly,
                    line=dict(color="#6b2d45", width=2),
                    fillcolor="#2a1b23",
                )

            if self.environment.u1s is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[self.environment.u1s[0]],
                        y=[self.environment.u1s[1]],
                        mode="markers",
                        marker=dict(color="#34d399", size=10),
                        name="Start",
                    )
                )
            if self.environment.u1d is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[self.environment.u1d[0]],
                        y=[self.environment.u1d[1]],
                        mode="markers",
                        marker=dict(color="#60a5fa", size=10),
                        name="Goal",
                    )
                )

            fig.update_layout(
                title=title,
                xaxis=dict(range=[0, float(self.environment.xmax)], autorange=False),
                yaxis=dict(range=[0, float(self.environment.ymax)], autorange=False, scaleanchor="x"),
                plot_bgcolor="#121826",
                paper_bgcolor="#0f1218",
                font=dict(color="#d6deeb"),
                showlegend=True,
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        x=0.05,
                        y=1.1,
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                            ),
                        ],
                    )
                ],
                sliders=[
                    dict(
                        steps=[
                            dict(
                                method="animate",
                                args=[[f"frame_{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                                label=str(i),
                            )
                            for i in range(len(frames))
                        ],
                        x=0.1,
                        y=0.0,
                        len=0.9,
                    )
                ],
            )

            fig.frames = [
                go.Frame(
                    name=f"frame_{i}",
                    data=[
                        go.Scatter(
                            x=f[:, 0],
                            y=f[:, 1],
                            mode="lines+markers",
                            line=dict(color="#4da3ff", width=3),
                            marker=dict(color="#cfe8ff", size=6),
                        )
                    ],
                )
                for i, f in enumerate(frames)
            ]

            fig.write_html(animation_html_path, auto_open=False, include_plotlyjs=animation_include_plotlyjs)

        return best.get_array_coords()
    
    def statistics(self)-> None:
        print('#=================================================#')
        print('#                 PSO Statistics                  #')
        print('#=================================================#')
        print(f'Number of Particules: {number_of_particules}')
        print(f'Number of Iterations: {number_of_iterations}')
        print(f'Number of Waypoints: {number_of_waypoints}')
        print(f'Inertia Weight: {inertia_weight}')
        print(f'Best Position Acceleration: {best_position_acceleration}')
        print(f'Global Best Position Acceleration: {global_best_position_acceleration}')
        print(f'Length Weight: {length_weight}')
        print(f'Smoothness Weight: {smoothness_weight}')
        print(f'Collision Weight: {collision_weight}')
        print(f'Corner Weight: {corner_weight}')
        print('#=================================================#')
        print('#                  Best Path Statistics           #')
        print('#=================================================#')
        print(f'Best Path Fitness: {self.solution.total_length() * length_weight + self.solution.smoothness() * smoothness_weight + self.solution.collisions_and_corners(self.environment, corner_radius)[0] * collision_weight + self.solution.collisions_and_corners(self.environment, corner_radius)[1] * corner_weight }')
        print(f'Best Path Length: {self.solution.total_length()}')
        print(f'Best Path Smoothness: {self.solution.smoothness()}')
        print(f'Best Path Collisions: {self.solution.collisions_and_corners(self.environment, corner_radius)[0]}')
        print(f'Best Path Corners: {self.solution.collisions_and_corners(self.environment, corner_radius)[1]}')
        print('#=================================================#')
    
    def plot_solution(self)-> None:
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(8, 6), num='PSO - Path planning')
        self.environment.render(self.solution, ax=self._ax, clear=True, show=False, title='Best solution')
        plt.ioff()
        plt.show()
         
if __name__ == "__main__":
    env = Environment()
    env.from_file("scenarios/scenario2.txt")
    pso = PSO(env)
    best_path = pso.plan_path(plot_steps=False, reset_waypoints=True, simulated_annealing=True, dimensional_learning=True, animation_html_path="pso_animation.html", animation_every=2)
    pso.plot_solution()
    pso.statistics()