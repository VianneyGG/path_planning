from __future__ import annotations
from typing import Any

import numpy as np
from tqdm import tqdm

from src.environment import Environment
from src.PSO.Particule import Particule
from src.PSO.Path import Path
from src.PSO.Swarm import Swarm
from src.PSO.Config import PSOConfig

class PSO:
    def __init__(self, env: Environment, config: PSOConfig | dict[str, Any] | None = None) -> None:
        self.environment = env

        if isinstance(config, PSOConfig):
            self.config: PSOConfig = config

        else:
            self.config = PSOConfig.from_dict(config) if config else PSOConfig()


        self.hyperparameters = {
            "inertia_weight": self.config.inertia_weight,
            "inertia_weight_end": self.config.inertia_weight_end,
            "best_position_acceleration": self.config.best_position_acceleration,
            "global_best_position_acceleration": self.config.global_best_position_acceleration,
            "length_weight": self.config.length_weight,
            "smoothness_weight": self.config.smoothness_weight,
            "collision_weight": self.config.collision_weight,
            "corner_weight": self.config.corner_weight,
            "corner_radius": self.config.corner_radius,
            "prune_straight_angles": self.config.prune_straight_angles,
            "straight_angle_tolerance": self.config.straight_angle_tolerance,
            "parallel_fitness_workers": self.config.parallel_fitness_workers,
            "reuse_fitness_thread_pool": self.config.reuse_fitness_thread_pool,
            "corner_check_stride": self.config.corner_check_stride,
            "max_number_of_iterations_without_improvement": self.config.max_number_of_iterations_without_improvement,
            "vectorized_fitness": self.config.vectorized_fitness,
        }


        self.solution: Path | None = None

        self._fig = None

        self._ax = None

    def path_fitness(self, path: Path) -> float:
        pcopy = path.copy()

        drop = self.hyperparameters.get("prune_straight_angles", False)

        tol = self.hyperparameters.get("straight_angle_tolerance", 1e-2)

        length = pcopy.total_length()

        smooth = pcopy.smoothness(drop, tol)

        collisions, corners = pcopy.collisions_and_corners(

            self.environment, self.hyperparameters["corner_radius"]
        )

        return (

            self.hyperparameters["length_weight"] * length

            + self.hyperparameters["smoothness_weight"] * smooth

            + self.hyperparameters["collision_weight"] * collisions

            + self.hyperparameters["corner_weight"] * corners
        )

    def _initialize_swarm(self) -> Swarm:
        return Swarm.initialize_swarm(

            int(self.config.number_of_particules),

            self.environment,

            self.hyperparameters,

            int(self.config.number_of_waypoints),
        )

    def _pre_heat(
        self,
        swarm: Swarm,
        *,
        progress: bool,
        verbose: bool,
        max_iterations: int | None = None,
        ) -> tuple[float, float]:

        temperature = float(self.config.initial_temperature)

        acceptance_probability = 0.1

        positive_fitness_variations = 0.0


        if verbose:

            print("Pre-heating phase...")


        _ph_max_iters = max_iterations if max_iterations is not None else int(self.config.pre_heat_max_iterations)
        iterator = range(_ph_max_iters)

        if progress and verbose:

            iterator = tqdm(iterator, desc="Pre-heating", leave=False)


        for pre_heat_iteration in iterator:

            particule = Particule.initialize_particule(

                self.environment,

                self.hyperparameters,

                int(self.config.number_of_waypoints),
            )


            ref_fitness = swarm.global_best_position_fitness

            candidate_prob = (

                1.0

                if particule.fitness < ref_fitness

                else np.exp((ref_fitness - particule.fitness) / max(temperature, 1e-9)) # avoid division by zero
            )

            alpha = float(self.config.pre_heat_learning_rate)

            acceptance_probability = alpha * acceptance_probability + (1 - alpha) * candidate_prob

            positive_fitness_variations = alpha * positive_fitness_variations + (1 - alpha) * abs(
                ref_fitness - particule.fitness
            )


            temperature = max(

                temperature,

                -1 * positive_fitness_variations / np.log(max(acceptance_probability, 1e-9)), # make sure to avoid log(0) and division by zero
            )


            if particule.fitness < swarm.global_best_position_fitness:

                swarm.global_best_position_fitness = particule.fitness

                swarm.global_best_position = particule.get_position().copy()

                swarm.best_path = particule.path.copy()


            if acceptance_probability >= float(self.config.pre_heat_target_acceptance_rate):

                if verbose:
                    print(

                        f"Pre-heating ended at iteration {pre_heat_iteration} with "

                        f"acceptance rate {acceptance_probability:.2f}, temperature {temperature:.4f}"
                    )

                break

            if verbose and pre_heat_iteration == _ph_max_iters - 1:
                print(

                    f"Pre-heating ended after max iterations with acceptance rate "

                    f"{acceptance_probability:.2f}, temperature {temperature:.4f}"
                )


        return temperature, acceptance_probability

    def _run_core(
        self,
        *,
        progress: bool,
        verbose: bool,
        capture_history: bool,
        animation_every: int,
        iteration_callback: Any = None,
        swarm_snapshot_callback: Any = None,
        snapshot_every: int = 1,
        ) -> tuple[Path, list[np.ndarray] | None]:

        swarm = self._initialize_swarm()

        saved_bests: list[Path] = []


        path_history: list[np.ndarray] | None = [] if capture_history else None

        if path_history is not None:

            path_history.append(swarm.get_best_path().get_array_coords().copy())


        temperature = float(self.config.initial_temperature)

        acceptance_probability = 0.1

        pre_heat_initial_temperature = temperature

        pre_heat_initial_probability = acceptance_probability


        if bool(self.config.pre_heat):

            temperature, acceptance_probability = self._pre_heat(

                swarm, progress=progress, verbose=verbose
            )

            pre_heat_initial_temperature = temperature

            pre_heat_initial_probability = acceptance_probability

            # Sync all-time bests from any discoveries made during pre-heat.
            # Without this, if the main loop never beats the pre-heat optimum,
            # get_best_path() would return the weaker initial-swarm path.
            if swarm.global_best_position_fitness < swarm._all_time_best_fitness:
                swarm._all_time_best_fitness   = swarm.global_best_position_fitness
                swarm._all_time_best_position  = swarm.global_best_position.copy()
                swarm._all_time_best_path      = swarm.best_path.copy()


        if bool(self.config.controlled_cooling) and verbose:

            print("Starting PSO with controlled cooling...")


        positive_fitness_variations = 0.0

        # Linear inertia weight decay (LDIW-PSO)
        _w_start = float(self.hyperparameters["inertia_weight"])
        _w_end   = float(self.hyperparameters.get("inertia_weight_end", _w_start))
        _n_iters = max(1, int(self.config.number_of_iterations))

        # Early stopping state
        _patience = max(0, int(self.config.early_stopping_patience))
        _no_improve_count = 0
        # Track the all-time best (never degraded by SA or waypoint resets)
        # so patience counts genuine stagnation only.
        _prev_best_fitness = float(swarm._all_time_best_fitness)

        # Adaptive waypoint count: persists across resets (grows when still colliding)
        _current_nwp = int(self.config.number_of_waypoints)

        iterations = range(_n_iters)
        reset_interval = max(1, _n_iters // max(1, int(self.config.reset_number)))

        if progress:

            iterations = tqdm(iterations, desc="PSO Progress")


        for iteration in iterations:

            # Linear inertia decay: w(t) = w_start + (w_end - w_start) * t / (T-1)
            if _w_start != _w_end:
                self.hyperparameters["inertia_weight"] = (
                    _w_start + (_w_end - _w_start) * iteration / max(1, _n_iters - 1)
                )

            swarm.forward(

                self.environment,

                self.hyperparameters,
                iteration,

                temperature,

                bool(self.config.simulated_annealing),

                bool(self.config.dimensional_learning),
            )


            if bool(self.config.controlled_cooling):

                positive_fitness_variations = 0.95 * positive_fitness_variations + 0.05 * abs(

                    swarm.global_best_position_fitness - swarm.particules[0].fitness
                )

                acceptance_probability -= float(self.config.acceptance_probability_decay)

                temperature = max(

                    temperature * float(self.config.cc_temperature_decay),

                    -1 * positive_fitness_variations / np.log(max(acceptance_probability, 1e-9)), # make sure to avoid log(0) and division by zero
                )

            else:

                temperature *= float(self.config.temperature_decay)

            # Early stopping: halt when global best stagnates
            if _patience > 0:
                if _prev_best_fitness - swarm._all_time_best_fitness > 1e-8:
                    _prev_best_fitness = swarm._all_time_best_fitness
                    _no_improve_count = 0
                else:
                    _no_improve_count += 1
                    if _no_improve_count >= _patience:
                        if verbose:
                            print(f"Early stopping at iteration {iteration} "
                                  f"(no improvement for {_patience} iterations)")
                        break

            if iteration_callback is not None:
                try:
                    # Include current collision count so consumers can track
                    # when the run first becomes collision-free.
                    _cb_collisions = int(
                        swarm.get_best_path().collisions_and_corners(
                            self.environment,
                            self.hyperparameters["corner_radius"],
                            check_corners=False,
                        )[0]
                    )
                    iteration_callback(
                        {
                            "iteration": int(iteration),
                            "best_fitness": float(swarm._all_time_best_fitness),
                            "collisions": _cb_collisions,
                            "is_collision_free": _cb_collisions == 0,
                            "temperature": float(temperature),
                            "acceptance_probability": float(acceptance_probability),
                        }
                    )
                except Exception:
                    pass


            if swarm_snapshot_callback is not None and iteration % snapshot_every == 0:
                swarm_snapshot_callback(swarm, iteration)

            if path_history is not None and iteration % animation_every == 0:

                path_history.append(swarm.get_best_path().get_array_coords().copy())

            if (

                bool(self.config.reset_waypoints)

                and iteration % reset_interval == 0

                and iteration >= reset_interval

            ):

                if bool(self.config.pre_heat):
                    _mini_iters = max(1, int(self.config.pre_heat_max_iterations) // 2)
                    temperature, acceptance_probability = self._pre_heat(
                        swarm, progress=False, verbose=verbose, max_iterations=_mini_iters,
                    )
                    pre_heat_initial_temperature = temperature
                    pre_heat_initial_probability = acceptance_probability
                else:
                    temperature = float(self.config.initial_temperature)
                    acceptance_probability = 0.1

                try:

                    saved_bests.append(swarm.get_best_path().copy())

                except Exception:
                    pass

                # Adaptive waypoint growth: add one waypoint on reset when the
                # all-time best path still has collisions (gives the planner
                # more degrees of freedom to route around obstacles).
                # NOTE: _current_nwp is declared before the loop so growth
                # accumulates across multiple resets.
                if bool(self.config.adaptive_waypoint_growth):
                    try:
                        _best_colls, _ = swarm._all_time_best_path.collisions_and_corners(
                            self.environment,
                            self.hyperparameters["corner_radius"],
                            check_corners=False,
                        )
                        if _best_colls > 0:
                            _cap = max(_current_nwp, int(self.config.max_waypoints_cap))
                            _current_nwp = min(_current_nwp + 1, _cap)
                    except Exception:
                        pass

                swarm.reset_waypoints(
                    self.environment,
                    _current_nwp,
                    self.hyperparameters,
                )

                if verbose:

                    print(f"Waypoints reset at iteration {iteration}; saved bests: {len(saved_bests)}")


        candidates = [p.copy() for p in saved_bests]

        candidates.append(swarm.get_best_path().copy())

        best = min(candidates, key=self.path_fitness)

        self.solution = best

        return best, path_history

    def run(
        self,
        *,
        progress: bool = False,
        verbose: bool = False,
        iteration_callback: Any = None,
    ) -> np.ndarray:
        """

        Run PSO for benchmarking (default silent).
        """

        best, _ = self._run_core(
            progress=progress,

            verbose=verbose,

            capture_history=False,

            animation_every=1,

            iteration_callback=iteration_callback,
        )

        return best.get_array_coords()

    def run_with_snapshots(
        self,
        *,
        every_n: int = 1,
        progress: bool = True,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[dict], bool]:
        """Run PSO and collect per-iteration swarm snapshots for GIF rendering.

        Returns
        -------
        final_coords : np.ndarray
            Best path coordinates at end of run.
        snapshots : list[dict]
            One dict per captured iteration with keys:
            ``iteration``, ``particle_positions``, ``best_path_coords``, ``best_fitness``.
        is_cf : bool
            True when the final solution has zero collisions.
        """
        snapshots: list[dict] = []

        def _capture(swarm_obj: Any, iter_num: int) -> None:
            snapshots.append({
                "iteration": iter_num,
                "particle_positions": [p.position.copy() for p in swarm_obj.particules],
                "best_path_coords": swarm_obj.get_best_path().get_array_coords().copy(),
                "best_fitness": float(swarm_obj._all_time_best_fitness),
            })

        best, _ = self._run_core(
            progress=progress,
            verbose=verbose,
            capture_history=False,
            animation_every=1,
            swarm_snapshot_callback=_capture,
            snapshot_every=max(1, int(every_n)),
        )

        is_cf = bool(
            best.collisions_and_corners(
                self.environment,
                self.hyperparameters["corner_radius"],
                check_corners=False,
            )[0] == 0
        )

        # Ensure at least one snapshot exists (edge case: zero iterations)
        if not snapshots:
            _capture_swarm_dummy = type("_D", (), {
                "particules": [],
                "get_best_path": lambda self: best,
                "_all_time_best_fitness": self.path_fitness(best),
            })()
            snapshots.append({
                "iteration": 0,
                "particle_positions": [],
                "best_path_coords": best.get_array_coords().copy(),
                "best_fitness": self.path_fitness(best),
            })

        return best.get_array_coords(), snapshots, is_cf

    def plan_path(
        self,
        plot_steps: bool = False,
        *,
        animation_html_path: str | None = None,
        animation_every: int = 1,
        animation_include_plotlyjs: bool | str = True,
        animation_title: str | None = None,
        ) -> np.ndarray:

        if animation_every < 1:

            raise ValueError("animation_every must be >= 1")


        if plot_steps:

            import matplotlib.pyplot as plt

            plt.ion()

            if self._fig is None or self._ax is None:

                self._fig, self._ax = plt.subplots(figsize=(8, 6), num="PSO - Path planning")

                plt.show(block=False)

            self.environment.render(path=None, ax=self._ax, title="Initializing swarm...")

            plt.pause(0.05)

        best, path_history = self._run_core(

            progress=True,

            verbose=True,

            capture_history=animation_html_path is not None,

            animation_every=animation_every,

            iteration_callback=None,
        )

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

            title = animation_title or f"PSO ({self.config.number_of_particules}x{self.config.number_of_iterations})"


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

                    layer="below",
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

            xmax = float(self.environment.xmax)

            ymax = float(self.environment.ymax)

            fig.update_layout(

                title=dict(text=title, font=dict(size=18)),

                margin=dict(t=100, b=80, l=80, r=80),

                xaxis=dict(

                    range=[0, xmax],

                    autorange=False,

                    fixedrange=True,

                    constrain="domain",

                    tickfont=dict(size=12),
                ),

                yaxis=dict(

                    range=[0, ymax],

                    autorange=False,

                    scaleanchor="x",

                    scaleratio=1,

                    fixedrange=True,

                    constrain="domain",

                    tickfont=dict(size=12),
                ),

                plot_bgcolor="#121826",

                paper_bgcolor="#0f1218",

                font=dict(color="#d6deeb", size=13),

                showlegend=True,

                legend=dict(font=dict(size=14), itemsizing="constant", itemwidth=30, x=1.02, y=1, xanchor="left"),

                updatemenus=[

                    dict(

                        type="buttons",

                        showactive=False,

                        x=0.02,

                        y=1.08,

                        xanchor="left",

                        yanchor="bottom",

                        bgcolor="#2a3344",

                        bordercolor="#8b95a7",

                        borderwidth=1,

                        font=dict(size=14),

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

                        x=0.02,

                        y=0.02,

                        len=0.96,

                        xanchor="left",

                        font=dict(size=11),

                        currentvalue=dict(visible=True, prefix="Frame: ", font=dict(size=12)),

                        transition=dict(duration=0),
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

    def statistics(self) -> None:

        if self.solution is None:

            raise ValueError("No solution available. Run PSO first.")


        print("#=================================================#")

        print("#                 PSO Statistics                  #")

        print("#=================================================#")

        print(f"Number of Particules: {self.config.number_of_particules}")

        print(f"Number of Iterations: {self.config.number_of_iterations}")

        print(f"Number of Waypoints: {self.config.number_of_waypoints}")

        print(f"Inertia Weight: {self.config.inertia_weight}")

        print(f"Best Position Acceleration: {self.config.best_position_acceleration}")

        print(f"Global Best Position Acceleration: {self.config.global_best_position_acceleration}")

        print(f"Length Weight: {self.config.length_weight}")

        print(f"Smoothness Weight: {self.config.smoothness_weight}")

        print(f"Collision Weight: {self.config.collision_weight}")

        print(f"Corner Weight: {self.config.corner_weight}")

        print("#=================================================#")

        print("#                  Best Path Statistics           #")

        print("#=================================================#")


        best_fit = self.path_fitness(self.solution)

        best_length = self.solution.total_length()

        best_smoothness = self.solution.smoothness()

        best_collisions, best_corners = self.solution.collisions_and_corners(

            self.environment, self.config.corner_radius
        )


        print(f"Best Path Fitness: {best_fit}")

        print(f"Best Path Length: {best_length}")

        print(f"Best Path Smoothness: {best_smoothness}")

        print(f"Best Path Collisions: {best_collisions}")

        print(f"Best Path Corners: {best_corners}")

        print("#=================================================#")


    def plot_solution(self) -> None:

        if self.solution is None:

            raise ValueError("No solution available. Run PSO first.")


        import matplotlib.pyplot as plt


        if self._fig is None or self._ax is None:

            self._fig, self._ax = plt.subplots(figsize=(8, 6), num="PSO - Path planning")

        self.environment.render(self.solution, ax=self._ax, title="Best solution")
        plt.ioff()

        plt.show()

