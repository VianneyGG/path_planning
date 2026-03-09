from src.PSO.Particule import Particule
from src.PSO.Path import Path
from src.environment import Environment
from typing import List
import numpy as np
import numpy.random as rd
from concurrent.futures import ThreadPoolExecutor

#==============================================================================#
#                           Swarm Class                                        #
#==============================================================================#

class Swarm:
    def __init__(self, particules: List[Particule], best_path: Path)-> None:
        self.particules = particules
        self.best_path = best_path.copy()
        self.global_best_position = self.best_path.get_array_coords()
        self.global_best_position_fitness = min(p.fitness for p in particules)
        self._fitness_executor: ThreadPoolExecutor | None = None
        self._fitness_executor_workers: int = 0
        # All-time best: tracked independently so SA exploration never overwrites the
        # true optimum found so far.
        self._all_time_best_fitness: float = self.global_best_position_fitness
        self._all_time_best_position: np.ndarray = self.global_best_position.copy()
        self._all_time_best_path: Path = self.best_path.copy()

    def _shutdown_executor(self) -> None:
        if self._fitness_executor is not None:
            self._fitness_executor.shutdown(wait=True)
            self._fitness_executor = None
            self._fitness_executor_workers = 0

    def _get_executor(self, workers: int) -> ThreadPoolExecutor:
        if self._fitness_executor is None or self._fitness_executor_workers != workers:
            self._shutdown_executor()
            self._fitness_executor = ThreadPoolExecutor(max_workers=workers)
            self._fitness_executor_workers = workers
        return self._fitness_executor

    def __del__(self) -> None:
        try:
            self._shutdown_executor()
        except Exception:
            pass
        
    def add_particule(self, particule: Particule)-> None:
        self.particules.append(particule)
    
    @staticmethod
    def initialize_swarm(num_particules: int, env : Environment, hyperparameters: dict, number_of_waypoints: int)-> 'Swarm':
        particules = []
        for _ in range(num_particules):
            particules.append(Particule.initialize_particule(env, hyperparameters, number_of_waypoints))
        best_particule = min(particules, key=lambda p: p.fitness)
        # print(f"Initialized swarm with {num_particules} particules.") # printing makes program slower, impacting benchmark purity.
        return Swarm(particules, best_particule.path)
    
    def reset_waypoints(self, env: Environment, number_of_waypoints: int, hyperparameters: dict)-> None:
        corner_delta = hyperparameters.get('corner_delta', 10.0)
        corner_init_ratio = hyperparameters.get('corner_init_ratio', 0.5)
        for particule in self.particules:
            new_path = Path.initialize_path(
                env, number_of_waypoints,
                corner_delta=corner_delta,
                corner_init_ratio=corner_init_ratio,
            )
            particule.path = new_path
            particule.position = new_path.get_array_coords()
            particule.best_position = particule.position.copy()
            particule.velocity = np.zeros_like(particule.position)
            particule.evaluate_fitness(env, self.global_best_position, hyperparameters)
        # After resetting all particules, reinitialize swarm-level bests to the best of the new particules
        if self.particules:
            best_particule = min(self.particules, key=lambda p: p.fitness)
            self.best_path = best_particule.path.copy()
            self.global_best_position = self.best_path.get_array_coords()
            self.global_best_position_fitness = best_particule.fitness
            # Reset all-time bests to the new post-reset optimum
            self._all_time_best_fitness = self.global_best_position_fitness
            self._all_time_best_position = self.global_best_position.copy()
            self._all_time_best_path = self.best_path.copy()

    def update_global_best_position(self, temperature: float, simulated_annealing: bool) -> None:
        temp = max(float(temperature), 1e-9)
        for particule in self.particules:
            if particule.fitness < self.global_best_position_fitness:
                self.global_best_position_fitness = particule.fitness
                self.global_best_position = particule.get_position().copy()
                self.best_path = particule.path.copy()
                # All-time best: updated only on genuine improvements, never degraded
                if particule.fitness < self._all_time_best_fitness:
                    self._all_time_best_fitness = particule.fitness
                    self._all_time_best_position = self.global_best_position.copy()
                    self._all_time_best_path = self.best_path.copy()
            elif simulated_annealing:
                prob = np.exp((self.global_best_position_fitness - particule.fitness) / temp)
                if rd.random() < prob:
                    # SA exploration: guide particles toward diverse positions without
                    # overwriting the all-time best record
                    self.global_best_position = particule.get_position().copy()
                    self.best_path = particule.path.copy()
                    self.global_best_position_fitness = particule.fitness

        
    def get_global_best_position(self)-> np.ndarray:
        return self.global_best_position  
    
    def get_best_path(self) -> Path:
        """Return the all-time best path (never overwritten by SA exploration)."""
        return self._all_time_best_path

    def _evaluate_fitness_vectorized(
        self,
        env: Environment,
        hyperparameters: dict,
        iteration: int,
    ) -> None:
        """Evaluate fitness for all particles in a single vectorised numpy pass.

        Constraints / fallback:
        - All particles must have the same number of waypoints.
        - ``dimensional_learning`` must be disabled (each particle may change
          shape independently in that mode).
        - ``prune_straight_angles`` must be disabled (same reason).

        Falls back to the sequential loop automatically when those conditions
        are not met.
        """
        # Check homogeneity
        shapes = set(len(p.path._coords) for p in self.particules)
        if len(shapes) != 1:
            for p in self.particules:
                p.evaluate_fitness(env, self.global_best_position, hyperparameters, False, iteration)
            return

        N = next(iter(shapes))
        P = len(self.particules)

        # Stack coords: (P, N, 2)
        coords = np.stack([p.path._coords for p in self.particules])  # (P, N, 2)

        # --- Batch lengths (P,) ---
        segs = coords[:, 1:, :] - coords[:, :-1, :]          # (P, N-1, 2)
        lengths = np.linalg.norm(segs, axis=2).sum(axis=1)   # (P,)

        # --- Batch smoothness (P,) ---
        if N >= 3:
            p1 = coords[:, :-2, :]   # (P, N-2, 2)
            p2 = coords[:, 1:-1, :]
            p3 = coords[:, 2:,   :]
            v1 = p2 - p1
            v2 = p3 - p2
            norms = np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2) + 1e-10
            dots  = np.einsum('pij,pij->pi', v1, v2)
            angles = np.arccos(np.clip(dots / norms, -1.0, 1.0))  # (P, N-2)
            # Zero out fixed intermediate waypoints (shared across particles)
            fixed_inter = self.particules[0].path._fixed_mask[1:-1]  # (N-2,)
            smoothnesses = (angles * (~fixed_inter)[np.newaxis, :]).sum(axis=1)
        else:
            smoothnesses = np.zeros(P)

        # --- Batch collisions (P,) ---
        collisions = env.check_paths_collisions_batch(coords)  # (P,)

        # --- Batch corner checks (P,) ---
        corner_stride = max(1, int(hyperparameters.get('corner_check_stride', 1)))
        check_corners = (iteration % corner_stride == 0)
        corner_radius = float(hyperparameters['corner_radius'])

        if check_corners and env._all_corners.size > 0:
            # Intermediate points for all particles: (P*(N-1), 2)
            pts_all = coords[:, :-1, :].reshape(-1, 2)
            near_all = env.check_path_corners(pts_all, corner_radius)   # (P*(N-1),)
            corners = near_all.reshape(P, N - 1).sum(axis=1).astype(int)
        else:
            corners = np.zeros(P, dtype=int)

        # --- Fitness (P,) ---
        lw = float(hyperparameters['length_weight'])
        sw = float(hyperparameters['smoothness_weight'])
        cw = float(hyperparameters['collision_weight'])
        ow = float(hyperparameters['corner_weight'])
        fitnesses = (
            lw * lengths
            + sw * smoothnesses
            + cw * collisions.astype(float)
            + ow * corners.astype(float)
        )

        # --- Update each particle ---
        for i, p in enumerate(self.particules):
            fit = float(fitnesses[i])
            p.fitness = fit
            p.position = p.path.get_array_coords(copy=False)
            if fit < p.best_fitness:
                p.best_position = p.position.copy()
                p.best_fitness = fit
                p.best_position_unchanged_count = 0
            else:
                p.best_position_unchanged_count += 1

    def forward(self, env : Environment, hyperparameters: dict, iteration: int, temperature: float, simulated_annealing: bool, dimensional_learning: bool)-> None:
        workers = int(hyperparameters.get("parallel_fitness_workers", 1))
        reuse_pool = bool(hyperparameters.get("reuse_fitness_thread_pool", True))
        use_vectorized = (
            bool(hyperparameters.get("vectorized_fitness", False))
            and not dimensional_learning
            and not hyperparameters.get("prune_straight_angles", False)
            and len(self.particules) > 1
        )

        if use_vectorized:
            self._evaluate_fitness_vectorized(env, hyperparameters, iteration)
        elif workers > 1 and len(self.particules) > 1:
            # Vectorized numpy collision & length computations release the Python GIL,
            # so ThreadPoolExecutor parallelism is effective here (no GIL bottleneck).
            def _evaluate(particule: Particule) -> None:
                particule.evaluate_fitness(
                    env,
                    self.global_best_position,
                    hyperparameters,
                    dimensional_learning,
                    iteration,
                )

            if reuse_pool:
                executor = self._get_executor(workers)
                list(executor.map(_evaluate, self.particules))
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    list(executor.map(_evaluate, self.particules))
        else:
            for particule in self.particules:
                particule.evaluate_fitness(
                    env,
                    self.global_best_position,
                    hyperparameters,
                    dimensional_learning,
                    iteration,
                ) # Update fitness and best position
        
        self.update_global_best_position(temperature, simulated_annealing) # Update global best position
        
        for particule in self.particules:
            fixed_mask = particule.path.get_fixed_mask_array()
            particule.update_velocity(fixed_mask, self.global_best_position, hyperparameters) # Velocity update
            particule.update_position(env.xmax, env.ymax) # Position update

    
