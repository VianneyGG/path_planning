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
        for particule in self.particules:
            new_path = Path.initialize_path(env, number_of_waypoints)
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
    
    def update_global_best_position(self, temperature : float, simulated_annealing: bool)-> None:
        temp = max(float(temperature), 1e-9)
        for particule in self.particules:
            if particule.fitness < self.global_best_position_fitness:
                self.global_best_position_fitness = particule.fitness
                self.global_best_position = particule.get_position().copy()
                self.best_path = particule.path.copy()
            elif simulated_annealing:
                prob = np.exp((self.global_best_position_fitness - particule.fitness) / temp)
                if rd.random() < prob:
                    self.global_best_position = particule.get_position().copy()
                    self.best_path = particule.path.copy()
                    self.global_best_position_fitness = particule.fitness

        
    def get_global_best_position(self)-> np.ndarray:
        return self.global_best_position  
    
    def get_best_path(self)-> Path:
        return self.best_path

    def forward(self, env : Environment, hyperparameters: dict, iteration: int, temperature: float, simulated_annealing: bool, dimensional_learning: bool)-> None:
        workers = int(hyperparameters.get("parallel_fitness_workers", 1))
        reuse_pool = bool(hyperparameters.get("reuse_fitness_thread_pool", True))

        if workers > 1 and len(self.particules) > 1:
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

    
