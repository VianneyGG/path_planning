from .Path import Path
from src.environment import Environment
import numpy as np
import numpy.random as rd

class Particule:
    def __init__(self, path: Path)-> None:
        self.path = path
        
        self.position = path.get_array_coords()
        self.best_position = self.position.copy()
        self.velocity = np.zeros_like(self.position)
        
        self.fitness = np.inf
        
    @staticmethod
    def initialize_particule(env : Environment, hyperparameters: dict, number_of_waypoints: int)-> 'Particule':
        particule = Particule(Path.initialize_path(env, number_of_waypoints))
        particule.evaluate_fitness(env, hyperparameters)
        return particule
    
        
    def __repr__(self) -> str:
        return f"Particule(position={self.position}, fitness={self.fitness})"
    
    def get_position(self)-> np.ndarray:
        return self.position
    
    def update_velocity(self, fixed_mask : list[bool], best_global_position: np.ndarray, hyperparameters: dict)-> None:
        r1 = rd.random()
        r2 = rd.random()
        unfixed_mask =  np.array([np.array([not fixed, not fixed]) for fixed in fixed_mask])
        new_velocity = (
            hyperparameters['inertia_weight'] * self.velocity[unfixed_mask] +
            hyperparameters['best_position_acceleration'] * r1 * (self.best_position[unfixed_mask] - self.position[unfixed_mask]) +
            hyperparameters['global_best_position_acceleration'] * r2 * (best_global_position[unfixed_mask] - self.position[unfixed_mask])
        )
        self.velocity[unfixed_mask] = new_velocity
        
    def update_position(self, xmax, ymax)-> None:
        new_position = self.position + self.velocity

        # Guard against NaN/Inf: keep previous coordinate if non-finite
        finite = np.isfinite(new_position)
        safe_position = np.where(finite, new_position, self.position)

        # Clamp to borders
        clamped = safe_position.copy()
        clamped[:, 0] = np.clip(clamped[:, 0], 0.0, float(xmax))
        clamped[:, 1] = np.clip(clamped[:, 1], 0.0, float(ymax))

        # "Sans vitesse" when hitting borders (or when the coord was non-finite)
        hit_border = (clamped != safe_position) | (~finite)
        self.velocity[hit_border] = 0.0

        # Apply (Path/Waypoint will also clamp, but now values are already safe)
        self.path.update_positions(clamped, xmax, ymax)
        self.position = self.path.get_array_coords()
            
    def evaluate_fitness(self, env: Environment, hyperparameters: dict)-> None:
        length = self.path.total_length()
        smoothness = self.path.smoothness()
        nb_collisions = self.path.nb_collisions(env)
        fitness = (
            hyperparameters['length_weight'] * length +
            hyperparameters['smoothness_weight'] * smoothness +
            hyperparameters['collision_weight'] * nb_collisions
        )
        if fitness < self.fitness:
            self.best_position = self.position.copy()
        self.fitness = fitness
    
    def forward(self, env: Environment, global_best_position: np.ndarray, hyperparameters: dict)-> None:
        fixed_mask = self.path.get_fixed_mask()
        self.update_velocity(fixed_mask, global_best_position, hyperparameters)
        self.update_position(env.xmax, env.ymax)
        self.evaluate_fitness(env, hyperparameters) # Update fitness and best position
