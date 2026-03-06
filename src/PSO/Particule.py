from .Path import Path
from src.environment import Environment
import numpy as np
import numpy.random as rd

class Particule:
    def __init__(self, path: Path)-> None:
        self.path = path
        
        self.position : np.ndarray = path.get_array_coords(copy=False)
        self.best_position : np.ndarray = self.position.copy()
        self.velocity = np.zeros_like(self.position)
        
        self.best_position_unchanged_count = 0
        self.fitness = np.inf
        self.best_fitness = np.inf
        
    @staticmethod
    def initialize_particule(env : Environment, hyperparameters: dict, number_of_waypoints: int)-> 'Particule':
        particule = Particule(Path.initialize_path(env, number_of_waypoints))
        particule.evaluate_fitness(env, None, hyperparameters)
        return particule
    
        
    def __repr__(self) -> str:
        return f"Particule(position={self.position}, fitness={self.fitness})"
    
    def get_position(self)-> np.ndarray:
        return self.position

    def _sync_after_prune(self)-> None:
        coords = self.path.get_array_coords(copy=False)
        if coords.shape != self.position.shape:
            self.position = coords
            self.velocity = np.zeros_like(coords)
            self.best_position = coords.copy()
    
    def update_velocity(self, fixed_mask: np.ndarray | list[bool], best_global_position: np.ndarray, hyperparameters: dict)-> None:
        fixed_array = fixed_mask if isinstance(fixed_mask, np.ndarray) else np.asarray(fixed_mask, dtype=bool)
        unfixed_mask = ~fixed_array
        if not unfixed_mask.any():
            return

        v = self.velocity[unfixed_mask]
        x = self.position[unfixed_mask]
        pbest = self.best_position[unfixed_mask]
        gbest = best_global_position[unfixed_mask]

        # Per-dimension random coefficients (canonical PSO — better exploration)
        r1 = rd.random(v.shape)
        r2 = rd.random(v.shape)

        w  = hyperparameters['inertia_weight']
        c1 = hyperparameters['best_position_acceleration']
        c2 = hyperparameters['global_best_position_acceleration']

        new_velocity = (
            w  * v
            + c1 * r1 * (pbest - x)
            + c2 * r2 * (gbest - x)
        )
        self.velocity[unfixed_mask] = new_velocity
        
    def update_position(self, xmax, ymax)-> None:
        new_position = self.position + self.velocity

        hit_border = self.path.update_positions(new_position, xmax, ymax)
        if hit_border.any():  # If any waypoint hit the border of the environment
            self.velocity[hit_border] = 0.0  # Set velocity to zero for those waypoints
        self.position = self.path.get_array_coords(copy=False)
            
    def evaluate_fitness(
        self,
        env: Environment,
        best_global_position: np.ndarray | None,
        hyperparameters: dict,
        dimensional_learning: bool = False,
        iteration: int | None = None,
    )-> None:
        if best_global_position is None:
            best_global_position = self.best_position
        corner_stride = max(1, int(hyperparameters.get('corner_check_stride', 1)))
        check_corners = iteration is None or (iteration % corner_stride == 0)

        if dimensional_learning and self.best_position_unchanged_count >= hyperparameters.get('max_number_of_iterations_without_improvement', np.inf):
            isfixed = self.path.get_fixed_mask_array()
            for dim in range(self.position.shape[0]):
                if isfixed[dim]:
                    continue
                original_path = self.path.copy()
                original_fitness = self.best_fitness
                
                self.position[dim] = best_global_position[dim].copy()            
                hit_border = self.path.update_positions(self.position, env.xmax, env.ymax)
                
                length = self.path.total_length()
                drop_straight = hyperparameters.get('prune_straight_angles', False)
                tolerance = hyperparameters.get('straight_angle_tolerance', 1e-2)
                smoothness = self.path.smoothness(drop_straight, tolerance)
                self._sync_after_prune()
                collisions, corners = self.path.collisions_and_corners(
                    env,
                    hyperparameters['corner_radius'],
                    check_corners=check_corners,
                )
                self.fitness = fitness = (
                    hyperparameters['length_weight'] * length +
                    hyperparameters['smoothness_weight'] * smoothness +
                    hyperparameters['collision_weight'] * collisions +
                    hyperparameters['corner_weight'] * corners
                )
                if self.fitness >= original_fitness:
                    self.path = original_path
                    self.position = original_path.get_array_coords(copy=False)
                    self.fitness = original_fitness
                else:
                    # Successful DL update: position improved — reset stagnation counter
                    self.best_position = self.position.copy()
                    self.best_fitness = self.fitness
                    self.best_position_unchanged_count = 0
                    if hit_border.any():
                        self.velocity[dim] = 0.0  # zero full waypoint velocity
        
        length = self.path.total_length()
        drop_straight = hyperparameters.get('prune_straight_angles', False)
        tolerance = hyperparameters.get('straight_angle_tolerance', 1e-2)
        smoothness = self.path.smoothness(drop_straight, tolerance)
        self._sync_after_prune()
        collisions, corners = self.path.collisions_and_corners(
            env,
            hyperparameters['corner_radius'],
            check_corners=check_corners,
        )
        fitness = (
            hyperparameters['length_weight'] * length +
            hyperparameters['smoothness_weight'] * smoothness +
            hyperparameters['collision_weight'] * collisions +
            hyperparameters['corner_weight'] * corners
        )
        self.position = self.path.get_array_coords(copy=False)
        
        if fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = fitness
            self.best_position_unchanged_count = 0
        else:
            self.best_position_unchanged_count += 1
        self.fitness = fitness
