from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict


@dataclass
class PSOConfig:
    # PSO parameters
    number_of_particules: int = 100
    number_of_iterations: int = 250
    number_of_waypoints: int = 2
    # PSO hyperparameters
    inertia_weight: float = 0.5
    best_position_acceleration: float = 1.5
    global_best_position_acceleration: float = 1.5
    # Reset waypoints
    reset_waypoints: bool = True
    reset_number: int = 2
    # Simulated annealing parameters
    simulated_annealing: bool = True
    initial_temperature: float = 5.0
    temperature_decay: float = 0.99
    pre_heat: bool = True
    pre_heat_learning_rate: float = 0.1
    pre_heat_max_iterations: int = 200
    pre_heat_target_acceptance_rate: float = 0.95
    controlled_cooling: bool = True
    acceptance_probability_decay: float | None = None
    # Dimensional learning
    dimensional_learning: bool = False
    parallel_fitness_workers: int = 1
    reuse_fitness_thread_pool: bool = True
    corner_check_stride: int = 1
    max_number_of_iterations_without_improvement: int = 20
    # Fitness function weights
    length_weight: float = 0.001
    smoothness_weight: float = 1.0
    collision_weight: float = 5.0
    corner_weight: float = -0.5
    corner_radius: float = 5.0
    # Extension parameters
    prune_straight_angles: bool = False
    straight_angle_tolerance: float = 1e-2

    def __post_init__(self) -> None:
        self.parallel_fitness_workers = max(1, int(self.parallel_fitness_workers))
        self.corner_check_stride = max(1, int(self.corner_check_stride))
        self.reset_number = max(1, int(self.reset_number))
        if self.acceptance_probability_decay is None:
            interval = max(1, int(self.number_of_iterations) // int(self.reset_number))
            self.acceptance_probability_decay = (0.01) ** (1 / interval)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "PSOConfig":
        if data is None:
            return cls()
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)
