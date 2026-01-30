from src.PSO.Particule import Particule
from src.PSO.Path import Path
from src.environment import Environment, Obstacle
from typing import List
import numpy as np

#==============================================================================#
#                           Swarm Class                                        #
#==============================================================================#

class Swarm:
    def __init__(self, particules: List[Particule], best_path: Path)-> None:
        self.particules = particules
        self.best_path = best_path
        self.global_best_position = best_path.get_array_coords()
        
    def add_particule(self, particule: Particule)-> None:
        self.particules.append(particule)
    
    @staticmethod
    def initialize_swarm(num_particules: int, env : Environment, hyperparameters: dict, number_of_waypoints: int)-> 'Swarm':
        particules = []
        for _ in range(num_particules):
            particules.append(Particule.initialize_particule(env, hyperparameters, number_of_waypoints))
        best_path = particules[0].path
        print(f"Initialized swarm with {num_particules} particules.")
        return Swarm(particules, best_path)
    
    def update_global_best_position(self)-> None:
        best_particule = min(self.particules, key=lambda p: p.fitness)
        self.global_best_position = best_particule.get_position().copy()
        
    def get_global_best_position(self)-> np.ndarray:
        return self.global_best_position  
    
    def get_best_path(self)-> Path:
        return self.best_path

    def forward(self, env : Environment, hyperparameters: dict)-> None:
        self.update_global_best_position()
        for particule in self.particules:
            particule.forward(env, self.global_best_position, hyperparameters)
    
    