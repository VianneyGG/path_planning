from src.PSO.Particule import Particule
from src.PSO.Path import Path
from src.environment import Environment, Obstacle
from typing import List
import numpy as np
import numpy.random as rd

#==============================================================================#
#                           Swarm Class                                        #
#==============================================================================#

class Swarm:
    def __init__(self, particules: List[Particule], best_path: Path)-> None:
        self.particules = particules
        self.best_path = best_path.copy()
        self.global_best_position = self.best_path.get_array_coords()
        self.global_best_position_fitness = np.inf
        
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
    
    def reset_waypoints(self, env: Environment, number_of_waypoints: int, hyperparameters: dict)-> None: 
        for particule in self.particules:
            new_path = Path.initialize_path(env, number_of_waypoints)
            particule.path = new_path
            particule.position = new_path.get_array_coords()
            particule.best_position = particule.position.copy()
            particule.velocity = np.zeros_like(particule.position)
            particule.evaluate_fitness(env, hyperparameters)
    
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

        
    def get_global_best_position(self)-> np.ndarray:
        return self.global_best_position  
    
    def get_best_path(self)-> Path:
        return self.best_path

    def forward(self, env : Environment, hyperparameters: dict, temperature: float, simulated_annealing: bool, dimensional_learning: bool)-> None:
        for particule in self.particules:

            particule.evaluate_fitness(env, hyperparameters) # Update fitness and best position
        
        self.update_global_best_position(temperature, simulated_annealing) # Update global best position
        
        if dimensional_learning:
            for particule in self.particules:
                particule.dimensional_learning_forward(env, self.global_best_position, hyperparameters) # Dimensional learning update
        
            self.update_global_best_position(temperature, simulated_annealing) # Update global best position
        
        for particule in self.particules:
            fixed_mask = particule.path.get_fixed_mask()
            particule.update_velocity(fixed_mask, self.global_best_position, hyperparameters) # Velocity update
            particule.update_position(env.xmax, env.ymax) # Position update

    
