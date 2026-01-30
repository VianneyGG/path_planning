#TODO: Add docstring
#add steps to PSO plot_solution method


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.environment import PathPlanning, Environment
from src.PSO.Swarm import Swarm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#==============================================================================#
#                           Hyperparameters for PSO                            #
#==============================================================================#

number_of_particules = 500
number_of_iterations = 250
number_of_waypoints = 4
waypoints_reset_interval = 50
initial_temperature = 1.0
temperature_decay = 0.95

inertia_weight = 0.5    
best_position_acceleration = 1.5
global_best_position_acceleration = 1.5
length_weight = 1.0
smoothness_weight = 1.0
collision_weight = 2000.0


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
            'collision_weight': collision_weight
        }
        self.solution = None
        self._fig = None
        self._ax = None
        
    def plan_path(self, plot_steps : bool = False, reset_waypoints : bool = False, simulated_annealing : bool = False)-> np.ndarray:
        plot_interval = 20
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
            swarm.update_global_best_position(temperature, simulated_annealing)
            swarm.forward(self.environment, self.hyperparameters, temperature, simulated_annealing)
            
            if reset_waypoints and iteration % waypoints_reset_interval == 0 and iteration >= waypoints_reset_interval:
                swarm.reset_waypoints(self.environment, number_of_waypoints, self.hyperparameters)
                
        best_path_coords = swarm.get_global_best_position()
        self.solution = swarm.get_best_path()
        return best_path_coords
    
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
        print('#=================================================#')
        print(f'Best Path Length: {self.solution.total_length()}')
        print(f'Best Path Smoothness: {self.solution.smoothness()}')
        print(f'Best Path Collisions: {self.solution.nb_collisions(self.environment)}')
        print('#=================================================#')
    
    def plot_solution(self)-> None:
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(8, 6), num='PSO - Path planning')
        self.environment.render(self.solution, ax=self._ax, clear=True, show=False, title='Best solution')
        plt.ioff()
        plt.show()
         
if __name__ == "__main__":
    env = Environment()
    env.from_file("scenarios/scenario3.txt")
    pso = PSO(env)
    best_path = pso.plan_path(plot_steps=True, reset_waypoints=True, simulated_annealing=True)
    pso.plot_solution()
    pso.statistics()