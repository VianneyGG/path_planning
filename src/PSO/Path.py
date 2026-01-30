import numpy as np

from src.environment import AbstractPath, Environment
from .Waypoint import Waypoint


class Path(AbstractPath):
    
    def __init__(self, waypoints: list[Waypoint], fitness: float = 0.0)-> None:
        """Initialize a Path object.
        Args:
            waypoints (list[Waypoint]): List of Waypoint objects representing the path.
            fitness (float, optional): Fitness score of the path. Defaults to 0.0.
        """
        self.waypoints = waypoints
        self.best_position = self.get_array_coords()
        
    @staticmethod
    def initialize_path(env: Environment, number_of_waypoints: int)-> 'Path':
        x, y = env.u1s
        waypoints = [Waypoint(x, y, isfixed=True)]  # Start waypoint
        for _ in range(number_of_waypoints):
            waypoint = Waypoint.random_waypoint(env.get_obstacles(), x_range=(0,env.xmax), y_range=(0,env.ymax), isfixed=False)
            waypoints.append(waypoint)
        x, y = env.u1d
        waypoints.append(Waypoint(x, y, isfixed=True))  # Goal waypoint
        return Path(waypoints)
    
    def get_waypoints(self)-> list[Waypoint]:
        return self.waypoints

    def add_waypoint(self, waypoint: Waypoint)-> None:
        self.waypoints.append(waypoint)
        
    def get_tuple_coords(self)-> list[tuple[float,float]]:
        return [waypoint.to_tuple() for waypoint in self.waypoints]
    
    def get_array_coords(self)-> np.ndarray:
        return np.array([waypoint.to_array() for waypoint in self.waypoints])
    
    def total_length(self)-> float:
        length = 0.0
        for i in range(1, len(self.waypoints)):
            length += self.waypoints[i - 1].distance_to(self.waypoints[i])
        return length
    
    def nb_collisions(self, environment: Environment)-> int:
        collisions = 0
        for i in range(1, len(self.waypoints)):
            p1 = self.waypoints[i - 1].to_array()
            p2 = self.waypoints[i].to_array()
            collisions += environment.check_line_collision(p1, p2)
        return collisions

    def smoothness(self)-> float:
        smoothness = 0.0
        for i in range(1, len(self.waypoints) - 1):
            p1 = np.array(self.waypoints[i - 1].to_array())
            p2 = np.array(self.waypoints[i].to_array())
            p3 = np.array(self.waypoints[i + 1].to_array())
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            angle = np.arccos(
                np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10),
                    -1.0,
                    1.0
                )
            )
            smoothness += angle
        return smoothness
    
    def update_positions(self, new_positions: np.ndarray, xmax, ymax)-> None:
        for i, pos in enumerate(new_positions):
            dx, dy = new_positions[i] - self.waypoints[i].to_array()
            self.waypoints[i].move(dx, dy, xmax, ymax)
    
    def get_fixed_mask(self)-> list[bool]:
        return [wp.isfixed for wp in self.waypoints]
    
            