import numpy as np

from src.environment import AbstractWaypoint 
from src.environment import Obstacle

class Waypoint(AbstractWaypoint):
    def __init__(self, x: float, y: float, isfixed: bool) -> None:
        self.x = x
        self.y = y
        self.isfixed = isfixed #useful for start and goal waypoints
        
    @staticmethod
    def random_waypoint(obstacles: list[Obstacle], x_range: tuple[float,float]=(0,100), y_range: tuple[float,float]=(0,100), isfixed: bool=False) -> 'Waypoint':
        while True:
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            waypoint = Waypoint(x, y, isfixed)
            collision = False
            for obstacle in obstacles:
                if obstacle.is_point_inside(waypoint.to_array()):
                    collision = True
                    break
            if not collision:
                return waypoint
    
    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def __repr__(self) -> str:
        return f"Waypoint(x={self.x}, y={self.y}, isfixed={self.isfixed})"
    
    def distance_to(self, other_waypoint: AbstractWaypoint) -> float:
        ox, oy = other_waypoint.to_array()
        return np.linalg.norm(np.array([self.x - ox, self.y - oy]))
        
    def copy(self) -> 'Waypoint':
        return Waypoint(self.x, self.y, self.isfixed)
    
    def move(self, dx: float, dy: float, xmax: float, ymax: float) -> None:
        if not self.isfixed:
            self.x = self.x + dx
            self.y = self.y + dy
            
    @staticmethod
    def from_tuple(position: tuple[float, float], isfixed: bool) -> 'Waypoint':
        return Waypoint(position[0], position[1], isfixed)
            