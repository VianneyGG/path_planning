"""Waypoint model module."""

import numpy as np

from src.environment import Obstacle


class Waypoint:
    """2D waypoint with optional fixed-state constraint."""

    def __init__(self, x: float, y: float, is_fixed: bool) -> None:
        """Create a waypoint."""
        self.x = x
        self.y = y
        self.is_fixed = is_fixed # useful for start and goal waypoints
        
    @staticmethod
    def random_waypoint(x_range: tuple[float,float]=(0,100), y_range: tuple[float,float]=(0,100), is_fixed: bool=False) -> 'Waypoint':
        """Sample a random waypoint in provided ranges."""
        while True:
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            waypoint = Waypoint(x, y, is_fixed)
            return waypoint
    
    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def __repr__(self) -> str:
        return f"Waypoint(x={self.x}, y={self.y}, is_fixed={self.is_fixed})"
    
    def distance_to(self, other: "Waypoint") -> float:
        ox, oy = other.to_array()
        return np.linalg.norm(np.array([self.x - ox, self.y - oy]))
        
    def copy(self) -> 'Waypoint':
        return Waypoint(self.x, self.y, self.is_fixed)
    
    def move(self, dx: float, dy: float, xmax: float, ymax: float) -> None:
        if not self.is_fixed:
            self.x = self.x + dx
            self.y = self.y + dy
            
    @staticmethod
    def from_tuple(position: tuple[float, float], is_fixed: bool) -> 'Waypoint':
        return Waypoint(position[0], position[1], is_fixed)
