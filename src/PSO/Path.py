import numpy as np

from src.environment import AbstractPath, Environment
from .Waypoint import Waypoint


class Path(AbstractPath):
    
    def __init__(self, waypoints: list[Waypoint])-> None:
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
    
    def copy(self)-> 'Path':
        copied_waypoints = [wp.copy() for wp in self.waypoints]
        return Path(copied_waypoints)
    
    def collisions_and_corners(self, environment: Environment, radius: float)-> int:
        collisions, corners = 0, 0
        for i in range(1, len(self.waypoints)):
            p1 = self.waypoints[i - 1].to_array()
            p2 = self.waypoints[i].to_array()
            collisions += environment.check_line_collision(p1, p2)
            if environment.near_obstacle_corner(p1, radius):
                corners += 1
        return collisions, corners

    def prune_waypoints(self, indices: list[int]) -> None:
        if not indices:
            return
        to_drop = set(indices)
        self.waypoints = [wp for idx, wp in enumerate(self.waypoints) if idx not in to_drop]
        self.best_position = self.get_array_coords()

    def smoothness(self, drop_near_straight: bool = False, tolerance: float = 1e-2) -> float:
        smoothness = 0.0
        drop_candidates: list[int] = []
        for i in range(1, len(self.waypoints) - 1):
            if self.waypoints[i].isfixed:
                continue
            p1 = np.array(self.waypoints[i - 1].to_array())
            p2 = np.array(self.waypoints[i].to_array())
            p3 = np.array(self.waypoints[i + 1].to_array())

            v1 = p2 - p1
            v2 = p3 - p2

            denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
            angle = np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            smoothness += angle

            if drop_near_straight and angle < tolerance:
                drop_candidates.append(i)

        if drop_candidates:
            self.prune_waypoints(drop_candidates)
        return smoothness
    
    def update_positions(self, new_positions: np.ndarray, xmax, ymax)-> np.ndarray:
        finite = np.isfinite(new_positions)
        safe_position = np.where(finite, new_positions, self.get_array_coords())

        clamped = safe_position.copy()
        clamped[:, 0] = np.clip(clamped[:, 0], 0.0, float(xmax))
        clamped[:, 1] = np.clip(clamped[:, 1], 0.0, float(ymax))

        hit_border_coord = (clamped != safe_position) | (~finite)
        hit_border = hit_border_coord.any(axis=1)

        for i, pos in enumerate(clamped):
            dx, dy = pos - self.waypoints[i].to_array()
            self.waypoints[i].move(dx, dy, xmax, ymax)

        return hit_border
    
    def get_fixed_mask(self)-> list[bool]:
        return [wp.isfixed for wp in self.waypoints]
    
            