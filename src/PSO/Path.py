import numpy as np

from src.environment import Environment
from .Waypoint import Waypoint


class Path:
    
    def __init__(self, waypoints: list[Waypoint])-> None:
        """Initialize a Path object.
        Args:
            waypoints (list[Waypoint]): List of Waypoint objects representing the path.
            fitness (float, optional): Fitness score of the path. Defaults to 0.0.
        """
        self.waypoints = waypoints
        self._rebuild_cache()
        self.best_position = self._coords.copy()

    def _rebuild_cache(self) -> None:
        self._coords = np.array([[wp.x, wp.y] for wp in self.waypoints], dtype=float)
        self._fixed_mask = np.array([wp.is_fixed for wp in self.waypoints], dtype=bool)
        
    @staticmethod
    def initialize_path(env: Environment, number_of_waypoints: int)-> 'Path':
        x, y = env.u1s
        waypoints = [Waypoint(x, y, is_fixed=True)]  # Start waypoint
        for _ in range(number_of_waypoints):
            waypoint = Waypoint.random_waypoint(x_range=(0,env.xmax), y_range=(0,env.ymax), is_fixed=False)
            waypoints.append(waypoint)
        x, y = env.u1d
        waypoints.append(Waypoint(x, y, is_fixed=True))  # Goal waypoint
        return Path(waypoints)
    
    def get_waypoints(self)-> list[Waypoint]:
        return self.waypoints

    def add_waypoint(self, waypoint: Waypoint)-> None:
        self.waypoints.append(waypoint)
        self._rebuild_cache()
        
    def get_tuple_coords(self)-> list[tuple[float,float]]:
        return [tuple(p) for p in self._coords.tolist()]
    
    def get_array_coords(self, copy: bool = True)-> np.ndarray:
        if copy:
            return self._coords.copy()
        return self._coords
    
    def total_length(self)-> float:
        if self._coords.shape[0] < 2:
            return 0.0
        segments = self._coords[1:] - self._coords[:-1]
        return float(np.linalg.norm(segments, axis=1).sum())
    
    def copy(self)-> 'Path':
        copied_waypoints = [wp.copy() for wp in self.waypoints]
        return Path(copied_waypoints)
    
    def collisions_and_corners(self, environment: Environment, radius: float, check_corners: bool = True)-> int:
        collisions, corners = 0, 0
        for i in range(1, len(self._coords)):
            p1 = self._coords[i - 1]
            p2 = self._coords[i]
            collisions += environment.check_line_collision(p1, p2)
            if check_corners and environment.near_obstacle_corner(p1, radius):
                corners += 1
        return collisions, corners

    def prune_waypoints(self, indices: list[int]) -> None:
        if not indices:
            return
        to_drop = set(indices)
        self.waypoints = [wp for idx, wp in enumerate(self.waypoints) if idx not in to_drop]
        self._rebuild_cache()
        self.best_position = self._coords.copy()

    def smoothness(self, drop_near_straight: bool = False, tolerance: float = 1e-2) -> float:
        if self._coords.shape[0] < 3:
            return 0.0

        p1 = self._coords[:-2]
        p2 = self._coords[1:-1]
        p3 = self._coords[2:]
        v1 = p2 - p1
        v2 = p3 - p2
        denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-10
        dots = np.einsum("ij,ij->i", v1, v2)
        angles = np.arccos(np.clip(dots / denom, -1.0, 1.0))

        smoothness = 0.0
        drop_candidates: list[int] = []
        for idx, angle in enumerate(angles, start=1):
            if self._fixed_mask[idx]:
                continue
            smoothness += float(angle)
            if drop_near_straight and angle < tolerance:
                drop_candidates.append(idx)

        if drop_candidates:
            # Avoid removing all consecutive non-fixed waypoints between fixed points.
            # For any consecutive run of droppable indices, keep the first one and drop the rest.
            drop_candidates_sorted = sorted(drop_candidates)
            final_drop: list[int] = []
            run_start = None
            run_prev = None
            for idx in drop_candidates_sorted:
                if run_start is None:
                    run_start = idx
                    run_prev = idx
                    continue
                if idx == run_prev + 1:
                    # continue run
                    final_drop.append(idx)  # drop this one, keep the run_start
                    run_prev = idx
                else:
                    # new run
                    run_start = idx
                    run_prev = idx
            self.prune_waypoints(final_drop)
        return smoothness
    
    def update_positions(self, new_positions: np.ndarray, xmax, ymax)-> np.ndarray:
        finite = np.isfinite(new_positions)
        safe_position = np.where(finite, new_positions, self._coords)

        clamped = safe_position.copy()
        clamped[:, 0] = np.clip(clamped[:, 0], 0.0, float(xmax))
        clamped[:, 1] = np.clip(clamped[:, 1], 0.0, float(ymax))

        hit_border_coord = (clamped != safe_position) | (~finite)
        hit_border = hit_border_coord.any(axis=1)

        movable = ~self._fixed_mask
        self._coords[movable] = clamped[movable]

        for i, wp in enumerate(self.waypoints):
            if wp.is_fixed:
                continue
            wp.x = float(self._coords[i, 0])
            wp.y = float(self._coords[i, 1])

        return hit_border
    
    def get_fixed_mask(self)-> list[bool]:
        return self._fixed_mask.tolist()

    def get_fixed_mask_array(self)-> np.ndarray:
        return self._fixed_mask
    
            