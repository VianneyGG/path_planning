"""Rapidly-Exploring Random Tree (RRT) with rewiring for path planning."""

import math
import numpy as np
from typing import List, Optional, Set, Tuple, Union

from src.environment import Environment


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _to_xy(p: Union[Tuple[float, float], np.ndarray]) -> Tuple[float, float]:
    if isinstance(p, (tuple, list)) and len(p) >= 2:
        return (float(p[0]), float(p[1]))
    arr = np.asarray(p, dtype=float)
    return (float(arr[0]), float(arr[1]))


class TreeNode:
    """One node in the RRT tree: position (x,y), parent link, and cost from start."""
    def __init__(self, position: Tuple[float, float], parent: Optional["TreeNode"] = None, cost: float = 0.0):
        self.position = position
        self.parent = parent
        self.cost = cost

    def path_to_root(self) -> List[Tuple[float, float]]:
        """Path from this node back to the root (start). Returns list of (x,y) in order start -> ... -> self."""
        out: List[Tuple[float, float]] = []
        node: Optional[TreeNode] = self
        while node is not None:
            out.append(node.position)
            node = node.parent
        return out[::-1]


class RRT:
    """RRT path planner with rewiring (RRT* style)."""

    def __init__(
        self,
        start: Union[Tuple[float, float], np.ndarray],
        target: Union[Tuple[float, float], np.ndarray],
        env: Environment,
        delta_s: float,
        delta_r: float,
        n_iter: int = 200,
        p: float = 0.0,
        smooth: bool = True,
    ):
        self.start = _to_xy(start)
        self.target = _to_xy(target)
        self.env = env
        self.delta_s = delta_s
        self.delta_r = delta_r
        self.n_iter = n_iter
        self.p = p
        self.smooth = smooth
        self.root: Optional[TreeNode] = None
        self.points: Set[TreeNode] = set()

    def _sample(self) -> Tuple[float, float]:
        x = float(np.random.uniform(0.0, self.env.xmax))
        y = float(np.random.uniform(0.0, self.env.ymax))
        return (x, y)

    def _biased_sample(self) -> Tuple[float, float]:
        if np.random.rand() < self.p:
            obstacles = self.env.obstacles
            if not obstacles:
                x = float(np.random.uniform(0.0, self.env.xmax))
                y = float(np.random.uniform(0.0, self.env.ymax))
                return (x, y)
            obs = np.random.choice(obstacles)
            edge = np.random.randint(0, 4)
            if edge == 0:  # bottom
                t = np.random.rand()
                x = obs.x + t * obs.lx
                y = obs.y
            elif edge == 1:  # right
                t = np.random.rand()
                x = obs.x + obs.lx
                y = obs.y + t * obs.ly
            elif edge == 2:  # top
                t = np.random.rand()
                x = obs.x + (1 - t) * obs.lx
                y = obs.y + obs.ly
            else:  # left
                t = np.random.rand()
                x = obs.x
                y = obs.y + (1 - t) * obs.ly
            noise = np.random.normal(loc=0.0, scale=1.5, size=2) # add noise to get out of edge
            x = min(max(0.0, x + noise[0]), self.env.xmax) # ensure within boundary with min/max
            y = min(max(0.0, y + noise[1]), self.env.ymax) # ensure within boundary with min/max
            return (x, y)
        else:
            x = float(np.random.uniform(0.0, self.env.xmax))
            y = float(np.random.uniform(0.0, self.env.ymax))
            return (x, y)

    def _steer(self, from_xy: Tuple[float, float], to_xy: Tuple[float, float], step: float) -> Tuple[float, float]:
        dx = to_xy[0] - from_xy[0]
        dy = to_xy[1] - from_xy[1]
        dist = _distance(from_xy, to_xy)
        if dist <= 1e-12:
            return from_xy
        step = min(step, dist)
        x = from_xy[0] + (dx / dist) * step
        y = from_xy[1] + (dy / dist) * step
        x = max(0.0, min(self.env.xmax, x))
        y = max(0.0, min(self.env.ymax, y))
        return (x, y)

    def _is_collision_free(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        return self.env.check_line_collision(np.array(p1), np.array(p2)) == 0

    def _nearest(self, xy: Tuple[float, float]) -> Optional[TreeNode]:
        if not self.points:
            return None
        best: Optional[TreeNode] = None
        best_d = None
        for n in self.points:
            d = _distance(n.position, xy)
            if best_d is None or d < best_d:
                best_d = d
                best = n
        return best

    def _neighbors(self, node: TreeNode, radius: float) -> List[TreeNode]:
        return [n for n in self.points if n is not node and _distance(n.position, node.position) <= radius]

    def _rewire(self, v_new: TreeNode) -> None:
        for v in self._neighbors(v_new, self.delta_r):
            new_cost = v_new.cost + _distance(v_new.position, v.position)
            if new_cost < v.cost and self._is_collision_free(v_new.position, v.position):
                v.parent = v_new
                v.cost = new_cost

    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Shortcut path using triangle inequality: drop intermediate points when direct segment is collision-free."""
        if len(path) <= 2:
            return list(path)
        smoothed: List[Tuple[float, float]] = [path[0]]
        current_index = 0
        while current_index < len(path) - 1:
            found = False
            for i in range(len(path) - 1, current_index, -1):
                if self._is_collision_free(path[current_index], path[i]):
                    smoothed.append(path[i])
                    current_index = i
                    found = True
                    break
            if not found:
                current_index += 1
                smoothed.append(path[current_index])
        return smoothed

    def run_algorithm(self) -> List[Tuple[float, float]]:
        """Run RRT with rewiring. Returns path as list of (x,y) from start to node closest to goal."""
        self.root = TreeNode(self.start, parent=None, cost=0.0)
        self.points = {self.root}

        for _ in range(self.n_iter):
            v_rand = self._biased_sample()
            v_nearest = self._nearest(v_rand)
            if v_nearest is None:
                continue
            v_new_xy = self._steer(v_nearest.position, v_rand, self.delta_s)
            if not self._is_collision_free(v_nearest.position, v_new_xy):
                continue
            v_new = TreeNode(v_new_xy, parent=v_nearest, cost=v_nearest.cost + _distance(v_nearest.position, v_new_xy))
            self.points.add(v_new)
            self._rewire(v_new)

        final = self._nearest(self.target)
        if final is None:
            return [self.start]
        path = final.path_to_root()
        if self.smooth:
            path = self._smooth_path(path)
        return path
