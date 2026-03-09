from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Sequence, Tuple, Union


class Obstacle:
    def __init__(self, x, y, lx, ly):
        self.x = x
        self.y = y
        self.lx = lx
        self.ly = ly
        self.bounds = (self.x, self.y, self.x + self.lx, self.y + self.ly)
        self.polygon = Polygon([
            (self.x, self.y),
            (self.x + self.lx, self.y),
            (self.x + self.lx, self.y + self.ly),
            (self.x, self.y + self.ly),
        ])
        self.corners = np.array(
            [
                (self.x, self.y),
                (self.x + self.lx, self.y),
                (self.x + self.lx, self.y + self.ly),
                (self.x, self.y + self.ly),
            ],
            dtype=float,
        )

    def is_point_inside(self, point: np.ndarray) -> bool:
        px, py = point
        return (self.x < px < self.x + self.lx) and (self.y < py < self.y + self.ly)


class Environment:
    def __init__(self):
        self.xmax = None
        self.ymax = None
        self.u1s = None
        self.u1d = None
        self.u2s = None
        self.u2d = None
        self.R = None
        self.obstacles = []
        self.path = None
        self._obs_minx = np.array([], dtype=float)
        self._obs_miny = np.array([], dtype=float)
        self._obs_maxx = np.array([], dtype=float)
        self._obs_maxy = np.array([], dtype=float)
        self._all_corners = np.empty((0, 2), dtype=float)

    def _rebuild_obstacle_index(self) -> None:
        if self.obstacles:
            bounds = np.array([obs.bounds for obs in self.obstacles], dtype=float)
            self._obs_minx = bounds[:, 0]
            self._obs_miny = bounds[:, 1]
            self._obs_maxx = bounds[:, 2]
            self._obs_maxy = bounds[:, 3]
            self._all_corners = np.vstack([obs.corners for obs in self.obstacles])
        else:
            self._obs_minx = np.array([], dtype=float)
            self._obs_miny = np.array([], dtype=float)
            self._obs_maxx = np.array([], dtype=float)
            self._obs_maxy = np.array([], dtype=float)
            self._all_corners = np.empty((0, 2), dtype=float)

    def get_corner_waypoint_candidates(self, delta: float = 10.0) -> np.ndarray:
        """Return candidate waypoints offset diagonally outward from each obstacle corner.

        Each of the four corners of each obstacle is nudged ``delta`` units outward
        along both axes (away from the obstacle interior).  Candidates that fall
        outside the map bounds or strictly inside any obstacle are filtered out.

        Returns:
            np.ndarray of shape (M, 2).  May be empty when there are no obstacles
            or all candidates are out-of-bounds / occluded.
        """
        if not self.obstacles:
            return np.empty((0, 2), dtype=float)

        # Build all raw candidates: 4 per obstacle
        raw = []
        for obs in self.obstacles:
            x0, y0 = obs.x, obs.y
            x1, y1 = obs.x + obs.lx, obs.y + obs.ly
            raw.extend([
                (x0 - delta, y0 - delta),  # bottom-left outward
                (x1 + delta, y0 - delta),  # bottom-right outward
                (x1 + delta, y1 + delta),  # top-right outward
                (x0 - delta, y1 + delta),  # top-left outward
            ])
        pts = np.array(raw, dtype=float)  # shape (4*O, 2)

        # Filter: within map bounds
        in_bounds = (
            (pts[:, 0] >= 0.0) & (pts[:, 0] <= self.xmax) &
            (pts[:, 1] >= 0.0) & (pts[:, 1] <= self.ymax)
        )
        pts = pts[in_bounds]
        if len(pts) == 0 or len(self._obs_minx) == 0:
            return pts

        # Filter: not strictly inside any obstacle
        # Broadcasting: px shape (M,1) vs _obs_minx shape (O,)
        px = pts[:, 0:1]
        py = pts[:, 1:2]
        inside_any = (
            (px > self._obs_minx) & (px < self._obs_maxx) &
            (py > self._obs_miny) & (py < self._obs_maxy)
        ).any(axis=1)
        return pts[~inside_any]

    @staticmethod
    def _segment_intersects_rectangles(
        p1: np.ndarray,
        p2: np.ndarray,
        xmin: np.ndarray,
        ymin: np.ndarray,
        xmax: np.ndarray,
        ymax: np.ndarray,
    ) -> np.ndarray:
        x0 = float(p1[0])
        y0 = float(p1[1])
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])

        eps = 1e-12
        t_enter = np.zeros_like(xmin, dtype=float)
        t_exit = np.ones_like(xmin, dtype=float)

        if abs(dx) < eps:
            valid_x = (x0 >= xmin) & (x0 <= xmax)
        else:
            inv_dx = 1.0 / dx
            tx1 = (xmin - x0) * inv_dx
            tx2 = (xmax - x0) * inv_dx
            tmin_x = np.minimum(tx1, tx2)
            tmax_x = np.maximum(tx1, tx2)
            t_enter = np.maximum(t_enter, tmin_x)
            t_exit = np.minimum(t_exit, tmax_x)
            valid_x = np.ones_like(xmin, dtype=bool)

        if abs(dy) < eps:
            valid_y = (y0 >= ymin) & (y0 <= ymax)
        else:
            inv_dy = 1.0 / dy
            ty1 = (ymin - y0) * inv_dy
            ty2 = (ymax - y0) * inv_dy
            tmin_y = np.minimum(ty1, ty2)
            tmax_y = np.maximum(ty1, ty2)
            t_enter = np.maximum(t_enter, tmin_y)
            t_exit = np.minimum(t_exit, tmax_y)
            valid_y = np.ones_like(ymin, dtype=bool)

        return valid_x & valid_y & (t_enter <= t_exit) & (t_exit >= 0.0) & (t_enter <= 1.0)

    def from_file(self, filename):
        try:
            with open(filename, "r") as infile:
                lines = [l.strip() for l in infile.readlines()]
            self.xmax = float(lines[0])
            self.ymax = float(lines[1])
            self.u1s = np.array([float(lines[2]), float(lines[3])], dtype=float)
            self.u1d = np.array([float(lines[4]), float(lines[5])], dtype=float)
            self.u2s = np.array([float(lines[6]), float(lines[7])], dtype=float)
            self.u2d = np.array([float(lines[8]), float(lines[9])], dtype=float)
            self.R = int(float(lines[10]))
            self.obstacles = []
            for s in lines[11:]:
                parts = s.split()
                if len(parts) < 4:
                    continue
                x, y, lx, ly = map(float, parts[:4])
                self.obstacles.append(Obstacle(x, y, lx, ly))
            self._rebuild_obstacle_index()
            self.path = filename
        except OSError as e:
            raise FileNotFoundError(f"Could not find or read file: {filename}") from e
        
    def sample_point(self) -> Tuple[float, float]:
        """Sample a point uniformly from the grid outside of obstacles. Returns (x, y)."""
        if self.xmax is None:
            raise Exception("Environment not initialised")
        while True:
            x = float(np.random.uniform(0.0, self.xmax))
            y = float(np.random.uniform(0.0, self.ymax))
            if not any(obs.is_point_inside(np.array([x, y])) for obs in self.obstacles):
                return (x, y)
            
    def render(self, 
            path: Optional[Union[Sequence[Tuple[float, float]], object]] = None,
            *, 
            path2: Optional[Union[Sequence[Tuple[float, float]], object]] = None,
            title: Optional[str] = None,
            figsize: Tuple[int, int] = (8, 6),
            ax=None
            ):
        """Draw environment and optionally one or two paths (list of (x,y) or object with get_waypoints())."""
        colors = {
            "fig_bg": "#0f1218",
            "ax_bg": "#121826",
            "board": "#8b95a7",
            "grid": "#2a3344",
            "obstacle_fill": "#2a1b23",
            "obstacle_edge": "#6b2d45",
            "path": "#4da3ff",
            "path_mark": "#cfe8ff",
            "start": "#34d399",
            "goal": "#60a5fa",
            "text": "#d6deeb",
            "muted": "#9aa6b2",
        }

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
            ax.cla()
        fig.patch.set_facecolor(colors["fig_bg"])
        ax.set_facecolor(colors["ax_bg"])
        ax.tick_params(colors=colors["muted"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(colors["grid"])

        board = Polygon([(0, 0), (0, self.ymax), (self.xmax, self.ymax), (self.xmax, 0)])
        ax.plot(*board.exterior.xy, color=colors["board"], linewidth=2.0, alpha=0.9)
        for obs in self.obstacles:
            box = Polygon([
                (obs.x, obs.y),
                (obs.x + obs.lx, obs.y),
                (obs.x + obs.lx, obs.y + obs.ly),
                (obs.x, obs.y + obs.ly),
            ])
            ax.fill(*box.exterior.xy, color=colors["obstacle_fill"], ec=colors["obstacle_edge"], lw=1.5, alpha=0.95)
        ax.scatter(self.u1s[0], self.u1s[1], label="Start 1", s=90, color=colors["start"], edgecolors=colors["fig_bg"], linewidths=1.0, zorder=6)
        ax.scatter(self.u1d[0], self.u1d[1], label="Goal 1", s=90, color=colors["goal"], edgecolors=colors["fig_bg"], linewidths=1.0, zorder=6)
        if path2 is not None and self.u2s is not None and self.u2d is not None:
            ax.scatter(self.u2s[0], self.u2s[1], label="Start 2", s=90, color="#fbbf24", edgecolors=colors["fig_bg"], linewidths=1.0, zorder=6)
            ax.scatter(self.u2d[0], self.u2d[1], label="Goal 2", s=90, color="#f59e0b", edgecolors=colors["fig_bg"], linewidths=1.0, zorder=6)

        ax.grid(True, lw=1, alpha=0.35, ls="-", color=colors["grid"])
        ax.set_xlim(0, self.xmax)
        ax.set_ylim(0, self.ymax)
        ax.set_aspect("equal", adjustable="box")

        def to_pts(p, default_start, default_goal):
            if hasattr(p, "get_waypoints"):
                pts = np.array([np.array(wp.to_tuple(), dtype=float) for wp in p.get_waypoints()])
            else:
                pts = np.array(p, dtype=float)
            if len(pts) == 0:
                pts = np.array([default_start], dtype=float)
            if not np.allclose(pts[0], np.array(default_start, dtype=float), atol=1e-9):
                pts = np.vstack([np.array(default_start, dtype=float), pts])
            if default_goal is not None and (len(pts) < 2 or not np.allclose(pts[-1], np.array(default_goal, dtype=float), atol=1e-6)):
                pts = np.vstack([pts, np.array(default_goal, dtype=float)])
            return pts

        if path is not None:
            pts = to_pts(path, self.u1s, self.u1d)
            ax.plot(pts[:, 0], pts[:, 1], color=colors["path"], lw=2.4, alpha=0.95, solid_capstyle="round", zorder=5, label="Robot 1")
            if pts.shape[0] > 2:
                ax.scatter(pts[1:-1, 0], pts[1:-1, 1], s=35, c=colors["path_mark"], edgecolors=colors["fig_bg"], linewidths=0.8, alpha=0.95, zorder=6)
        if path2 is not None and self.u2s is not None and self.u2d is not None:
            pts2 = to_pts(path2, self.u2s, self.u2d)
            ax.plot(pts2[:, 0], pts2[:, 1], color="#f59e0b", lw=2.4, alpha=0.95, solid_capstyle="round", zorder=5, label="Robot 2")
            if pts2.shape[0] > 2:
                ax.scatter(pts2[1:-1, 0], pts2[1:-1, 1], s=35, c="#fcd34d", edgecolors=colors["fig_bg"], linewidths=0.8, alpha=0.95, zorder=6)

        leg = ax.legend(loc="upper left", frameon=True)
        leg.get_frame().set_facecolor(colors["fig_bg"])
        leg.get_frame().set_edgecolor(colors["grid"])
        leg.get_frame().set_alpha(0.85)
        for txt in leg.get_texts():
            txt.set_color(colors["text"])

        ax.set_title(title or f"Board from file: {self.path}", loc="center", color=colors["text"], fontsize=14, fontweight="bold")
        fig.tight_layout()
        if ax is None:
            plt.show(block=True)
    
    def __getstate__(self) -> dict:
        """Custom pickle support: exclude heavy Shapely Polygon objects.

        Polygon is only needed for ``render()``; all fitness-relevant state is
        in the numpy arrays and Obstacle scalar fields.  Excluding it cuts
        pickle/unpickle time significantly when environments are sent to loky
        workers via joblib.
        """
        state = self.__dict__.copy()
        # Replace Obstacle objects with lightweight plain dicts
        state['obstacles'] = [
            {'x': o.x, 'y': o.y, 'lx': o.lx, 'ly': o.ly}
            for o in self.obstacles
        ]
        return state

    def __setstate__(self, state: dict) -> None:
        obs_dicts = state.pop('obstacles')
        self.__dict__.update(state)
        # Reconstruct full Obstacle objects (Polygon rebuilt in __init__)
        self.obstacles = [Obstacle(d['x'], d['y'], d['lx'], d['ly']) for d in obs_dicts]

    def get_obstacles(self) -> List[Obstacle]:
        return self.obstacles
    
    def check_line_collision(self, p1: np.ndarray, p2: np.ndarray) -> int:
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)

        if self._obs_minx.size == 0:
            return 0

        minx = min(float(p1[0]), float(p2[0]))
        maxx = max(float(p1[0]), float(p2[0]))
        miny = min(float(p1[1]), float(p2[1]))
        maxy = max(float(p1[1]), float(p2[1]))

        overlap = (
            (self._obs_maxx >= minx)
            & (self._obs_minx <= maxx)
            & (self._obs_maxy >= miny)
            & (self._obs_miny <= maxy)
        )
        if not np.any(overlap):
            return 0

        xmin = self._obs_minx[overlap]
        ymin = self._obs_miny[overlap]
        xmax = self._obs_maxx[overlap]
        ymax = self._obs_maxy[overlap]

        hits = self._segment_intersects_rectangles(p1, p2, xmin, ymin, xmax, ymax)
        return int(np.count_nonzero(hits))
    
    def near_obstacle_corner(self, point: np.ndarray, radius: float) -> bool:
        if self._all_corners.size == 0:
            return False
        px, py = point
        radius2 = float(radius) * float(radius)
        dx = self._all_corners[:, 0] - float(px)
        dy = self._all_corners[:, 1] - float(py)
        return bool(np.any(dx * dx + dy * dy <= radius2))

    def _collisions_from_segments(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Core vectorized slab-intersection test for arbitrary (S, 2) segment endpoints.

        Parameters
        ----------
        starts, ends : ndarray, shape (S, 2)

        Returns
        -------
        ndarray of int, shape (S,) -- collision count per segment.
        """
        S = len(starts)
        if S == 0 or self._obs_minx.size == 0:
            return np.zeros(S, dtype=int)

        # Per-segment bounding box
        seg_minx = np.minimum(starts[:, 0], ends[:, 0])[:, np.newaxis]  # (S, 1)
        seg_maxx = np.maximum(starts[:, 0], ends[:, 0])[:, np.newaxis]
        seg_miny = np.minimum(starts[:, 1], ends[:, 1])[:, np.newaxis]
        seg_maxy = np.maximum(starts[:, 1], ends[:, 1])[:, np.newaxis]

        # AABB overlap filter: (S, O)
        ox_min = self._obs_minx[np.newaxis, :]  # (1, O)
        ox_max = self._obs_maxx[np.newaxis, :]
        oy_min = self._obs_miny[np.newaxis, :]
        oy_max = self._obs_maxy[np.newaxis, :]

        aabb = (
            (ox_max >= seg_minx) & (ox_min <= seg_maxx) &
            (oy_max >= seg_miny) & (oy_min <= seg_maxy)
        )  # (S, O)
        if not np.any(aabb):
            return np.zeros(S, dtype=int)

        # Slab intersection test, fully vectorized over (S, O)
        x0 = starts[:, 0:1]            # (S, 1)
        y0 = starts[:, 1:2]
        dx = (ends - starts)[:, 0:1]   # (S, 1)
        dy = (ends - starts)[:, 1:2]

        eps = 1e-12

        # X slab
        nonhoriz = np.abs(dx) >= eps
        safe_dx  = np.where(nonhoriz, dx, 1.0)
        tx1 = (ox_min - x0) / safe_dx
        tx2 = (ox_max - x0) / safe_dx
        t_enter_x = np.where(nonhoriz, np.minimum(tx1, tx2), 0.0)
        t_exit_x  = np.where(nonhoriz, np.maximum(tx1, tx2), 1.0)
        valid_x   = np.where(nonhoriz, np.ones((S, 1), dtype=bool),
                             (x0 >= ox_min) & (x0 <= ox_max))

        # Y slab
        nonvert = np.abs(dy) >= eps
        safe_dy = np.where(nonvert, dy, 1.0)
        ty1 = (oy_min - y0) / safe_dy
        ty2 = (oy_max - y0) / safe_dy
        t_enter_y = np.where(nonvert, np.minimum(ty1, ty2), 0.0)
        t_exit_y  = np.where(nonvert, np.maximum(ty1, ty2), 1.0)
        valid_y   = np.where(nonvert, np.ones((S, 1), dtype=bool),
                             (y0 >= oy_min) & (y0 <= oy_max))

        t_ent = np.maximum(t_enter_x, t_enter_y)
        t_ext = np.minimum(t_exit_x,  t_exit_y)

        hits = (
            valid_x & valid_y
            & (t_ent <= t_ext) & (t_ext >= 0.0) & (t_ent <= 1.0)
            & aabb
        )  # (S, O)
        return hits.sum(axis=1).astype(int)  # (S,)

    def check_path_collisions(self, coords: np.ndarray) -> np.ndarray:
        """Vectorized collision count for every consecutive segment in `coords`.

        Parameters
        ----------
        coords : ndarray, shape (N, 2)
            Waypoint coordinates. Defines N-1 segments.

        Returns
        -------
        ndarray of int, shape (N-1,)
            Number of obstacle collisions per segment.
        """
        n_seg = len(coords) - 1
        if n_seg <= 0:
            return np.zeros(max(0, n_seg), dtype=int)
        return self._collisions_from_segments(coords[:-1], coords[1:])

    def check_paths_collisions_batch(self, all_coords: np.ndarray) -> np.ndarray:
        """Vectorized total collision count for P particles simultaneously.

        Parameters
        ----------
        all_coords : ndarray, shape (P, N, 2)
            Stacked waypoint arrays for P particles, each with N waypoints.

        Returns
        -------
        ndarray of int, shape (P,)
            Total collision count (summed over all N-1 segments) per particle.
        """
        P, N, _ = all_coords.shape
        S_total = P * (N - 1)
        if S_total <= 0:
            return np.zeros(P, dtype=int)
        starts_all = all_coords[:, :-1, :].reshape(S_total, 2)
        ends_all   = all_coords[:, 1:,  :].reshape(S_total, 2)
        per_seg = self._collisions_from_segments(starts_all, ends_all)  # (S_total,)
        return per_seg.reshape(P, N - 1).sum(axis=1).astype(int)        # (P,)

    def check_path_corners(self, points: np.ndarray, radius: float) -> np.ndarray:
        """Vectorized near-corner check for multiple points.

        Parameters
        ----------
        points : ndarray, shape (N, 2)
        radius : float

        Returns
        -------
        ndarray of bool, shape (N,)
        """
        if self._all_corners.size == 0 or len(points) == 0:
            return np.zeros(len(points), dtype=bool)
        radius2 = float(radius) ** 2
        # (N, C) distance squared
        diff  = points[:, np.newaxis, :] - self._all_corners[np.newaxis, :, :]  # (N, C, 2)
        dist2 = (diff * diff).sum(axis=2)                                        # (N, C)
        return np.any(dist2 <= radius2, axis=1)  # (N,)

if __name__ == "__main__":
    env = Environment()
    print("Enter scenario number: ")
    nb_scenario = int(input())
    env.from_file(f"scenarios/scenario{nb_scenario}.txt")
    path = [(100, 200), (600, 900), (1000, 1000)]
    env.render(path)
