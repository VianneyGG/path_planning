from shapely.geometry import LineString, Polygon, box
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Sequence, Tuple, Union

try:
    from shapely.strtree import STRtree
except Exception:  # pragma: no cover
    STRtree = None


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
        self._obstacle_tree = None
        self._poly_to_obstacle = {}

    def _rebuild_obstacle_index(self) -> None:
        self._obstacle_tree = None
        self._poly_to_obstacle = {}
        if not self.obstacles or STRtree is None:
            return
        polygons = [obs.polygon for obs in self.obstacles]
        self._obstacle_tree = STRtree(polygons)
        self._poly_to_obstacle = {id(poly): obs for poly, obs in zip(polygons, self.obstacles)}

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
    
    def get_obstacles(self) -> List[Obstacle]:
        return self.obstacles
    
    def check_line_collision(self, p1: np.ndarray, p2: np.ndarray) -> int:
        nb_collisions = 0
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        line = LineString([p1, p2])

        minx = min(float(p1[0]), float(p2[0]))
        maxx = max(float(p1[0]), float(p2[0]))
        miny = min(float(p1[1]), float(p2[1]))
        maxy = max(float(p1[1]), float(p2[1]))

        candidates = self.obstacles
        if self._obstacle_tree is not None and len(self.obstacles) >= 20:
            query_geom = box(minx, miny, maxx, maxy)
            geoms = self._obstacle_tree.query(query_geom)
            candidates = [self._poly_to_obstacle.get(id(poly)) for poly in geoms]

        for obs in candidates:
            if obs is None:
                continue
            ox0, oy0, ox1, oy1 = obs.bounds
            if maxx < ox0 or minx > ox1 or maxy < oy0 or miny > oy1:
                continue
            if line.intersects(obs.polygon):
                nb_collisions += 1
        return nb_collisions
    
    def near_obstacle_corner(self, point: np.ndarray, radius: float) -> bool:
        px, py = point
        radius2 = float(radius) * float(radius)
        point_arr = np.array([px, py], dtype=float)
        for obs in self.obstacles:
            deltas = obs.corners - point_arr
            if np.any(np.sum(deltas * deltas, axis=1) <= radius2):
                return True
        return False

if __name__ == "__main__":
    env = Environment()
    print("Enter scenario number: ")
    nb_scenario = int(input())
    env.from_file(f"scenarios/scenario{nb_scenario}.txt")
    path = [(100, 200), (600, 900), (1000, 1000)]
    env.render(path)
