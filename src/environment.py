from abc import ABC, abstractmethod

from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Optional, Sequence, Tuple


class Obstacle:
    def __init__(self, x, y, lx, ly):
        self.x = x
        self.y = y
        self.lx = lx
        self.ly = ly

    def is_point_inside(self, point: np.ndarray) -> bool:
        px, py = point
        return (self.x < px < self.x + self.lx) and (self.y < py < self.y + self.ly)

#======================================================================#
#                    Abstract Classes                                  #
#======================================================================#

class AbstractWaypoint(ABC):
    """
    Interface abstraite pour un point de passage dans un chemin.
    À implémenter différemment pour PSO (avec vitesse) et RRT (avec parent).
    """
    
    @abstractmethod
    def to_tuple(self) -> Tuple[float, float]:
        """Retourne les coordonnées (x, y) du waypoint"""
        raise NotImplementedError
    
    @abstractmethod
    def to_array(self) -> Sequence[float]:
        """Retourne les coordonnées [x, y] du waypoint sous forme de séquence"""
        raise NotImplementedError
    
    @abstractmethod
    def distance_to(self, other: 'AbstractWaypoint') -> float:
        """Distance euclidienne à un autre waypoint"""
        raise NotImplementedError
    
    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def copy(self) -> 'AbstractWaypoint':
        """Retourne une copie du waypoint"""
        raise NotImplementedError

class AbstractPath(ABC):
    """
    Interface abstraite pour un chemin.
    Représentation différente pour PSO (liste de waypoints) et RRT (arbre).
    """
    
    @abstractmethod
    def get_waypoints(self) -> List[AbstractWaypoint]:
        """Retourne la séquence de waypoints du chemin"""
        raise NotImplementedError
    
    @abstractmethod
    def total_length(self) -> float:
        """Calcule la longueur totale du chemin"""
        raise NotImplementedError
    
    @abstractmethod
    def nb_collisions(self, environment: 'Environment') -> int:
        """Compte le nombre de collisions du chemin avec l'environnement"""
        raise NotImplementedError
    
    @abstractmethod
    def smoothness(self) -> float:
        """Calcule la mesure de douceur du chemin"""
        raise NotImplementedError
    
    @abstractmethod
    def get_tuple_coords(self) -> List[Tuple[float, float]]:
        """Convertit le chemin en listes de coordonnées x, y"""
        raise NotImplementedError
    
    @abstractmethod
    def get_array_coords(self) -> np.ndarray:
        """Convertit le chemin en listes de coordonnées x, y sous forme de tableau numpy"""
        raise NotImplementedError

class StaticWaypoint(AbstractWaypoint):
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def distance_to(self, other: AbstractWaypoint) -> float:
        ox, oy = other.to_array()
        return np.linalg.norm(np.array([self.x - ox, self.y - oy]))

    def __repr__(self) -> str:
        return f"StaticWaypoint(x={self.x}, y={self.y})"

    def copy(self) -> 'StaticWaypoint':
        return StaticWaypoint(self.x, self.y)

class StaticPath(AbstractPath):
    def __init__(self, points: Sequence[Tuple[float, float]]):
        self._waypoints = [StaticWaypoint(x, y) for x, y in points]

    def get_waypoints(self) -> List[AbstractWaypoint]:
        return list(self._waypoints)
    
    def total_length(self) -> float:
        length = 0.0
        for i in range(1, len(self._waypoints)):
            length += self._waypoints[i - 1].distance_to(self._waypoints[i])
        return length
    
    def nb_collisions(self, environment: 'Environment') -> int:
        collisions = 0
        for i in range(1, len(self._waypoints)):
            p1 = self._waypoints[i - 1].to_array()
            p2 = self._waypoints[i].to_array()
            collisions += environment.check_line_collision(p1, p2)
        return collisions
    
    def get_tuple_coords(self) -> List[Tuple[float, float]]:
        return [wp.to_tuple() for wp in self._waypoints]
    
    def get_array_coords(self) -> np.ndarray:
        return np.array([wp.to_array() for wp in self._waypoints])
    
    def smoothness(self) -> float:
        smoothness = 0.0
        for i in range(1, len(self._waypoints) - 1):
            p1 = np.array(self._waypoints[i - 1].to_array())
            p2 = np.array(self._waypoints[i].to_array())
            p3 = np.array(self._waypoints[i + 1].to_array())
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
    
class PathPlanning(ABC):
    @abstractmethod
    def plan_path(self, plot_steps : bool) -> AbstractPath:
        """Planifie un chemin dans l'environnement donné"""
        raise NotImplementedError
    
    @abstractmethod
    def statistics(self) -> None:
        """Retourne des statistiques sur le processus de planification"""
        raise NotImplementedError
    
    @abstractmethod
    def plot_solution(self) -> None:
        """Affiche le chemin planifié dans l'environnement"""
        raise NotImplementedError
    
#======================================================================#
#               Environment Class                                      #
#======================================================================#

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

    def from_file(self, filename):
        try:
            with open(filename, 'r') as infile:
                lines = [l.strip() for l in infile.readlines()]
                # lines = [l.strip() for l in lines]
                self.xmax = float(lines[0])
                self.ymax = float(lines[1])
                self.u1s = float(lines[2]), float(lines[3])
                self.u1d = float(lines[4]), float(lines[5])
                self.u2s = float(lines[6]), float(lines[7])
                self.u2d = float(lines[8]), float(lines[9])
                self.R = int(float(lines[10]))
                # print(lines[11:])
                for x in lines[11:]:
                    # print(x)
                    x = x.strip()
                    x = x.split("   ")
                    x = [float(y.strip()) for y in x]
                    # self.obstacles.append(tuple(x))
                    self.obstacles.append(Obstacle(*x))
                self.path = filename
        except:
            raise FileNotFoundError(f"Could not find the fiel at {filename}")

    def render(
        self,
        path=None,
        *,
        ax=None,
        clear: bool = True,
        show: bool = True,
        pause: Optional[float] = None,
        title: Optional[str] = None,
        label_waypoints: bool = False,
        figsize: Tuple[int, int] = (8, 6),
        block: bool = False,
    ):
        """Render the environment (and optionally a path).

        This method supports iterative plotting by reusing an existing Matplotlib Axes.

        Args:
            path: Optional path object (must expose get_waypoints()).
            ax: Optional Matplotlib Axes to draw into. If None, a new figure is created.
            clear: If True, clears the axes before drawing.
            show: If True, calls plt.show(). Set to False in iterative loops.
            pause: If set, calls plt.pause(pause) after drawing (useful for animations).
            title: Custom title. Defaults to "Board from file: ...".
            figsize: Size used when creating a new figure.
            block: Passed to plt.show(block=...).

        Returns:
            (fig, ax)
        """
        # --- Dark theme (professional look) ---
        COLORS = {
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
            # New axes => (re)draw everything
            clear = True
        else:
            fig = ax.figure

        # Apply theme once per Axes
        if not getattr(ax, "_pp_theme_applied", False):
            try:
                fig.patch.set_facecolor(COLORS["fig_bg"])
            except Exception:
                pass
            ax.set_facecolor(COLORS["ax_bg"])
            ax.tick_params(colors=COLORS["muted"], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(COLORS["grid"])
            ax._pp_theme_applied = True

        # For iterative updates: draw static environment once, then only update path.
        static_drawn = getattr(ax, "_pp_static_drawn", False)
        if clear or not static_drawn:
            ax.cla()

            # Re-apply theme after cla()
            ax.set_facecolor(COLORS["ax_bg"])
            ax.tick_params(colors=COLORS["muted"], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(COLORS["grid"])

            board = Polygon(
                [
                    (0, 0),
                    (0, self.ymax),
                    (self.xmax, self.ymax),
                    (self.xmax, 0),
                ]
            )
            ax.plot(*board.exterior.xy, color=COLORS["board"], linewidth=2.0, alpha=0.9)

            for obs in self.obstacles:
                box = Polygon([
                    (obs.x, obs.y),
                    (obs.x + obs.lx, obs.y),
                    (obs.x + obs.lx, obs.y + obs.ly),
                    (obs.x, obs.y + obs.ly),
                ])
                ax.fill(
                    *box.exterior.xy,
                    color=COLORS["obstacle_fill"],
                    ec=COLORS["obstacle_edge"],
                    lw=1.5,
                    alpha=0.95,
                )

            ax.scatter(
                self.u1s[0],
                self.u1s[1],
                label='Start',
                s=90,
                color=COLORS["start"],
                edgecolors=COLORS["fig_bg"],
                linewidths=1.0,
                zorder=6,
            )
            ax.scatter(
                self.u1d[0],
                self.u1d[1],
                label='Goal',
                s=90,
                color=COLORS["goal"],
                edgecolors=COLORS["fig_bg"],
                linewidths=1.0,
                zorder=6,
            )
            ax.grid(True, lw=1, alpha=0.35, ls='-', color=COLORS["grid"])
            ax.set_xlim(0, self.xmax)
            ax.set_ylim(0, self.ymax)
            ax.set_aspect('equal', adjustable='box')

            leg = ax.legend(loc='upper left', frameon=True)
            try:
                leg.get_frame().set_facecolor(COLORS["fig_bg"])
                leg.get_frame().set_edgecolor(COLORS["grid"])
                leg.get_frame().set_alpha(0.85)
                for txt in leg.get_texts():
                    txt.set_color(COLORS["text"])
            except Exception:
                pass

            # Tight layout only once (expensive in a loop)
            try:
                fig.tight_layout()
            except Exception:
                pass

            ax._pp_static_drawn = True
            ax._pp_path_line = None
            ax._pp_path_scatter = None

        # Efficient path update: keep one Line2D + one scatter and update their data.
        line = getattr(ax, "_pp_path_line", None)
        scatter = getattr(ax, "_pp_path_scatter", None)
        wp_texts = getattr(ax, "_pp_wp_texts", None)

        if path is None:
            if line is not None:
                line.set_data([], [])
            if scatter is not None:
                scatter.set_offsets(np.empty((0, 2)))
            if wp_texts:
                for t in wp_texts:
                    try:
                        t.remove()
                    except Exception:
                        pass
                ax._pp_wp_texts = []
        else:
            wp_coords = [np.array(wp.to_tuple(), dtype=float) for wp in path.get_waypoints()]
            if len(wp_coords) == 0:
                pts = np.array([np.array(self.u1s, dtype=float)], dtype=float)
            else:
                pts = np.vstack(wp_coords)
                # Avoid duplicating the start point if the path already includes it.
                if not np.allclose(pts[0], np.array(self.u1s, dtype=float), atol=1e-9):
                    pts = np.vstack([np.array(self.u1s, dtype=float), pts])

            if line is None:
                (line,) = ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color=COLORS["path"],
                    lw=2.4,
                    alpha=0.95,
                    solid_capstyle='round',
                    zorder=5,
                )
                ax._pp_path_line = line
            else:
                line.set_data(pts[:, 0], pts[:, 1])

            # waypoints markers (excluding start)
            wp_pts = pts[1:, :]
            if scatter is None:
                scatter = ax.scatter(
                    wp_pts[:, 0],
                    wp_pts[:, 1],
                    s=35,
                    c=COLORS["path_mark"],
                    edgecolors=COLORS["fig_bg"],
                    linewidths=0.8,
                    alpha=0.95,
                    zorder=6,
                )
                ax._pp_path_scatter = scatter
            else:
                scatter.set_offsets(wp_pts)

            # Optional: label waypoints with their index
            if label_waypoints:
                if wp_texts is None:
                    wp_texts = []
                # Clear previous texts
                for t in list(wp_texts):
                    try:
                        t.remove()
                    except Exception:
                        pass
                wp_texts = []

                # Small vertical offset in data units
                y_offset = 0.015 * float(self.ymax)

                # Numbering corresponds to the order in path.get_waypoints()
                for i in range(wp_pts.shape[0]):
                    x, y = float(wp_pts[i, 0]), float(wp_pts[i, 1])
                    txt = ax.text(
                        x,
                        y + y_offset,
                        str(i + 1),
                        color=COLORS["text"],
                        fontsize=9,
                        ha='center',
                        va='bottom',
                        zorder=7,
                    )
                    wp_texts.append(txt)

                ax._pp_wp_texts = wp_texts
            else:
                if wp_texts:
                    for t in wp_texts:
                        try:
                            t.remove()
                        except Exception:
                            pass
                    ax._pp_wp_texts = []

        if title is None:
            title = f"Board from file: {self.path}"
        ax.set_title(title, loc='center', color=COLORS["text"], fontsize=14, fontweight='bold')

        # draw_idle() can be coalesced by some backends; force a draw+flush so
        # iterative titles/markers update at the expected rate.
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception:
            fig.canvas.draw_idle()

        if pause is not None:
            plt.pause(pause)
        if show:
            plt.show(block=block)

        return fig, ax

    def get_hyperparameters(self)-> dict:
        return {
            'inertia_weight': 0.5,
            'best_position_acceleration': 1.5,
            'global_best_position_acceleration': 1.5,
        }
    
    def get_obstacles(self) -> List[Obstacle]:
        return self.obstacles
    
    def check_line_collision(self, p1 : np.ndarray, p2 : np.ndarray)-> int:
        nb_collisions = 0
        line = LineString([p1, p2])
        for obs in self.obstacles:
            box = Polygon([
                (obs.x, obs.y),
                (obs.x + obs.lx, obs.y),
                (obs.x+obs.lx, obs.y + obs.ly),
                (obs.x, obs.y + obs.ly),
            ])
            if line.intersects(box):
                nb_collisions += 1
        return nb_collisions

if __name__ == "__main__":
    env = Environment()
    print("Enter scenario number: ")
    nb_scenario = int(input())
    env.from_file(f"scenarios/scenario{nb_scenario}.txt")
    path = StaticPath([(100, 200), (600, 900), (1000, 1000)])
    print(path.nb_collisions(env))
    env.render(path)
