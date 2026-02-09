"""Rapidly-Exploring Random Tree (RRT) with rewiring for path planning."""

import math
import numpy as np
from typing import List, Optional, Set, Tuple, Union

from tqdm import tqdm

import plotly.graph_objects as go


from src.environment import Environment


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _to_xy(p: Union[Tuple[float, float], np.ndarray]) -> Tuple[float, float]:
    if isinstance(p, (tuple, list)) and len(p) >= 2:
        return (float(p[0]), float(p[1]))
    arr = np.asarray(p, dtype=float)
    return (float(arr[0]), float(arr[1]))


def get_position_at_distance(path: List[Tuple[float, float]], d: float) -> Tuple[float, float]:
    """Position on path at distance d from start; if d >= path length returns goal."""
    if not path:
        raise ValueError("path is empty")
    if d <= 0:
        return path[0]
    total = 0.0
    for i in range(1, len(path)):
        seg_len = _distance(path[i - 1], path[i])
        if total + seg_len >= d:
            t = (d - total) / seg_len if seg_len > 1e-12 else 0.0
            return (
                path[i - 1][0] + t * (path[i][0] - path[i - 1][0]),
                path[i - 1][1] + t * (path[i][1] - path[i - 1][1]),
            )
        total += seg_len
    return path[-1]


def discretize_segment(p_a: Tuple[float, float], p_b: Tuple[float, float], step_size: float) -> List[Tuple[float, float]]:
    """Points along segment p_a->p_b with spacing at most step_size."""
    dist = _distance(p_a, p_b)
    if dist <= 1e-12:
        return [p_a]
    n = max(1, int(math.ceil(dist / step_size)))
    points: List[Tuple[float, float]] = []
    for k in range(n + 1):
        t = k / n
        points.append((
            p_a[0] + t * (p_b[0] - p_a[0]),
            p_a[1] + t * (p_b[1] - p_a[1]),
        ))
    return points


class TreeNode:
    """RRT tree node: position, parent, cost from start."""
    def __init__(self, position: Tuple[float, float], parent: Optional["TreeNode"] = None, cost: float = 0.0):
        self.position = position
        self.parent = parent
        self.cost = cost

    def path_to_root(self) -> List[Tuple[float, float]]:
        """Path from this node to root (start) as list of (x,y)."""
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
            t = np.random.rand()
            if edge == 0:
                x, y = obs.x + t * obs.lx, obs.y
            elif edge == 1:
                x, y = obs.x + obs.lx, obs.y + t * obs.ly
            elif edge == 2:
                x, y = obs.x + (1 - t) * obs.lx, obs.y + obs.ly
            else:
                x, y = obs.x, obs.y + (1 - t) * obs.ly
            noise = np.random.normal(loc=0.0, scale=1.5, size=2)
            x = min(max(0.0, x + noise[0]), self.env.xmax)
            y = min(max(0.0, y + noise[1]), self.env.ymax)
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
        """Shortcut path: drop points when direct segment is collision-free."""
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

    def run_algorithm(self, progress_bar: bool = True) -> List[Tuple[float, float]]:
        """Run RRT with rewiring; returns path (x,y) from start to nearest-to-goal node."""
        self.root = TreeNode(self.start, parent=None, cost=0.0)
        self.points = {self.root}

        iterator = tqdm(range(self.n_iter), desc="RRT") if progress_bar else range(self.n_iter)
        for _ in iterator:
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


def _path_cumulative_distances(path: List[Tuple[float, float]]) -> List[float]:
    """Cumulative distance from start at each waypoint (cum[0]=0)."""
    cum: List[float] = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + _distance(path[i - 1], path[i]))
    return cum


class RRTDynamic(RRT):
    """RRT for Robot 2 with Robot 1 as dynamic obstacle (prioritized planning; time=distance)."""

    def __init__(
        self,
        start: Union[Tuple[float, float], np.ndarray],
        target: Union[Tuple[float, float], np.ndarray],
        env: Environment,
        path1: List[Tuple[float, float]],
        R: float,
        delta_s: float,
        delta_r: float,
        n_iter: int = 200,
        p: float = 0.0,
        smooth: bool = True,
        dynamic_step: Optional[float] = None,
    ):
        super().__init__(start, target, env, delta_s, delta_r, n_iter, p, smooth)
        self.path1 = path1
        self.R = R
        self._dynamic_step = dynamic_step if dynamic_step is not None else min(R / 2.0, delta_s / 2.0)

    def _is_segment_collision_free_dynamic(self, p_a: Tuple[float, float], p_b: Tuple[float, float], dist_a: float) -> bool:
        """Check segment for static obstacles and Robot 1 at same distance (time)."""
        if self.env.check_line_collision(np.array(p_a), np.array(p_b)) != 0:
            return False
        for p_2 in discretize_segment(p_a, p_b, self._dynamic_step):
            d = dist_a + _distance(p_a, p_2)
            p_1 = get_position_at_distance(self.path1, d)
            if _distance(p_1, p_2) < self.R:
                return False
        return True

    def _is_collision_free_dynamic(self, node_from: TreeNode, p_to: Tuple[float, float]) -> bool:
        return self._is_segment_collision_free_dynamic(node_from.position, p_to, node_from.cost)

    def _rewire(self, v_new: TreeNode) -> None:
        for v in self._neighbors(v_new, self.delta_r):
            new_cost = v_new.cost + _distance(v_new.position, v.position)
            if new_cost < v.cost and self._is_collision_free_dynamic(v_new, v.position):
                v.parent = v_new
                v.cost = new_cost

    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Shortcut path with dynamic collision check (Robot 1)."""
        if len(path) <= 2:
            return list(path)
        cum = _path_cumulative_distances(path)
        smoothed: List[Tuple[float, float]] = [path[0]]
        current_index = 0
        while current_index < len(path) - 1:
            found = False
            for i in range(len(path) - 1, current_index, -1):
                if self._is_segment_collision_free_dynamic(
                    path[current_index], path[i], cum[current_index]
                ):
                    smoothed.append(path[i])
                    current_index = i
                    found = True
                    break
            if not found:
                current_index += 1
                smoothed.append(path[current_index])
        return smoothed

    def run_algorithm(self, progress_bar: bool = True) -> List[Tuple[float, float]]:
        """Run RRT for Robot 2 with Robot 1 as dynamic obstacle."""
        self.root = TreeNode(self.start, parent=None, cost=0.0)
        self.points = {self.root}

        iterator = tqdm(range(self.n_iter), desc="RRT (Robot 2)") if progress_bar else range(self.n_iter)
        for _ in iterator:
            v_rand = self._biased_sample()
            v_nearest = self._nearest(v_rand)
            if v_nearest is None:
                continue
            v_new_xy = self._steer(v_nearest.position, v_rand, self.delta_s)
            if not self._is_collision_free_dynamic(v_nearest, v_new_xy):
                continue
            v_new = TreeNode(
                v_new_xy,
                parent=v_nearest,
                cost=v_nearest.cost + _distance(v_nearest.position, v_new_xy),
            )
            self.points.add(v_new)
            self._rewire(v_new)

        final = self._nearest(self.target)
        if final is None:
            return [self.start]
        path = final.path_to_root()
        if self.smooth:
            path = self._smooth_path(path)
        return path


def multi_robot_planner(
    env: Environment,
    delta_s: float = 40.0,
    delta_r: float = 120.0,
    n_iter: int = 2000,
    p: float = 0.2,
    smooth: bool = True,
    progress_bar: bool = True,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Prioritized planning: Path1 with standard RRT, then Path2 with RRT treating Robot 1 as dynamic obstacle.
    Uses env.u1s, u1d, u2s, u2d, env.R. Returns (path1, path2).
    """
    if env.u1s is None or env.u1d is None or env.u2s is None or env.u2d is None or env.R is None:
        raise ValueError("Environment must have u1s, u1d, u2s, u2d, and R set (e.g. from_file for two-robot scenario)")
    start1 = _to_xy(env.u1s)
    goal1 = _to_xy(env.u1d)
    start2 = _to_xy(env.u2s)
    goal2 = _to_xy(env.u2d)
    R = float(env.R)

    rrt1 = RRT(start1, goal1, env, delta_s, delta_r, n_iter=n_iter, p=p, smooth=smooth)
    path1 = rrt1.run_algorithm(progress_bar=progress_bar)

    rrt2 = RRTDynamic(
        start2, goal2, env, path1, R, delta_s, delta_r, n_iter=n_iter, p=p, smooth=smooth
    )
    path2 = rrt2.run_algorithm(progress_bar=progress_bar)

    return path1, path2


def _path_length(path: List[Tuple[float, float]]) -> float:
    """Total length of path (sum of segment lengths)."""
    if len(path) <= 1:
        return 0.0
    return sum(_distance(path[i - 1], path[i]) for i in range(1, len(path)))


def export_rrt_animation_html(env: Environment, path1: List[Tuple[float, float]], path2: Optional[List[Tuple[float, float]]] = None,
                             html_path: str = "rrt_animation.html", n_frames: int = 60, frame_duration_ms: int = 80,
                             title: Optional[str] = None, include_plotlyjs: bool = True) -> None:
    """Export Plotly HTML animation of robot(s) moving along paths (same speed=distance). Single or two robots."""
    if go is None:
        raise ImportError("plotly is required for RRT animation. Install with: pip install plotly")
    len1 = _path_length(path1)
    len2 = _path_length(path2) if path2 else 0.0
    max_len = max(len1, len2, 1.0)
    xmax = float(env.xmax)
    ymax = float(env.ymax)

    def frame_data(i: int) -> list:
        d = (i / max(1, n_frames - 1)) * max_len
        p1 = get_position_at_distance(path1, d)
        traces: list = []
        pts1 = np.array(path1, dtype=float)
        traces.append(
            go.Scatter(
                x=pts1[:, 0],
                y=pts1[:, 1],
                mode="lines",
                line=dict(color="#4da3ff", width=2, dash="dot"),
                name="Path 1",
                legendgroup="path1",
            )
        )
        if path2 is not None:
            pts2 = np.array(path2, dtype=float)
            traces.append(
                go.Scatter(
                    x=pts2[:, 0],
                    y=pts2[:, 1],
                    mode="lines",
                    line=dict(color="#f59e0b", width=2, dash="dot"),
                    name="Path 2",
                    legendgroup="path2",
                )
            )
        traces.append(
            go.Scatter(
                x=[p1[0]],
                y=[p1[1]],
                mode="markers",
                marker=dict(color="#34d399", size=14, symbol="circle", line=dict(color="#0f1218", width=2)),
                name="Robot 1",
                legendgroup="r1",
            )
        )
        if path2 is not None:
            p2 = get_position_at_distance(path2, d)
            traces.append(
                go.Scatter(
                    x=[p2[0]],
                    y=[p2[1]],
                    mode="markers",
                    marker=dict(color="#f59e0b", size=14, symbol="circle", line=dict(color="#0f1218", width=2)),
                    name="Robot 2",
                    legendgroup="r2",
                )
            )
        return traces

    data0 = frame_data(0)
    fig = go.Figure(data=data0)

    for obs in env.get_obstacles():
        fig.add_shape(
            type="rect",
            x0=obs.x,
            y0=obs.y,
            x1=obs.x + obs.lx,
            y1=obs.y + obs.ly,
            line=dict(color="#6b2d45", width=2),
            fillcolor="#2a1b23",
            layer="below",
        )

    if env.u1s is not None:
        fig.add_trace(
            go.Scatter(
                x=[env.u1s[0]],
                y=[env.u1s[1]],
                mode="markers",
                marker=dict(color="#34d399", size=10, symbol="x-open"),
                name="Start 1",
            )
        )
    if env.u1d is not None:
        fig.add_trace(
            go.Scatter(
                x=[env.u1d[0]],
                y=[env.u1d[1]],
                mode="markers",
                marker=dict(color="#60a5fa", size=10, symbol="x-open"),
                name="Goal 1",
            )
        )
    if path2 is not None and env.u2s is not None:
        fig.add_trace(
            go.Scatter(
                x=[env.u2s[0]],
                y=[env.u2s[1]],
                mode="markers",
                marker=dict(color="#fbbf24", size=10, symbol="x-open"),
                name="Start 2",
            )
        )
    if path2 is not None and env.u2d is not None:
        fig.add_trace(
            go.Scatter(
                x=[env.u2d[0]],
                y=[env.u2d[1]],
                mode="markers",
                marker=dict(color="#f59e0b", size=10, symbol="x-open"),
                name="Goal 2",
            )
        )

    fig.update_layout(
        title=dict(text=title or "RRT – Robot(s) motion", font=dict(size=18)),
        margin=dict(t=80, b=60, l=60, r=60),
        xaxis=dict(
            range=[0, xmax],
            autorange=False,
            fixedrange=True,
            constrain="domain",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            range=[0, ymax],
            autorange=False,
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True,
            constrain="domain",
            tickfont=dict(size=11),
        ),
        plot_bgcolor="#121826",
        paper_bgcolor="#0f1218",
        font=dict(color="#d6deeb", size=12),
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor="left", font=dict(size=12)),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.02,
                xanchor="left",
                yanchor="bottom",
                bgcolor="#2a3344",
                bordercolor="#8b95a7",
                borderwidth=1,
                font=dict(size=13),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": frame_duration_ms, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        method="animate",
                        args=[[f"frame_{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        label=str(i),
                    )
                    for i in range(n_frames)
                ],
                x=0.02,
                y=0.02,
                len=0.96,
                xanchor="left",
                font=dict(size=10),
                currentvalue=dict(visible=True, prefix="Frame: ", font=dict(size=11)),
                transition=dict(duration=0),
            )
        ],
    )

    fig.frames = [go.Frame(name=f"frame_{i}", data=frame_data(i)) for i in range(n_frames)]
    fig.write_html(html_path, auto_open=False, include_plotlyjs=include_plotlyjs)
