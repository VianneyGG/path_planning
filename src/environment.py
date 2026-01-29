from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from typing import List, Tuple


class Obstacle:
    def __init__(self, x, y, lx, ly):
        self.x = x
        self.y = y
        self.lx = lx
        self.ly = ly


class Path:
    def __init__(self, points: List[Tuple[float, float]] = []):
        self.points = points # list of pairs of corrdinates


class Environment:
    def __init__(self):
        self.x = None
        self.y = None
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
                self.x = float(lines[0])
                self.y = float(lines[1])
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

    def render(self, path = None) -> None:
        fig = plt.figure(figsize=(8,6))

        board = Polygon(
            [
                (0, 0),
                (0, self.y),
                (self.x, self.y),
                (self.x, 0),
            ]
        )
        plt.plot(*board.exterior.xy, color='black', linewidth=2)

        for obs in self.obstacles:
            box = Polygon([
                (obs.x, obs.y),
                (obs.x + obs.lx, obs.y),
                (obs.x+obs.lx, obs.y + obs.ly),
                (obs.x, obs.y + obs.ly),
            ])
            plt.fill(*box.exterior.xy, color='red', ec='black', lw=2, alpha=0.8)

        if path is not None:
            lp = self.u1s[0], self.u1s[1]
            # c = 'springgreen'
            c = 'black'
            for p in path.points:
                # c = "springgreen" if c == "skyblue" else "skyblue"
                plt.plot([lp[0], p[0]], [lp[1],p[1]], lw=2, color=c)
                plt.scatter(p[0],p[1],s=40,c='black')
                lp = p

        plt.scatter(self.u1s[0], self.u1s[1], label='Starting', s=128, color='green', zorder=5)
        plt.scatter(self.u1d[0], self.u1d[1], label='Arriving', s=128, color='teal', zorder=5)
        plt.grid(True, lw = 1, alpha=0.3, ls='--', snap=True)
        plt.title(f"Board from file: {self.path}", loc='center', fontdict={"fontsize":16, "weight":"bold"})
        plt.legend(bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = Environment()
    env.from_file("scenarios/scenario4.txt")
    path = Path([(100, 200), (600, 900), (1000, 1000)])
    env.render(path)
