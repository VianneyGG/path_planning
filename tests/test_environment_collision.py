import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.environment import Environment, Obstacle


def make_environment(obstacles):
    env = Environment()
    env.obstacles = obstacles
    return env


def test_line_intersects_obstacle():
    environment = make_environment([Obstacle(1.0, 1.0, 2.0, 2.0)])
    test = environment.check_line_collision((0.0, 0.0), (3.0, 3.0))
    print(f"Collision test1 result: {test}")
    assert test != 0


def test_line_does_not_intersect_obstacle():
    environment = make_environment([Obstacle(1.0, 1.0, 2.0, 2.0)])
    test = environment.check_line_collision((0.0, 0.0), (0.0, 3.0))
    print(f"Collision test2 result: {test}")
    assert test == 0

def test_line_touches_obstacle_edge():
    environment = make_environment([Obstacle(1.0, 1.0, 2.0, 2.0)])
    test = environment.check_line_collision((0.0, 1.0), (3.0, 1.0))
    print(f"Collision test3 result: {test}")
    assert test != 0
    
def test_line_touches_obstacle_corner():
    environment = make_environment([Obstacle(1.0, 1.0, 2.0, 2.0)])
    test = environment.check_line_collision((0.0, 0.0), (3.0, 1.0))
    print(f"Collision test4 result: {test}")
    assert test != 0
    

if __name__ == "__main__":
    tests = [
        ("test_line_intersects_obstacle", test_line_intersects_obstacle),
        ("test_line_does_not_intersect_obstacle", test_line_does_not_intersect_obstacle),
        ("test_line_touches_obstacle_edge", test_line_touches_obstacle_edge),
        ("test_line_touches_obstacle_corner", test_line_touches_obstacle_corner),
    ]

    for name, fn in tests:
        fn()
        print(f"{name} passed")

    print("All collision tests passed")
