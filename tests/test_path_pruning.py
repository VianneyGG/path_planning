import numpy as np
from src.PSO.Path import Path
from src.PSO.Waypoint import Waypoint

def test_prune_removes_multiple_colinear_segments():
    # Three consecutive colinear segments, all non-fixed
    waypoints = [
        Waypoint(0.0, 0.0, True),
        Waypoint(50.0, 0.0, False),
        Waypoint(100.0, 0.0, False),
        Waypoint(150.0, 0.0, False),
        Waypoint(200.0, 0.0, True),
    ]
    path = Path(waypoints)
    smooth = path.smoothness(drop_near_straight=True, tolerance=1e-3)
    # At least one intermediate waypoint should remain (we keep one per consecutive run)
    coords = path.get_array_coords()
    assert coords.shape[0] == 3
    assert np.allclose(coords[0], [0.0, 0.0])
    assert np.allclose(coords[-1], [200.0, 0.0])
    # middle should be one of the original non-fixed waypoints
    assert any(np.allclose(coords[1], p) for p in ([50.0, 0.0], [100.0, 0.0], [150.0, 0.0]))
    assert smooth >= 0.0

def test_prune_keeps_noncolinear():
    # Only one angle is nearly straight, the other is not
    waypoints = [
        Waypoint(0.0, 0.0, True),
        Waypoint(100.0, 0.0, False),
        Waypoint(150.0, 50.0, False),
        Waypoint(200.0, 0.0, True),
    ]
    path = Path(waypoints)
    smooth = path.smoothness(drop_near_straight=True, tolerance=1e-3)
    coords = path.get_array_coords()
    # Should keep all waypoints (none are nearly colinear)
    assert coords.shape[0] == 4
    assert smooth >= 0.0


def test_prune_removes_almost_colinear():
    # The middle waypoint is almost, but not exactly, colinear
    waypoints = [
        Waypoint(0.0, 0.0, True),
        Waypoint(100.0, 0.0, False),
        Waypoint(200.0, 0.001, False),  # Slightly off the line
        Waypoint(300.0, 0.0, True),
    ]
    path = Path(waypoints)
    smooth = path.smoothness(drop_near_straight=True, tolerance=1e-2)
    coords = path.get_array_coords()
    # Should prune the almost-colinear point (200, 0.001)
    assert coords.shape[0] == 3
    assert np.allclose(coords[0], [0.0, 0.0])
    assert np.allclose(coords[-1], [300.0, 0.0])
    # The remaining non-fixed should be the one not pruned
    assert any(np.allclose(coords[i], [100.0, 0.0]) for i in range(1, coords.shape[0]-1))
    assert smooth >= 0.0

def test_prune_does_not_remove_fixed():
    # Middle is fixed, should not be pruned
    waypoints = [
        Waypoint(0.0, 0.0, True),
        Waypoint(100.0, 0.0, True),
        Waypoint(200.0, 0.0, True),
    ]
    path = Path(waypoints)
    smooth = path.smoothness(drop_near_straight=True, tolerance=1e-3)
    coords = path.get_array_coords()
    assert coords.shape[0] == 3
    assert np.allclose(coords[1], [100.0, 0.0])
    assert smooth >= 0.0

def test_prune_handles_short_paths():
    # Path with only two points should not error
    waypoints = [
        Waypoint(0.0, 0.0, True),
        Waypoint(100.0, 0.0, True),
    ]
    path = Path(waypoints)
    smooth = path.smoothness(drop_near_straight=True, tolerance=1e-3)
    coords = path.get_array_coords()
    assert coords.shape[0] == 2
    assert smooth == 0.0
