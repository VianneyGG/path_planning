import numpy as np
from src.environment import Environment
from src.PSO.Particule import Particule
from src.PSO.Path import Path
from src.PSO.Swarm import Swarm
from src.PSO.Waypoint import Waypoint


def test_velocity_shape_matches_position_after_update() -> None:
    env = Environment()
    env.from_file("scenarios/scenario0.txt")
    hyperparams = {
        'inertia_weight': 0.5,
        'best_position_acceleration': 1.5,
        'global_best_position_acceleration': 1.5,
        'length_weight': 1.0,
        'smoothness_weight': 100.0,
        'collision_weight': 3000.0,
        'corner_weight': -200.0,
        'corner_radius': 5.0,
    }

    part = Particule.initialize_particule(env, hyperparams, number_of_waypoints=4)
    before_shape = part.velocity.shape
    part.update_velocity(part.path.get_fixed_mask(), part.get_position().copy(), hyperparams)
    after_shape = part.velocity.shape

    assert before_shape == after_shape, "Velocity shape changed after update"
    assert after_shape == part.position.shape == part.best_position.shape


def test_masks_remain_2d_and_need_fix() -> None:
    env = Environment()
    env.from_file("scenarios/scenario0.txt")
    hyperparams = {
        'inertia_weight': 0.5,
        'best_position_acceleration': 1.5,
        'global_best_position_acceleration': 1.5,
        'length_weight': 1.0,
        'smoothness_weight': 100.0,
        'collision_weight': 3000.0,
        'corner_weight': -200.0,
        'corner_radius': 5.0,
    }

    part = Particule.initialize_particule(env, hyperparams, number_of_waypoints=4)
    recorded: dict[str, np.ndarray] = {}

    orig_update_velocity = Particule.update_velocity
    orig_update_positions = part.path.update_positions

    def patch_update_velocity(self, fixed_mask, best_global_position, hyperparameters):
        mask = np.array(fixed_mask, dtype=bool)
        recorded['velocity_mask'] = mask
        return orig_update_velocity(self, fixed_mask, best_global_position, hyperparameters)

    def patch_update_positions(new_positions, xmax, ymax):
        mask = orig_update_positions(new_positions, xmax, ymax)
        recorded['hit_border'] = mask
        return mask

    try:
        Particule.update_velocity = patch_update_velocity
        part.path.update_positions = patch_update_positions
        part.update_velocity(part.path.get_fixed_mask(), part.get_position().copy(), hyperparams)
        part.update_position(env.xmax, env.ymax)
    finally:
        Particule.update_velocity = orig_update_velocity
        part.path.update_positions = orig_update_positions

    assert recorded['velocity_mask'].ndim == 1
    assert recorded['hit_border'].ndim == 1


def test_swarm_forward_updates_best_path_when_improvement_available() -> None:
    env = Environment()
    env.from_file("scenarios/scenario0.txt")

    hyperparams = {
        'inertia_weight': 0.5,
        'best_position_acceleration': 1.5,
        'global_best_position_acceleration': 1.5,
        'length_weight': 1.0,
        'smoothness_weight': 100.0,
        'collision_weight': 3000.0,
        'corner_weight': -200.0,
        'corner_radius': 5.0,
    }
    swarm = Swarm.initialize_swarm(
        num_particules=2,
        env=env,
        hyperparameters=hyperparams,
        number_of_waypoints=4,
    )

    better = swarm.particules[1]
    worse = swarm.particules[0]
    better._test_fitness = 1.0
    worse._test_fitness = 10.0

    orig_eval = Particule.evaluate_fitness

    def fake_eval(self, env_arg, hyperparams):
        orig_eval(self, env_arg, hyperparams)
        target = getattr(self, '_test_fitness', None)
        if target is not None:
            self.fitness = target
            self.best_fitness = target
            self.best_position = self.position.copy()

    try:
        Particule.evaluate_fitness = fake_eval
        swarm.forward(env, hyperparams, temperature=1.0, simulated_annealing=False, dimensional_learning=False)
    finally:
        Particule.evaluate_fitness = orig_eval

    assert np.allclose(swarm.get_best_path().get_array_coords(), better.path.get_array_coords())
    assert np.allclose(swarm.get_global_best_position(), better.get_position())


def test_reset_waypoints_retains_best_path() -> None:
    env = Environment()
    env.from_file("scenarios/scenario0.txt")
    hyperparams = {
        'inertia_weight': 0.5,
        'best_position_acceleration': 1.5,
        'global_best_position_acceleration': 1.5,
        'length_weight': 1.0,
        'smoothness_weight': 100.0,
        'collision_weight': 3000.0,
        'corner_weight': -200.0,
        'corner_radius': 5.0,
    }
    swarm = Swarm.initialize_swarm(2, env, hyperparams, 4)
    swarm.forward(env, hyperparams, temperature=1.0, simulated_annealing=False, dimensional_learning=False)
    saved_best = swarm.get_best_path().get_array_coords().copy()

    swarm.reset_waypoints(env, number_of_waypoints=4, hyperparameters=hyperparams)

    assert np.allclose(swarm.get_best_path().get_array_coords(), saved_best)
    assert np.allclose(swarm.get_global_best_position(), saved_best)


def test_simulated_annealing_does_not_replace_best_path() -> None:
    env = Environment()
    env.from_file("scenarios/scenario0.txt")
    hyperparams = {
        'inertia_weight': 0.5,
        'best_position_acceleration': 1.5,
        'global_best_position_acceleration': 1.5,
        'length_weight': 1.0,
        'smoothness_weight': 100.0,
        'collision_weight': 3000.0,
        'corner_weight': -200.0,
        'corner_radius': 5.0,
    }
    swarm = Swarm.initialize_swarm(2, env, hyperparams, 4)
    swarm.forward(env, hyperparams, temperature=1.0, simulated_annealing=False, dimensional_learning=False)
    saved_best = swarm.get_best_path().get_array_coords().copy()

    orig_eval = Particule.evaluate_fitness

    def degraded_eval(self, env_arg, hyperparams_arg):
        orig_eval(self, env_arg, hyperparams_arg)
        self.fitness = 1e6
        self.best_fitness = 1e6

    try:
        Particule.evaluate_fitness = degraded_eval
        swarm.forward(env, hyperparams, temperature=1.0, simulated_annealing=True, dimensional_learning=False)
    finally:
        Particule.evaluate_fitness = orig_eval

    assert np.allclose(swarm.get_best_path().get_array_coords(), saved_best)


def test_path_pruning_removes_near_straight_waypoints() -> None:
    waypoints = [
        Waypoint(0.0, 0.0, True),
        Waypoint(100.0, 0.0, False),
        Waypoint(200.0, 0.0, True),
    ]
    path = Path(waypoints)
    smooth = path.smoothness(drop_near_straight=True, tolerance=1e-3)
    assert len(path.get_waypoints()) == 2
    assert smooth >= 0.0


def test_particule_prune_keeps_arrays_consistent() -> None:
    env = Environment()
    env.from_file("scenarios/scenario0.txt")
    waypoints = [
        Waypoint(0.0, 0.0, True),
        Waypoint(100.0, 0.0, False),
        Waypoint(200.0, 0.0, True),
    ]
    path = Path(waypoints)
    part = Particule(path)
    part.position = part.path.get_array_coords()
    part.best_position = part.position.copy()
    part.velocity = np.zeros_like(part.position)
    hyperparams = {
        'inertia_weight': 0.5,
        'best_position_acceleration': 1.5,
        'global_best_position_acceleration': 1.5,
        'length_weight': 1.0,
        'smoothness_weight': 100.0,
        'collision_weight': 3000.0,
        'corner_weight': -200.0,
        'corner_radius': 5.0,
        'prune_straight_angles': True,
        'straight_angle_tolerance': 1e-2,
    }
    part.evaluate_fitness(env, hyperparams)
    assert part.position.shape[0] == 2
    assert part.velocity.shape == part.position.shape