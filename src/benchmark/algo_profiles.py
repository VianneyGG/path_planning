from __future__ import annotations

from typing import Any


ALGO_FLAGS: dict[str, dict[str, bool]] = {
    "vanilla": {
        "reset_waypoints": False,
        "dimensional_learning": False,
        "simulated_annealing": False,
        "controlled_cooling": False,
        "pre_heat": False,
    },
    "RS": {
        "reset_waypoints": True,
        "dimensional_learning": False,
        "simulated_annealing": False,
        "controlled_cooling": False,
        "pre_heat": False,
    },
    "RS_SA_noCC": {
        "reset_waypoints": True,
        "dimensional_learning": False,
        "simulated_annealing": True,
        "controlled_cooling": False,
        "pre_heat": False,
    },
    "RS_SA_noCC_DL": {
        "reset_waypoints": True,
        "dimensional_learning": True,
        "simulated_annealing": True,
        "controlled_cooling": False,
        "pre_heat": False,
    },
    "RS_SA_CC": {
        "reset_waypoints": True,
        "dimensional_learning": False,
        "simulated_annealing": True,
        "controlled_cooling": True,
        "pre_heat": True,
    },
}


ALGO_LABELS: dict[str, str] = {
    "vanilla": "Basic",
    "RS": "RS",
    "RS_SA_noCC": "RS+SAnoCC",
    "RS_SA_noCC_DL": "RS+SAnoCC+DL",
    "RS_SA_CC": "RS+SA+CC",
}

# Ordered list of algorithm *keys* used to generate the blue->green color gradient for plots.
# Adjust this order if you want a different color assignment; labels are taken from ALGO_LABELS.
ALGO_PLOT_ORDER: list[str] = [
    "vanilla",
    "RS",
    "RS_SA_noCC",
    "RS_SA_noCC_DL",
    "RS_SA_CC",
]


DEFAULT_SEARCH_SPACE: dict[str, dict[str, tuple[float, float]]] = {
    "vanilla": {
        "number_of_particules": (1.0, 150.0),
        "number_of_iterations": (5.0, 400.0),
        
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
    },
    "RS": {
        "number_of_particules": (1.0, 150.0),
        "number_of_iterations": (10.0, 400.0),
        
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
        "reset_number": (1.0, 4.0),
    },
    "RS_SA_noCC": {
        "number_of_particules": (10.0, 150.0),
        "number_of_iterations": (30.0, 500.0),
    
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
        "reset_number": (1.0, 4.0),
        "initial_temperature": (2, 60.0),
    },
    "RS_SA_noCC_DL": {
        "number_of_particules": (10.0, 150.0),
        "number_of_iterations": (30.0, 500.0),
        
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
        "reset_number": (1.0, 4.0),
        "initial_temperature": (2, 60.0),
        "max_number_of_iterations_without_improvement": (25.0, 50.0),
    },
    "RS_SA_CC": {
        "number_of_particules": (10.0, 150.0),
        "number_of_iterations": (30.0, 500.0),
        
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
        "reset_number": (1.0, 4.0),
        "acceptance_probability_decay": (0.9, 0.999),
    },
}


# Waypoint bounds are scenario-specific (low, high) and no longer defined per-algo.
SCENARIO_WAYPOINT_BOUNDS: dict[int, tuple[float, float]] = {
    0: (1.0, 1.0),
    1: (1.0, 3.0),
    2: (2.0, 4.0),
    3: (3.0, 4.0),
    4: (4.0, 6.0),
}


INT_HYPERPARAMETERS = {
    "number_of_particules",
    "number_of_iterations",
    "number_of_waypoints",
    "reset_number",
    "max_number_of_iterations_without_improvement",
}


def get_algo_flags(algo: str) -> dict[str, bool]:
    if algo not in ALGO_FLAGS:
        raise ValueError(f"Unknown algo '{algo}'. Allowed: {list(ALGO_FLAGS.keys())}")
    return dict(ALGO_FLAGS[algo])


def get_search_space(algo: str) -> dict[str, tuple[float, float]]:
    if algo not in DEFAULT_SEARCH_SPACE:
        raise ValueError(f"Unknown algo '{algo}'. Allowed: {list(DEFAULT_SEARCH_SPACE.keys())}")
    return dict(DEFAULT_SEARCH_SPACE[algo])


def cast_hyperparameters(raw_params: dict[str, float], base_config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base_config)
    for key, value in raw_params.items():
        if key in INT_HYPERPARAMETERS:
            cfg[key] = int(round(float(value)))
        else:
            cfg[key] = float(value)

    cfg["number_of_particules"] = max(1, int(cfg.get("number_of_particules", 1)))
    cfg["number_of_iterations"] = max(1, int(cfg.get("number_of_iterations", 1)))
    cfg["number_of_waypoints"] = max(1, int(cfg.get("number_of_waypoints", 1)))
    cfg["reset_number"] = max(1, int(cfg.get("reset_number", 1)))
    if "max_number_of_iterations_without_improvement" in cfg:
        cfg["max_number_of_iterations_without_improvement"] = max(
            1, int(cfg["max_number_of_iterations_without_improvement"])
        )

    return cfg


def apply_algo_flags(config: dict[str, Any], algo: str) -> dict[str, Any]:
    cfg = dict(config)
    cfg.update(get_algo_flags(algo))
    return cfg
