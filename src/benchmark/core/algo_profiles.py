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
    "RS_SA_PH": {
        "reset_waypoints": True,
        "dimensional_learning": False,
        "simulated_annealing": True,
        "controlled_cooling": False,
        "pre_heat": True,
    },
    "RS_SA_CC_DL": {
        "reset_waypoints": True,
        "dimensional_learning": True,
        "simulated_annealing": True,
        "controlled_cooling": True,
        "pre_heat": True,
    },
}


ALGO_LABELS: dict[str, str] = {
    "vanilla": "Basic",
    "RS": "RS",
    "RS_SA_noCC": "SA",
    "RS_SA_noCC_DL": "SA+DL",
    "RS_SA_CC": "SA+CC",
    "RS_SA_PH": "SA+PH",
    "RS_SA_CC_DL": "SA+CC+DL",
}

# Ordered list of algorithm *keys* used to generate the blue->green color gradient for plots.
# Adjust this order if you want a different color assignment; labels are taken from ALGO_LABELS.
ALGO_PLOT_ORDER: list[str] = [
    "vanilla",
    "RS",
    "RS_SA_noCC",
    "RS_SA_noCC_DL",
    "RS_SA_CC",
    "RS_SA_PH",
    "RS_SA_CC_DL",
]

DEFAULT_SEARCH_SPACE: dict[str, dict[str, tuple[float, float]]] = {
    "vanilla": {
        "number_of_particules": (1.0, 150.0),
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
    },
    "RS": {
        "number_of_particules": (1.0, 150.0),
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
        "reset_number": (1.0, 5.0),
    },
    "RS_SA_noCC": {
        "number_of_particules": (10.0, 150.0),
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
        "reset_number": (1.0, 10.0),
        "initial_temperature": (2, 60.0),
    },
    "RS_SA_noCC_DL": {
        "number_of_particules": (10.0, 150.0),
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
        "reset_number": (1.0, 10.0),
        "initial_temperature": (2, 60.0),
        "max_number_of_iterations_without_improvement": (10.0, 50.0),
    },
    "RS_SA_CC": {
        "number_of_particules": (10.0, 150.0),
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
        "reset_number": (1.0, 10.0),
        "initial_temperature": (2, 60.0),
        "cc_temperature_decay": (0.985, 0.9999),
        "pre_heat_target_acceptance_rate": (0.8, 0.99),
    },
    "RS_SA_PH": {
        "number_of_particules": (10.0, 150.0),
        "inertia_weight": (0.3, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
        "reset_number": (1.0, 10.0),
        "initial_temperature": (1.0, 60.0),
        "temperature_decay": (0.98, 0.999),
        "pre_heat_target_acceptance_rate": (0.6, 0.99),
        "pre_heat_max_iterations": (50.0, 300.0),
    },
    "RS_SA_CC_DL": {
        "number_of_particules": (10.0, 150.0),
        "inertia_weight": (0.4, 0.9),
        "best_position_acceleration": (1.0, 2.5),
        "global_best_position_acceleration": (1.0, 2.5),
        "reset_number": (1.0, 10.0),
        "initial_temperature": (2, 60.0),
        "cc_temperature_decay": (0.985, 0.9999),
        "max_number_of_iterations_without_improvement": (10.0, 50.0),
        "pre_heat_target_acceptance_rate": (0.8, 0.99),
    },
}


# Waypoint bounds are scenario-specific (low, high) and no longer defined per-algo.
SCENARIO_WAYPOINT_BOUNDS: dict[int, tuple[float, float]] = {
    0: (1.0, 1.0),
    1: (1.0, 3.0),
    2: (3.0, 4.0),
    3: (4.0, 5.0),
    4: (5.0, 6.0),
}

# Iteration bounds per scenario: used as [min, max] clip range when pruning
# the fixed-budget number_of_iterations via stagnation detection post-tuning.
SCENARIO_ITERATION_BOUNDS: dict[int, tuple[float, float]] = {
    0: (10.0,  25.0),
    1: (20.0,  50.0),
    2: (200.0, 500.0),
    3: (300.0, 600.0),
    4: (600.0, 1000.0),
}

# Fixed iteration budget used during tuning (not tuned as a hyperparameter).
# This is a generous upper bound; the actual number_of_iterations is determined
# via stagnation-based pruning applied after tuning completes.
SCENARIO_TUNING_BUDGET: dict[int, int] = {
    0: 100,
    1: 200,
    2: 600,
    3: 800,
    4: 1300,
}


INT_HYPERPARAMETERS = {
    "number_of_particules",
    "number_of_iterations",
    "number_of_waypoints",
    "reset_number",
    "pre_heat_max_iterations",
    "max_number_of_iterations_without_improvement",
}

# Default reset number used by PSO and the benchmarking pipeline when not specified.
DEFAULT_RESET_NUMBER: int = 2


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

    if (
        "acceptance_probability_decay" not in raw_params
        and (
            "number_of_iterations" in raw_params
            or "reset_number" in raw_params
        )
    ):
        cfg["acceptance_probability_decay"] = None

    return cfg


def apply_algo_flags(config: dict[str, Any], algo: str) -> dict[str, Any]:
    cfg = dict(config)
    cfg.update(get_algo_flags(algo))
    return cfg
