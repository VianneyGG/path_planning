from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.PSO.pso_solver import PSO
from src.environment import Environment


ROOT = Path(__file__).resolve().parents[3]


def scenario_path(scenario_id: int) -> Path:
    path = ROOT / "scenarios" / f"scenario{int(scenario_id)}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing scenario file: {path}")
    return path


def extract_metrics(pso: PSO, env: Environment) -> dict[str, Any]:
    if pso.solution is None:
        return {
            "fitness": float("inf"),
            "length": float("inf"),
            "smoothness": float("inf"),
            "collisions": int(1_000_000),
            "corners": int(1_000_000),
            "coords": np.empty((0, 2), dtype=float),
        }

    best = pso.solution
    collisions, corners = best.collisions_and_corners(env, pso.hyperparameters["corner_radius"])
    return {
        "fitness": float(pso.path_fitness(best)),
        "length": float(best.total_length()),
        "smoothness": float(best.smoothness()),
        "collisions": int(collisions),
        "corners": int(corners),
        "coords": np.asarray(best.get_array_coords(), dtype=float),
    }


def objective_cost(
    *,
    fitness: float,
    collisions: int,
    elapsed: float,
    collision_penalty: float,
    non_collision_free_penalty: float,
    time_weight: float,
) -> float:
    collision_free_penalty = float(non_collision_free_penalty) if int(collisions) > 0 else 0.0
    return float(fitness) + float(collision_penalty) * int(collisions) + collision_free_penalty + float(time_weight) * float(elapsed)

