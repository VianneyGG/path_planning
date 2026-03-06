from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineDefaults:
    """Single source of truth for every default value used across the pipeline.

    Import ``DEFAULTS`` at the bottom of this module instead of instantiating
    this class directly.
    """

    # ── scenarios ──────────────────────────────────────────────────────────
    scenarios: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # ── benchmark runs ─────────────────────────────────────────────────────
    runs: int = 100
    seed_base: int = 42

    # ── tuning ─────────────────────────────────────────────────────────────
    init_points: int = 20
    n_iter: int = 80
    eval_repeats: int = 2
    grid_warmstart_points: int = 9
    grid_focus_params: int = 4
    hpo_backend: str = "optuna"

    # ── penalties ──────────────────────────────────────────────────────────
    collision_penalty: float = 50.0
    non_collision_free_penalty: float = 200.0
    collision_free_weight: float = 500.0
    no_feasible_penalty: float = 150.0
    time_weight: float = 5.0
    penalty_calibration_runs: int = 2

    # ── parallelism ────────────────────────────────────────────────────────
    n_jobs: int = 1
    chunk_size: int = 24
    adaptive_chunking: bool = True

    # ── pipeline ───────────────────────────────────────────────────────────
    mode: str = "compare"
    exp_id: str = "exp01"
    vanilla_params_summary: str = (
        "src/benchmark/artifacts/basic/tuning/tuning_summary.json"
    )


#: Module-level singleton — use this everywhere instead of hardcoding values.
DEFAULTS = PipelineDefaults()
