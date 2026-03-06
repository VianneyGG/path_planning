"""Smoke tests: fast sanity checks that don't require a full run.

Run with:
    uv run pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

def test_import_core_config() -> None:
    from src.benchmark.core.config import DEFAULTS, PipelineDefaults  # noqa: F401
    assert isinstance(DEFAULTS, PipelineDefaults)


def test_import_core_algo_profiles() -> None:
    from src.benchmark.core.algo_profiles import apply_algo_flags, get_search_space  # noqa: F401


def test_import_core_common() -> None:
    from src.benchmark.core.common import extract_metrics, objective_cost, scenario_path  # noqa: F401


def test_import_jobs_tune() -> None:
    from src.benchmark.jobs.tune_algo_bayes import main  # noqa: F401
    assert callable(main)


def test_import_jobs_benchmark() -> None:
    from src.benchmark.jobs.benchmark_algo_vs_basic import main  # noqa: F401
    assert callable(main)


def test_import_jobs_pipeline() -> None:
    from src.benchmark.jobs.run_pipeline import main  # noqa: F401
    assert callable(main)


def test_import_viz_plot_tuning_curves() -> None:
    from src.benchmark.viz.plot_tuning_curves import main  # noqa: F401
    assert callable(main)


# ---------------------------------------------------------------------------
# 2. Config defaults
# ---------------------------------------------------------------------------

def test_defaults_values() -> None:
    from src.benchmark.core.config import DEFAULTS
    assert DEFAULTS.collision_penalty == 50.0
    assert DEFAULTS.non_collision_free_penalty == 200.0
    assert DEFAULTS.time_weight == 5.0
    assert DEFAULTS.chunk_size == 24
    assert DEFAULTS.n_iter == 80


# ---------------------------------------------------------------------------
# 3. CLI --help (no crash, no real computation)
# ---------------------------------------------------------------------------

def _help_exits_cleanly(main_fn, extra_args: list[str] | None = None) -> None:
    """Assert that calling main(['--help']) raises SystemExit(0)."""
    with pytest.raises(SystemExit) as exc_info:
        main_fn(["--help"] + (extra_args or []))
    assert exc_info.value.code == 0


def test_pipeline_help() -> None:
    from src.benchmark.jobs.run_pipeline import main
    _help_exits_cleanly(main)


def test_tune_help() -> None:
    from src.benchmark.jobs.tune_algo_bayes import main
    _help_exits_cleanly(main)


def test_benchmark_help() -> None:
    from src.benchmark.jobs.benchmark_algo_vs_basic import main
    _help_exits_cleanly(main)


def test_plot_tuning_curves_help() -> None:
    from src.benchmark.viz.plot_tuning_curves import main
    _help_exits_cleanly(main)


# ---------------------------------------------------------------------------
# 4. Dry-run pipeline (no computation, just arg parsing + step planning)
# ---------------------------------------------------------------------------

def test_pipeline_dry_run(tmp_path: Path) -> None:
    from src.benchmark.jobs.run_pipeline import main
    # Should complete in < 1s with no computation
    main([
        "--algo", "RS",
        "--scenarios", "0",
        "--runs", "1",
        "--n-iter", "1",
        "--init-points", "1",
        "--exp-id", "_smoke_dryrun",
        "--dry-run",
    ])


# ---------------------------------------------------------------------------
# 5. Scenario path resolution
# ---------------------------------------------------------------------------

def test_scenario_paths_exist() -> None:
    from src.benchmark.core.common import scenario_path
    for sid in range(5):
        p = scenario_path(sid)
        assert p.exists(), f"scenario{sid}.txt not found at {p}"


# ---------------------------------------------------------------------------
# 6. objective_cost sanity
# ---------------------------------------------------------------------------

def test_objective_cost_no_collision() -> None:
    from src.benchmark.core.common import objective_cost
    cost = objective_cost(
        fitness=10.0,
        collisions=0,
        elapsed=1.0,
        collision_penalty=50.0,
        non_collision_free_penalty=200.0,
        time_weight=5.0,
    )
    # 10 + 50*0 + 0 (no ncf penalty) + 5*1 = 15
    assert cost == pytest.approx(15.0)


def test_objective_cost_with_collision() -> None:
    from src.benchmark.core.common import objective_cost
    cost = objective_cost(
        fitness=10.0,
        collisions=2,
        elapsed=1.0,
        collision_penalty=50.0,
        non_collision_free_penalty=200.0,
        time_weight=5.0,
    )
    # 10 + 50*2 + 200 + 5*1 = 315
    assert cost == pytest.approx(315.0)


def test_optuna_logging_silenced(capsys) -> None:
    """Ensure optuna backend doesn't spew trial messages.

    The tuner bumps optuna's logger to WARNING; we verify that the
    verbosity is raised and that no "Trial ... finished" text appears in
    stdout/stderr when _maximize_with_backend is invoked with the
    optuna backend.  Skip the test if optuna isn't installed.
    """
    from src.benchmark.jobs.tune_algo_bayes import _maximize_with_backend
    import optuna

    if optuna is None:
        pytest.skip("optuna not installed")

    # reset to info so that our call has somewhere to improve from
    optuna.logging.set_verbosity(optuna.logging.INFO)
    _maximize_with_backend(
        backend="optuna",
        objective=lambda **kwargs: 0.0,
        search_space={},
        init_points=0,
        n_iter=0,
        random_seed=0,
        grid_warmstart_points=0,
        grid_focus_params=1,
    )

    # verbosity should have been raised to WARNING
    assert optuna.logging.get_verbosity() >= optuna.logging.WARNING
    captured = capsys.readouterr()
    assert "Trial" not in captured.out
    assert "finished" not in captured.out


def test_find_stagnation_iter_simple() -> None:
    """Basic sanity for the stagnation finder using a fabricated curve."""
    from src.benchmark.jobs.tune_algo_bayes import _find_stagnation_iter
    import pandas as pd

    # Fitness is a cost that DECREASES as the run improves, then flattens.
    rows: list[dict[str, float]] = []
    for it in range(1, 21):
        fitness = float(20 - it if it <= 10 else 10)  # cost: 19→10, then flat at 10
        rows.append({"pso_iteration": it, "best_fitness": fitness, "is_collision_free": True})
    df = pd.DataFrame(rows)
    pruned = _find_stagnation_iter(df, min_iters=1, max_iters=100, window=3, threshold=0.2)
    # diff(3) at iter 13 = 10 - 10 = 0; 0 > -0.2 is True → stagnation detected at 13
    assert pruned == 13


def test_prune_iterations_in_summary(tmp_path: Path) -> None:
    """Ensure the summary JSON is rewritten with pruned iteration values."""
    import pandas as pd
    import json
    from src.benchmark.jobs.tune_algo_bayes import _prune_iterations_in_summary

    # build a mini pso_df with two scenario rows and one global row
    rows = [
        {"scope": "scenario", "scenario": 0, "pso_iteration": 1, "best_fitness": 1.0, "is_collision_free": True},
        {"scope": "scenario", "scenario": 0, "pso_iteration": 2, "best_fitness": 1.0, "is_collision_free": True},
        {"scope": "global", "scenario": 0, "pso_iteration": 1, "best_fitness": 1.0, "is_collision_free": True},
        {"scope": "global", "scenario": 0, "pso_iteration": 2, "best_fitness": 1.0, "is_collision_free": True},
    ]
    pso_df = pd.DataFrame(rows)

    summary = {
        "per_scenario": [
            {"scenario": 0, "best_params": {"number_of_iterations": 500}},
        ],
        "global": {"best_params_by_scenario": {"0": {"number_of_iterations": 500}}},
    }
    summary_path = tmp_path / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh)

    # run pruning with a very low threshold so that stagnation occurs early
    _prune_iterations_in_summary(summary_path, pso_df, stagnation_window=2, stagnation_threshold=0.5)

    with summary_path.open("r", encoding="utf-8") as fh:
        new = json.load(fh)

    per = new["per_scenario"][0]["best_params"]
    assert per["tuning_iter_budget"] == 500
    # since curve is flat improvement=0, expect pruned >= min bound (10) and <= max
    assert 10 <= per["number_of_iterations"] <= 500

    glob = new["global"]["best_params_by_scenario"]["0"]
    assert glob["tuning_iter_budget"] == 500
    assert 10 <= glob["number_of_iterations"] <= 500
