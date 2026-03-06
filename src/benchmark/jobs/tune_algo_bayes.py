from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
from tqdm import tqdm

try:
    import optuna
except ImportError:
    optuna = None

try:
    from joblib import Parallel, delayed
    from joblib import parallel as joblib_parallel
except ImportError:
    Parallel = None
    delayed = None
    joblib_parallel = None

from src.environment import Environment
from src.PSO.Config import PSOConfig
from src.PSO.PSO import PSO
from src.benchmark.core.algo_profiles import (
    SCENARIO_ITERATION_BOUNDS,
    SCENARIO_TUNING_BUDGET,
    SCENARIO_WAYPOINT_BOUNDS,
    DEFAULT_RESET_NUMBER,
    apply_algo_flags,
    cast_hyperparameters,
    get_search_space,
)
from src.benchmark.core.common import extract_metrics, objective_cost, scenario_path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_LOG = logging.getLogger(__name__)

_ENV_CACHE: dict[int, Environment] = {}


@contextmanager
def _tqdm_joblib(tqdm_object: Any):
    if Parallel is None or joblib_parallel is None:
        yield tqdm_object
        return

    class _TqdmBatchCallback:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._wrapped = _original_callback(*args, **kwargs)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            tqdm_object.update(n=self._wrapped.batch_size)
            return self._wrapped(*args, **kwargs)

        def __getattr__(self, item: str) -> Any:
            return getattr(self._wrapped, item)

    _original_callback = joblib_parallel.BatchCompletionCallBack
    joblib_parallel.BatchCompletionCallBack = _TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib_parallel.BatchCompletionCallBack = _original_callback
        tqdm_object.close()


def _resolve_n_jobs(requested_n_jobs: int, num_tasks: int) -> int:
    if requested_n_jobs == 1:
        return 1
    cpu = max(1, int(os.cpu_count() or 1))
    if requested_n_jobs <= 0:
        requested_n_jobs = max(1, cpu - 1)
    return max(1, min(int(requested_n_jobs), cpu, max(1, num_tasks)))


def _load_search_space(algo: str, json_path: str | None) -> dict[str, tuple[float, float]]:
    if json_path is None:
        return get_search_space(algo)

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Search-space JSON must be an object mapping hyperparameter -> [low, high].")

    parsed: dict[str, tuple[float, float]] = {}
    for key, value in payload.items():
        if not isinstance(value, list | tuple) or len(value) != 2:
            raise ValueError(f"Invalid bounds for '{key}', expected [low, high].")
        low, high = float(value[0]), float(value[1])
        if low >= high:
            raise ValueError(f"Invalid bounds for '{key}': low must be < high.")
        parsed[str(key)] = (low, high)
    return parsed


def _load_base_config(scenario_id: int) -> dict[str, Any]:
    cfg = asdict(PSOConfig())
    best_params: dict[str, Any] | None = None
    best_path = ROOT / "src" / "benchmarking" / f"Ob_scenario{scenario_id}" / f"scenario{scenario_id}_best.json"
    if best_path.exists():
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        best_params = payload.get("best_params")
        if isinstance(best_params, dict):
            cfg.update(best_params)
    cfg["reset_waypoints"] = bool(scenario_id not in {0, 1})
    if not isinstance(best_params, dict) or "reset_number" not in best_params:
        cfg["reset_number"] = DEFAULT_RESET_NUMBER
    cfg["reset_number"] = max(1, int(cfg.get("reset_number", DEFAULT_RESET_NUMBER)))
    return cfg


def _get_env(scenario_id: int) -> Environment:
    sid = int(scenario_id)
    env = _ENV_CACHE.get(sid)
    if env is not None:
        return env
    env = Environment()
    env.from_file(str(scenario_path(sid)))
    _ENV_CACHE[sid] = env
    return env


def _build_grid_warmstart_points(
    search_space: dict[str, tuple[float, float]],
    max_points: int,
    focus_params: int,
) -> list[dict[str, float]]:
    if max_points <= 0 or not search_space:
        return []

    priority = [
        "number_of_particules",
        "number_of_iterations",
        "number_of_waypoints",
        "initial_temperature",
        "collision_weight",
        "acceptance_probability_decay",
        "reset_number",
        "inertia_weight",
        "best_position_acceleration",
        "global_best_position_acceleration",
        "pre_heat_target_acceptance_rate",
    ]
    mids = {name: float((bounds[0] + bounds[1]) / 2.0) for name, bounds in search_space.items()}
    points: list[dict[str, float]] = [dict(mids)]

    ordered_params = [p for p in priority if p in search_space]
    ordered_params.extend(p for p in search_space if p not in ordered_params)
    for param in islice(ordered_params, max(1, int(focus_params))):
        low, high = search_space[param]
        low_point = dict(mids)
        high_point = dict(mids)
        low_point[param] = float(low)
        high_point[param] = float(high)
        points.append(low_point)
        points.append(high_point)

    unique: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for point in points:
        key = tuple(sorted((k, float(v)) for k, v in point.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
        if len(unique) >= int(max_points):
            break
    return unique


def _objective_cost(
    metrics: dict[str, Any],
    collision_penalty: float,
    time_weight: float,
    non_collision_free_penalty: float,
) -> float:
    return objective_cost(
        fitness=float(metrics["fitness"]),
        collisions=int(metrics["collisions"]),
        elapsed=float(metrics["time_sec"]),
        collision_penalty=float(collision_penalty),
        non_collision_free_penalty=float(non_collision_free_penalty),
        time_weight=float(time_weight),
    )


def _aggregate_trial_target(
    costs: list[float],
    collision_free_flags: list[bool],
    collision_free_weight: float,
    no_feasible_penalty: float,
) -> tuple[float, float, float]:
    mean_cost = float(np.mean(costs))
    collision_free_rate = float(np.mean(collision_free_flags)) if collision_free_flags else 0.0
    target = -mean_cost + collision_free_weight * collision_free_rate
    if collision_free_rate == 0.0:
        target -= float(no_feasible_penalty)
    return float(target), mean_cost, collision_free_rate


def _sanitize_config_for_parallel(config: dict[str, Any], force_single_thread_fitness: bool) -> dict[str, Any]:
    cfg = dict(config)
    if force_single_thread_fitness:
        cfg["parallel_fitness_workers"] = 1
        cfg["reuse_fitness_thread_pool"] = True
    return cfg


def _run_once(
    scenario_id: int,
    config: dict[str, Any],
    seed: int,
    pso_iteration_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    env = _get_env(int(scenario_id))

    np.random.seed(int(seed))
    pso = PSO(env, config=config)

    t0 = time.perf_counter()
    pso.run(progress=False, verbose=False, iteration_callback=pso_iteration_callback)
    elapsed = float(time.perf_counter() - t0)

    metrics = extract_metrics(pso, env)
    metrics["time_sec"] = elapsed
    return metrics


def _estimate_auto_penalties_for_scenario(
    scenario_id: int,
    algo: str,
    eval_repeats: int,
    seed_base: int,
    default_collision_penalty: float,
    default_non_collision_free_penalty: float,
    default_collision_free_weight: float,
    default_no_feasible_penalty: float,
    time_weight: float,
    force_single_thread_fitness: bool,
) -> dict[str, float]:
    base_cfg = apply_algo_flags(_load_base_config(scenario_id), algo)
    base_cfg = _sanitize_config_for_parallel(base_cfg, force_single_thread_fitness)

    baseline_costs: list[float] = []
    samples = max(2, int(eval_repeats))
    for repeat_idx in range(samples):
        seed = int(seed_base + scenario_id * 100_000 + 900_000 + repeat_idx)
        metrics = _run_once(scenario_id=scenario_id, config=base_cfg, seed=seed)
        neutral_cost = float(metrics["fitness"]) + (float(time_weight) / (int(scenario_id) + 1)) * float(metrics["time_sec"])
        baseline_costs.append(neutral_cost)

    median_cost = float(np.median(baseline_costs)) if baseline_costs else 100.0
    scale = float(np.clip(median_cost / 100.0, 0.25, 20.0))
    return {
        "collision_penalty": float(default_collision_penalty) * scale,
        "non_collision_free_penalty": float(default_non_collision_free_penalty) * scale,
        "collision_free_weight": float(default_collision_free_weight) * scale,
        "no_feasible_penalty": float(default_no_feasible_penalty) * scale,
        "scale": scale,
        "baseline_median_cost": median_cost,
    }


def _run_grid_warmstart(
    optimizer: BayesianOptimization,
    objective: Callable[..., float],
    search_space: dict[str, tuple[float, float]],
    max_points: int,
    focus_params: int,
) -> int:
    points = _build_grid_warmstart_points(
        search_space=search_space,
        max_points=max_points,
        focus_params=focus_params,
    )
    executed = 0
    for params in points:
        target = float(objective(**params))
        optimizer.register(params=params, target=target)
        executed += 1
    return executed


def _maximize_with_backend(
    *,
    backend: str,
    objective: Callable[..., float],
    search_space: dict[str, tuple[float, float]],
    init_points: int,
    n_iter: int,
    random_seed: int,
    grid_warmstart_points: int,
    grid_focus_params: int,
    hpo_sampler: str = "tpe",
    enable_pruning: bool = False,
) -> tuple[dict[str, float], float, int, int, dict[str, float] | None]:
    if backend == "bayes_opt":
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=search_space,
            random_state=int(random_seed),
            verbose=0,
            allow_duplicate_points=True,
        )
        warm_count = _run_grid_warmstart(
            optimizer=optimizer,
            objective=objective,
            search_space=search_space,
            max_points=max(0, int(grid_warmstart_points)),
            focus_params=max(1, int(grid_focus_params)),
        )
        remaining_init = max(0, int(init_points) - int(warm_count))
        optimizer.maximize(init_points=remaining_init, n_iter=int(n_iter))
        best_raw = dict(optimizer.max["params"])
        best_target = float(optimizer.max["target"])
        total_trials = int(warm_count + remaining_init + int(n_iter))
        return best_raw, best_target, total_trials, int(warm_count), None

    if backend == "optuna":
        if optuna is None:
            raise ImportError("optuna is required for --hpo-backend optuna. Install with `uv pip install optuna`.")

        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except Exception:  # pragma: no cover
            pass

        # --- Sampler selection (HPO-B) ---
        if hpo_sampler == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(
                seed=int(random_seed),
                n_startup_trials=max(1, int(init_points)),
                warn_independent_sampling=False,
            )
        else:  # default: tpe
            sampler = optuna.samplers.TPESampler(
                seed=int(random_seed),
                n_startup_trials=max(1, int(init_points)),
            )

        # --- Pruner (HPO-A) ---
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
            if enable_pruning
            else optuna.pruners.NopPruner()
        )

        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner,
        )

        warm_points = _build_grid_warmstart_points(
            search_space=search_space,
            max_points=max(0, int(grid_warmstart_points)),
            focus_params=max(1, int(grid_focus_params)),
        )
        for params in warm_points:
            study.enqueue_trial({k: float(v) for k, v in params.items()})

        def _optuna_objective(trial: "optuna.Trial") -> float:
            params = {
                key: trial.suggest_float(key, float(bounds[0]), float(bounds[1]))
                for key, bounds in search_space.items()
            }
            return float(objective(_trial=trial, **params))

        total_trials = max(1, int(init_points) + int(n_iter))
        study.optimize(_optuna_objective, n_trials=total_trials, show_progress_bar=False)

        best_raw = {str(k): float(v) for k, v in study.best_params.items()}
        best_target = float(study.best_value)

        # --- Parameter importance (HPO-D) ---
        param_importances: dict[str, float] | None = None
        try:
            raw_imp = optuna.importance.get_param_importances(study)
            param_importances = {str(k): float(v) for k, v in raw_imp.items()}
        except Exception:
            pass

        return best_raw, best_target, int(len(study.trials)), int(len(warm_points)), param_importances

    raise ValueError(f"Unknown HPO backend '{backend}'. Use 'bayes_opt' or 'optuna'.")


def _find_stagnation_iter(
    rows_df: "pd.DataFrame",
    min_iters: int,
    max_iters: int,
    window: int,
    threshold: float,
) -> int:
    """Return the PSO iteration at which the collision-free best_fitness stagnates.

    Groups CF rows by pso_iteration, takes the median best_fitness across all
    trials, then locates the first window of ``window`` consecutive steps where
    the cumulative improvement since the start of that window is below
    ``threshold``.  The result is clipped to [min_iters, max_iters].

    Returns ``max_iters`` when there is not enough data to make a decision.
    """
    if rows_df is None or rows_df.empty:
        return max_iters

    if "is_collision_free" in rows_df.columns:
        cf = rows_df[rows_df["is_collision_free"]]
    else:
        cf = rows_df

    if cf.empty:
        return max_iters

    curve = (
        cf.groupby("pso_iteration")["best_fitness"]
        .median()
        .sort_index()
        .cummin()  # enforce monotone: as new runs join the CF set at later
                   # iterations their higher fitness would otherwise push the
                   # median up, creating false "stagnation" signals.
    )

    if len(curve) < window + 1:
        return max_iters

    # Skip the initial "warmup" plateau. When only a handful of runs have
    # achieved CF at early iterations their median best_fitness is an
    # outlier floor; the cummin stays flat there until a representative
    # population of CF runs has accumulated. Detecting stagnation on that
    # flat region is meaningless — it always fires immediately.
    initial_value = float(curve.iloc[0])
    past_warmup = curve[curve < initial_value - threshold]
    if past_warmup.empty:
        # The curve never improved meaningfully — cannot determine stagnation.
        return max_iters
    curve = curve.loc[past_warmup.index[0]:]

    if len(curve) < window + 1:
        return max_iters

    # Guard against a second immediate flat region (e.g. when CF data is very
    # sparse and each improvement is a single outlier run, so the curve is a
    # step-function with instant drops followed by long plateaus). If the
    # curve shows no further improvement in the first `window` steps after the
    # warmup there is not enough data to infer a reliable stagnation point.
    if float(curve.iloc[window]) >= float(curve.iloc[0]) - threshold:
        return max_iters

    # best_fitness is a *cost* (lower = better).  curve.diff(window) gives
    # negative values while the run is still converging (cost is dropping).
    # Stagnation occurs when the cost drop over the window shrinks below
    # threshold in absolute terms, i.e. diff > -threshold.
    improvements = curve.diff(window).dropna()
    stagnant = improvements[improvements > -threshold]

    if stagnant.empty:
        return max_iters

    pruned = int(stagnant.index[0])

    # Sanity-check: if stagnation fired fewer than 2*window iterations after
    # the warmup end the "active convergence" phase was too short (only one
    # decline window then immediately flat — typical of step-function curves
    # produced by very few CF runs).  Treat as insufficient data.
    if pruned - int(curve.index[0]) < 2 * window:
        return max_iters

    return int(np.clip(pruned, min_iters, max_iters))


def _prune_iterations_in_summary(
    summary_path: "Path",
    pso_df: "pd.DataFrame",
    stagnation_window: int,
    stagnation_threshold: float,
) -> None:
    """Rewrite ``tuning_summary.json`` with pruned ``number_of_iterations``.

    For every per-scenario entry the fixed tuning budget stored in
    ``best_params["number_of_iterations"]`` is moved to
    ``best_params["tuning_iter_budget"]`` and replaced with the stagnation-
    pruned value derived from the PSO iteration curves collected during tuning.

    The global ``best_params_by_scenario`` entries are updated the same way.
    """
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    modified = False

    # --- per-scenario entries ---
    for entry in summary.get("per_scenario", []):
        scenario_id = int(entry.get("scenario", -1))
        if scenario_id < 0:
            continue

        lo, hi = SCENARIO_ITERATION_BOUNDS.get(scenario_id, (10.0, 1200.0))
        min_iters, max_iters = int(lo), int(hi)

        mask = (
            (pso_df["scope"] == "scenario")
            & (pso_df["scenario"] == scenario_id)
        )
        pruned = _find_stagnation_iter(
            pso_df[mask],
            min_iters=min_iters,
            max_iters=max_iters,
            window=stagnation_window,
            threshold=stagnation_threshold,
        )

        best_params = entry.get("best_params") or {}
        budget = int(best_params.get("number_of_iterations", SCENARIO_TUNING_BUDGET.get(scenario_id, max_iters)))
        entry["best_params"] = best_params
        best_params["tuning_iter_budget"] = budget
        best_params["number_of_iterations"] = pruned
        modified = True
        _LOG.info(
            "Scenario %d: number_of_iterations %d → %d (stagnation window=%d, threshold=%.4f)",
            scenario_id, budget, pruned, stagnation_window, stagnation_threshold,
        )

    # --- global best_params_by_scenario ---
    global_entry = summary.get("global")
    if global_entry and "best_params_by_scenario" in global_entry:
        for sid_str, params in global_entry["best_params_by_scenario"].items():
            sid = int(sid_str)
            lo, hi = SCENARIO_ITERATION_BOUNDS.get(sid, (10.0, 1200.0))
            min_iters, max_iters = int(lo), int(hi)

            mask = (pso_df["scope"] == "global") & (pso_df["scenario"] == sid)
            pruned = _find_stagnation_iter(
                pso_df[mask],
                min_iters=min_iters,
                max_iters=max_iters,
                window=stagnation_window,
                threshold=stagnation_threshold,
            )

            budget = int(params.get("number_of_iterations", SCENARIO_TUNING_BUDGET.get(sid, max_iters)))
            params["tuning_iter_budget"] = budget
            params["number_of_iterations"] = pruned
            modified = True

    if modified:
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        _LOG.info("Pruned number_of_iterations written back to %s", summary_path)


def _optimize_for_scenario(
    scenario_id: int,
    algo: str,
    search_space: dict[str, tuple[float, float]],
    init_points: int,
    n_iter: int,
    eval_repeats: int,
    seed_base: int,
    penalty_cfg: dict[str, float],
    time_weight: float,
    grid_warmstart_points: int,
    grid_focus_params: int,
    force_single_thread_fitness: bool,
    hpo_backend: str,
    hpo_sampler: str = "tpe",
    enable_pruning: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_cfg = _load_base_config(scenario_id)
    # Use a generous fixed iteration budget; actual number_of_iterations is
    # determined post-tuning via stagnation pruning.
    if scenario_id in SCENARIO_TUNING_BUDGET:
        base_cfg["number_of_iterations"] = int(SCENARIO_TUNING_BUDGET[scenario_id])
    pso_rows: list[dict[str, Any]] = []
    trial_counter = {"value": 0}
    best_target_so_far = {"value": -np.inf}

    def objective(_trial=None, **raw_params: float) -> float:
        trial_idx = trial_counter["value"]
        trial_counter["value"] += 1

        tuned_cfg = cast_hyperparameters(raw_params, base_cfg)
        tuned_cfg = apply_algo_flags(tuned_cfg, algo)
        tuned_cfg = _sanitize_config_for_parallel(tuned_cfg, force_single_thread_fitness)

        costs: list[float] = []
        collision_free_flags: list[bool] = []
        repeat_targets: list[float] = []

        for repeat_idx in range(eval_repeats):
            seed = int(seed_base + scenario_id * 100_000 + repeat_idx)

            def _on_iter(payload: dict[str, Any]) -> None:
                pso_rows.append(
                    {
                        "scope": "scenario",
                        "scope_id": str(scenario_id),
                        "algo": algo,
                        "scenario": int(scenario_id),
                        "trial_index": int(trial_idx),
                        "repeat_index": int(repeat_idx),
                        "pso_iteration": int(payload.get("iteration", -1)),
                        "best_fitness": float(payload.get("best_fitness", np.nan)),
                        "is_collision_free": bool(payload.get("is_collision_free", False)),
                    }
                )

            metrics = _run_once(
                scenario_id=scenario_id,
                config=tuned_cfg,
                seed=seed,
                pso_iteration_callback=_on_iter,
            )
            cost = _objective_cost(
                metrics,
                collision_penalty=float(penalty_cfg["collision_penalty"]),
                time_weight=float(time_weight) / (int(scenario_id) + 1),
                non_collision_free_penalty=float(penalty_cfg["non_collision_free_penalty"]),
            )
            is_collision_free = bool(int(metrics["collisions"]) == 0)
            target = -float(cost)
            costs.append(float(cost))
            collision_free_flags.append(is_collision_free)
            repeat_targets.append(target)

            # Trial pruning: report intermediate cost and check if Optuna
            # wants to prune this trial early (HPO-A).
            if _trial is not None:
                try:
                    _trial.report(float(cost), step=int(repeat_idx))
                    if _trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                except optuna.exceptions.TrialPruned:
                    raise
                except Exception:
                    pass

        aggregate_target, mean_cost, collision_free_rate = _aggregate_trial_target(
            costs=costs,
            collision_free_flags=collision_free_flags,
            collision_free_weight=float(penalty_cfg["collision_free_weight"]),
            no_feasible_penalty=float(penalty_cfg["no_feasible_penalty"]),
        )
        best_target_so_far["value"] = max(best_target_so_far["value"], float(aggregate_target))

        return float(aggregate_target)

    best_raw, best_target, total_trials, warm_count, param_importances = _maximize_with_backend(
        backend=hpo_backend,
        objective=objective,
        search_space=search_space,
        init_points=int(init_points),
        n_iter=int(n_iter),
        random_seed=int(seed_base + scenario_id),
        grid_warmstart_points=int(grid_warmstart_points),
        grid_focus_params=int(grid_focus_params),
        hpo_sampler=str(hpo_sampler),
        enable_pruning=bool(enable_pruning),
    )

    best_cfg = apply_algo_flags(cast_hyperparameters(best_raw, base_cfg), algo)
    best_cfg = _sanitize_config_for_parallel(best_cfg, force_single_thread_fitness)

    summary = {
        "scenario": int(scenario_id),
        "algo": algo,
        "best_target": float(best_target),
        "best_params_raw": best_raw,
        "best_params": best_cfg,
        "total_trials": int(max(total_trials, trial_counter["value"])),
        "grid_warmstart_trials": int(warm_count),
        "penalties": {k: float(v) for k, v in penalty_cfg.items()},
        "hpo_backend": str(hpo_backend),
        "hpo_sampler": str(hpo_sampler),
        "param_importances": param_importances,
    }
    return pso_rows, summary


def _optimize_global(
    scenarios: list[int],
    algo: str,
    search_space: dict[str, tuple[float, float]],
    init_points: int,
    n_iter: int,
    eval_repeats: int,
    seed_base: int,
    penalty_by_scenario: dict[int, dict[str, float]],
    time_weight: float,
    grid_warmstart_points: int,
    grid_focus_params: int,
    force_single_thread_fitness: bool,
    hpo_backend: str,
    hpo_sampler: str = "tpe",
    enable_pruning: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_cfg_by_scenario = {sid: _load_base_config(sid) for sid in scenarios}
    # Override number_of_iterations with fixed tuning budget per scenario.
    for _sid in scenarios:
        if _sid in SCENARIO_TUNING_BUDGET:
            base_cfg_by_scenario[_sid] = dict(base_cfg_by_scenario[_sid])
            base_cfg_by_scenario[_sid]["number_of_iterations"] = int(SCENARIO_TUNING_BUDGET[_sid])
    pso_rows: list[dict[str, Any]] = []
    trial_counter = {"value": 0}
    best_target_so_far = {"value": -np.inf}

    def objective(_trial=None, **raw_params: float) -> float:
        trial_idx = trial_counter["value"]
        trial_counter["value"] += 1

        costs: list[float] = []
        collision_free_flags: list[bool] = []
        repeat_targets: list[float] = []

        for scenario_id in scenarios:
            base_cfg = base_cfg_by_scenario[scenario_id]
            tuned_cfg = apply_algo_flags(cast_hyperparameters(raw_params, base_cfg), algo)
            tuned_cfg = _sanitize_config_for_parallel(tuned_cfg, force_single_thread_fitness)
            penalty_cfg = penalty_by_scenario[int(scenario_id)]

            for repeat_idx in range(eval_repeats):
                seed = int(seed_base + scenario_id * 100_000 + repeat_idx)

                def _on_iter(payload: dict[str, Any]) -> None:
                    pso_rows.append(
                        {
                            "scope": "global",
                            "scope_id": "all",
                            "algo": algo,
                            "scenario": int(scenario_id),
                            "trial_index": int(trial_idx),
                            "repeat_index": int(repeat_idx),
                            "pso_iteration": int(payload.get("iteration", -1)),
                            "best_fitness": float(payload.get("best_fitness", np.nan)),
                            "is_collision_free": bool(payload.get("is_collision_free", False)),
                        }
                    )

                metrics = _run_once(
                    scenario_id=scenario_id,
                    config=tuned_cfg,
                    seed=seed,
                    pso_iteration_callback=_on_iter,
                )

                cost = _objective_cost(
                    metrics,
                    collision_penalty=float(penalty_cfg["collision_penalty"]),
                    time_weight=float(time_weight),
                    non_collision_free_penalty=float(penalty_cfg["non_collision_free_penalty"]),
                )
                is_collision_free = bool(int(metrics["collisions"]) == 0)
                target = -float(cost)
                costs.append(float(cost))
                collision_free_flags.append(is_collision_free)
                repeat_targets.append(target)

                # Trial pruning (HPO-A): use a running step counter across
                # scenario × repeat pairs within this trial.
                _global_step = len(costs) - 1
                if _trial is not None:
                    try:
                        _trial.report(float(cost), step=int(_global_step))
                        if _trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                    except optuna.exceptions.TrialPruned:
                        raise
                    except Exception:
                        pass

        penalty_mean = {
            key: float(np.mean([penalty_by_scenario[sid][key] for sid in scenarios]))
            for key in [
                "collision_penalty",
                "non_collision_free_penalty",
                "collision_free_weight",
                "no_feasible_penalty",
            ]
        }
        aggregate_target, mean_cost, collision_free_rate = _aggregate_trial_target(
            costs=costs,
            collision_free_flags=collision_free_flags,
            collision_free_weight=penalty_mean["collision_free_weight"],
            no_feasible_penalty=penalty_mean["no_feasible_penalty"],
        )
        best_target_so_far["value"] = max(best_target_so_far["value"], float(aggregate_target))
        return float(aggregate_target)

    best_raw, best_target, total_trials, warm_count, param_importances = _maximize_with_backend(
        backend=hpo_backend,
        objective=objective,
        search_space=search_space,
        init_points=int(init_points),
        n_iter=int(n_iter),
        random_seed=int(seed_base + 10_000),
        grid_warmstart_points=int(grid_warmstart_points),
        grid_focus_params=int(grid_focus_params),
        hpo_sampler=str(hpo_sampler),
        enable_pruning=bool(enable_pruning),
    )

    summary = {
        "algo": algo,
        "best_target": float(best_target),
        "best_params_raw": best_raw,
        "best_params_by_scenario": {
            str(sid): _sanitize_config_for_parallel(
                apply_algo_flags(cast_hyperparameters(best_raw, _load_base_config(sid)), algo),
                force_single_thread_fitness,
            )
            for sid in scenarios
        },
        "total_trials": int(max(total_trials, trial_counter["value"])),
        "grid_warmstart_trials": int(warm_count),
        "hpo_backend": str(hpo_backend),
        "hpo_sampler": str(hpo_sampler),
        "param_importances": param_importances,
    }
    return pso_rows, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bayesian tuning for a selected PSO algorithm profile.")
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--search-space-json", type=str, default=None)
    parser.add_argument("--init-points", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=80)
    parser.add_argument("--eval-repeats", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--collision-penalty", type=float, default=50.0)
    parser.add_argument("--non-collision-free-penalty", type=float, default=200.0)
    parser.add_argument("--collision-free-weight", type=float, default=500.0)
    parser.add_argument("--no-feasible-penalty", type=float, default=150.0)
    parser.add_argument("--time-weight", type=float, default=5.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--grid-warmstart-points", type=int, default=9)
    parser.add_argument("--grid-focus-params", type=int, default=4)
    parser.add_argument("--hpo-backend", type=str, choices=["bayes_opt", "optuna"], default="optuna")
    parser.add_argument("--disable-auto-penalties", action="store_true")
    parser.add_argument("--penalty-calibration-runs", type=int, default=2)
    parser.add_argument(
        "--hpo-sampler", "--hpo_sampler", dest="hpo_sampler",
        type=str, choices=["tpe", "cmaes"], default="tpe",
        help="Optuna sampler backend. 'cmaes' uses CmaEsSampler (HPO-B).",
    )
    parser.add_argument(
        "--enable-pruning", "--enable_pruning", dest="enable_pruning",
        action="store_true",
        help="Enable Optuna MedianPruner to cut unpromising trials early (HPO-A).",
    )
    parser.add_argument(
        "--stagnation-window", "--stagnation_window", dest="stagnation_window",
        type=int, default=25,
        help="Rolling window (in PSO iterations) used for post-tuning stagnation detection.",
    )
    parser.add_argument(
        "--stagnation-threshold", "--stagnation_threshold", dest="stagnation_threshold",
        type=float, default=0.01,
        help="Minimum improvement over the stagnation window to keep running; below this the run is considered converged.",
    )
    parser.add_argument("--skip-global", "--skip_global", dest="skip_global", action="store_true")
    parser.add_argument(
        "--skip-per-scenario",
        "--skip_per_scenario",
        dest="skip_per_scenario",
        action="store_true",
    )
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--out-dir", type=str, default="artifacts/exp/tuning")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # when optuna is chosen as the backend make sure its own logger stays quiet
    # regardless of the global log level; this prevents the per‑trial "Trial…"
    # messages that users asked us to stop printing.
    if args.hpo_backend == "optuna" and optuna is not None:
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except Exception:  # pragma: no cover
            pass

    scenarios = sorted(set(int(s) for s in args.scenarios))
    search_space = _load_search_space(args.algo, args.search_space_json)

    n_jobs = _resolve_n_jobs(int(args.n_jobs), num_tasks=len(scenarios))
    force_single_thread_fitness = n_jobs != 1

    if n_jobs != 1 and (Parallel is None or delayed is None):
        raise ImportError("joblib is required for n_jobs != 1. Install with `uv pip install joblib`.")

    penalty_by_scenario: dict[int, dict[str, float]] = {}
    for scenario_id in scenarios:
        if args.disable_auto_penalties:
            penalty_by_scenario[int(scenario_id)] = {
                "collision_penalty": float(args.collision_penalty),
                "non_collision_free_penalty": float(args.non_collision_free_penalty),
                "collision_free_weight": float(args.collision_free_weight),
                "no_feasible_penalty": float(args.no_feasible_penalty),
                "scale": 1.0,
                "baseline_median_cost": np.nan,
            }
        else:
            penalty_by_scenario[int(scenario_id)] = _estimate_auto_penalties_for_scenario(
                scenario_id=int(scenario_id),
                algo=str(args.algo),
                eval_repeats=max(2, int(args.penalty_calibration_runs)),
                seed_base=int(args.seed_base),
                default_collision_penalty=float(args.collision_penalty),
                default_non_collision_free_penalty=float(args.non_collision_free_penalty),
                default_collision_free_weight=float(args.collision_free_weight),
                default_no_feasible_penalty=float(args.no_feasible_penalty),
                time_weight=float(args.time_weight),
                force_single_thread_fitness=force_single_thread_fitness,
            )

    def _search_space_for_scenario(base_space: dict[str, tuple[float, float]], sid: int) -> dict[str, tuple[float, float]]:
        space = dict(base_space)
        if sid in SCENARIO_WAYPOINT_BOUNDS:
            space["number_of_waypoints"] = tuple(SCENARIO_WAYPOINT_BOUNDS[sid])
        # number_of_iterations is NOT tuned; a fixed budget (SCENARIO_TUNING_BUDGET)
        # is injected into base_cfg inside the optimizer instead.
        return space

    def _search_space_for_global(base_space: dict[str, tuple[float, float]], scenario_list: list[int]) -> dict[str, tuple[float, float]]:
        space = dict(base_space)
        wp_lows: list[float] = []
        wp_highs: list[float] = []
        for sid in scenario_list:
            if sid in SCENARIO_WAYPOINT_BOUNDS:
                low, high = SCENARIO_WAYPOINT_BOUNDS[sid]
                wp_lows.append(float(low))
                wp_highs.append(float(high))
        if wp_lows and wp_highs:
            space["number_of_waypoints"] = (min(wp_lows), max(wp_highs))
        # number_of_iterations is NOT tuned; fixed budgets are injected per-scenario
        # inside _optimize_global.
        return space

    pso_rows: list[dict[str, Any]] = []
    per_scenario_summaries: list[dict[str, Any]] = []
    global_summary: dict[str, Any] | None = None

    if not args.skip_per_scenario:
        if n_jobs == 1:
            for scenario_id in tqdm(scenarios, desc="Tuning per scenario"):
                pso_curve_rows, summary = _optimize_for_scenario(
                    scenario_id=scenario_id,
                    algo=args.algo,
                    search_space=_search_space_for_scenario(search_space, scenario_id),
                    init_points=int(args.init_points),
                    n_iter=int(args.n_iter),
                    eval_repeats=int(args.eval_repeats),
                    seed_base=int(args.seed_base),
                    penalty_cfg=penalty_by_scenario[int(scenario_id)],
                    time_weight=float(args.time_weight),
                    grid_warmstart_points=int(args.grid_warmstart_points),
                    grid_focus_params=int(args.grid_focus_params),
                    force_single_thread_fitness=force_single_thread_fitness,
                    hpo_backend=str(args.hpo_backend),
                    hpo_sampler=str(args.hpo_sampler),
                    enable_pruning=bool(args.enable_pruning),
                )
                pso_rows.extend(pso_curve_rows)
                per_scenario_summaries.append(summary)
        else:
            _LOG.info("Running per-scenario tuning in parallel with n_jobs=%d", n_jobs)
            with tqdm(total=len(scenarios), desc="Tuning per scenario (parallel)") as pbar:
                with _tqdm_joblib(pbar):
                    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                        delayed(_optimize_for_scenario)(
                            scenario_id=scenario_id,
                            algo=args.algo,
                            search_space=_search_space_for_scenario(search_space, scenario_id),
                            init_points=int(args.init_points),
                            n_iter=int(args.n_iter),
                            eval_repeats=int(args.eval_repeats),
                            seed_base=int(args.seed_base),
                            penalty_cfg=penalty_by_scenario[int(scenario_id)],
                            time_weight=float(args.time_weight),
                            grid_warmstart_points=int(args.grid_warmstart_points),
                            grid_focus_params=int(args.grid_focus_params),
                            force_single_thread_fitness=force_single_thread_fitness,
                            hpo_backend=str(args.hpo_backend),
                            hpo_sampler=str(args.hpo_sampler),
                            enable_pruning=bool(args.enable_pruning),
                        )
                        for scenario_id in scenarios
                    )
            for pso_curve_rows, summary in results:
                pso_rows.extend(pso_curve_rows)
                per_scenario_summaries.append(summary)

    if not args.skip_global:
        with tqdm(total=1, desc="Global tuning") as pbar:
            pso_curve_rows, summary = _optimize_global(
                scenarios=scenarios,
                algo=args.algo,
                search_space=_search_space_for_global(search_space, scenarios),
                init_points=int(args.init_points),
                n_iter=int(args.n_iter),
                eval_repeats=int(args.eval_repeats),
                seed_base=int(args.seed_base),
                penalty_by_scenario=penalty_by_scenario,
                time_weight=float(args.time_weight),
                grid_warmstart_points=int(args.grid_warmstart_points),
                grid_focus_params=int(args.grid_focus_params),
                force_single_thread_fitness=force_single_thread_fitness,
                hpo_backend=str(args.hpo_backend),
                hpo_sampler=str(args.hpo_sampler),
                enable_pruning=bool(args.enable_pruning),
            )
            pso_rows.extend(pso_curve_rows)
            global_summary = summary
            pbar.update(1)

    pso_df = pd.DataFrame(pso_rows)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_out = out_dir / "tuning_summary.json"
    pso_out = out_dir / "tuning_pso_curves.parquet"

    if not pso_df.empty:
        pso_df.to_parquet(pso_out, index=False)

    with summary_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "algo": args.algo,
                "scenarios": scenarios,
                "search_space": {k: [float(v[0]), float(v[1])] for k, v in search_space.items()},
                "global": global_summary,
                "per_scenario": per_scenario_summaries,
                "n_jobs": int(n_jobs),
                "force_single_thread_fitness": bool(force_single_thread_fitness),
                "auto_penalties": [
                    {
                        "scenario": int(sid),
                        **{k: (float(v) if isinstance(v, float | int | np.floating) and not np.isnan(v) else None) for k, v in vals.items()},
                    }
                    for sid, vals in sorted(penalty_by_scenario.items())
                ],
                "grid_warmstart_points": int(args.grid_warmstart_points),
                "grid_focus_params": int(args.grid_focus_params),
                "hpo_backend": str(args.hpo_backend),
                "hpo_sampler": str(args.hpo_sampler),
                "enable_pruning": bool(args.enable_pruning),
            },
            handle,
            indent=2,
        )

    _LOG.info("=== Bayesian tuning complete ===")
    _LOG.info("Summary JSON: %s", summary_out)
    if not pso_df.empty:
        _LOG.info("PSO curves parquet: %s", pso_out)

    # Post-tuning: determine pruned number_of_iterations via stagnation detection.
    if not pso_df.empty:
        _prune_iterations_in_summary(
            summary_path=summary_out,
            pso_df=pso_df,
            stagnation_window=int(args.stagnation_window),
            stagnation_threshold=float(args.stagnation_threshold),
        )


if __name__ == "__main__":
    main()
