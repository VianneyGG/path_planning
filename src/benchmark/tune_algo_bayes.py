from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
from tqdm import tqdm

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
from src.benchmark.algo_profiles import (
    apply_algo_flags,
    cast_hyperparameters,
    get_search_space,
)
from src.benchmark.algo_profiles import SCENARIO_WAYPOINT_BOUNDS
from src.benchmark.pso_dimVtempVvanilla import _extract_metrics, _scenario_path


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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_search_space(algo: str, json_path: str | None) -> dict[str, tuple[float, float]]:
    if json_path is None:
        return get_search_space(algo)

    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    best_path = ROOT / "src" / "benchmarking" / f"Ob_scenario{scenario_id}" / f"scenario{scenario_id}_best.json"
    if best_path.exists():
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        best_params = payload.get("best_params")
        if isinstance(best_params, dict):
            cfg.update(best_params)

    cfg["reset_waypoints"] = bool(scenario_id not in {0, 1})
    return cfg


def _run_once(
    scenario_id: int,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    env = Environment()
    env.from_file(str(_scenario_path(scenario_id)))

    np.random.seed(int(seed))
    pso = PSO(env, config=config)

    t0 = time.perf_counter()
    pso.run(progress=False, verbose=False)
    elapsed = float(time.perf_counter() - t0)

    metrics = _extract_metrics(pso, env)
    metrics["time_sec"] = elapsed
    return metrics


def _objective_cost(metrics: dict[str, Any], collision_penalty: float, time_weight: float) -> float:
    fitness = float(metrics["fitness"])
    collisions = int(metrics["collisions"])
    elapsed = float(metrics["time_sec"])
    return fitness + collision_penalty * collisions + time_weight * elapsed


def _optimize_for_scenario(
    scenario_id: int,
    algo: str,
    search_space: dict[str, tuple[float, float]],
    init_points: int,
    n_iter: int,
    eval_repeats: int,
    seed_base: int,
    collision_penalty: float,
    time_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_cfg = _load_base_config(scenario_id)
    trial_rows: list[dict[str, Any]] = []
    trial_counter = {"value": 0}

    def objective(**raw_params: float) -> float:
        trial_idx = trial_counter["value"]
        trial_counter["value"] += 1

        tuned_cfg = cast_hyperparameters(raw_params, base_cfg)
        tuned_cfg = apply_algo_flags(tuned_cfg, algo)

        scores: list[float] = []
        for repeat_idx in range(eval_repeats):
            seed = int(seed_base + scenario_id * 100_000 + trial_idx * 100 + repeat_idx)
            metrics = _run_once(scenario_id=scenario_id, config=tuned_cfg, seed=seed)
            cost = _objective_cost(metrics, collision_penalty=collision_penalty, time_weight=time_weight/(int(scenario_id)+1))
            target = -float(cost)
            scores.append(target)

            row = {
                "scope": "scenario",
                "scope_id": str(scenario_id),
                "algo": algo,
                "scenario": int(scenario_id),
                "trial_index": int(trial_idx),
                "repeat_index": int(repeat_idx),
                "seed": int(seed),
                "target": float(target),
                "fitness": float(metrics["fitness"]),
                "length": float(metrics["length"]),
                "smoothness": float(metrics["smoothness"]),
                "collisions": int(metrics["collisions"]),
                "corners": int(metrics["corners"]),
                "time_sec": float(metrics["time_sec"]),
                "is_collision_free": bool(int(metrics["collisions"]) == 0),
                **tuned_cfg,
            }
            trial_rows.append(row)

        return float(np.mean(scores))

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=search_space,
        random_state=seed_base + scenario_id,
        verbose=0,
        allow_duplicate_points=True,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_raw = dict(optimizer.max["params"])
    best_cfg = apply_algo_flags(cast_hyperparameters(best_raw, base_cfg), algo)

    summary = {
        "scenario": int(scenario_id),
        "algo": algo,
        "best_target": float(optimizer.max["target"]),
        "best_params_raw": best_raw,
        "best_params": best_cfg,
        "total_trials": int(trial_counter["value"]),
    }
    return trial_rows, summary


def _optimize_global(
    scenarios: list[int],
    algo: str,
    search_space: dict[str, tuple[float, float]],
    init_points: int,
    n_iter: int,
    eval_repeats: int,
    seed_base: int,
    collision_penalty: float,
    time_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_cfg_by_scenario = {sid: _load_base_config(sid) for sid in scenarios}
    trial_rows: list[dict[str, Any]] = []
    trial_counter = {"value": 0}

    def objective(**raw_params: float) -> float:
        trial_idx = trial_counter["value"]
        trial_counter["value"] += 1

        targets: list[float] = []
        for scenario_id in scenarios:
            base_cfg = base_cfg_by_scenario[scenario_id]
            tuned_cfg = apply_algo_flags(cast_hyperparameters(raw_params, base_cfg), algo)

            for repeat_idx in range(eval_repeats):
                seed = int(seed_base + scenario_id * 100_000 + trial_idx * 100 + repeat_idx)
                metrics = _run_once(scenario_id=scenario_id, config=tuned_cfg, seed=seed)
                cost = _objective_cost(metrics, collision_penalty=collision_penalty, time_weight=time_weight)
                target = -float(cost)
                targets.append(target)

                row = {
                    "scope": "global",
                    "scope_id": "all",
                    "algo": algo,
                    "scenario": int(scenario_id),
                    "trial_index": int(trial_idx),
                    "repeat_index": int(repeat_idx),
                    "seed": int(seed),
                    "target": float(target),
                    "fitness": float(metrics["fitness"]),
                    "length": float(metrics["length"]),
                    "smoothness": float(metrics["smoothness"]),
                    "collisions": int(metrics["collisions"]),
                    "corners": int(metrics["corners"]),
                    "time_sec": float(metrics["time_sec"]),
                    "is_collision_free": bool(int(metrics["collisions"]) == 0),
                    **tuned_cfg,
                }
                trial_rows.append(row)

        return float(np.mean(targets))

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=search_space,
        random_state=seed_base + 10_000,
        verbose=0,
        allow_duplicate_points=True,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_raw = dict(optimizer.max["params"])
    summary = {
        "algo": algo,
        "best_target": float(optimizer.max["target"]),
        "best_params_raw": best_raw,
        "best_params_by_scenario": {
            str(sid): apply_algo_flags(cast_hyperparameters(best_raw, _load_base_config(sid)), algo)
            for sid in scenarios
        },
        "total_trials": int(trial_counter["value"]),
    }
    return trial_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian tuning for a selected PSO algorithm profile.")
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--search-space-json", type=str, default=None)
    parser.add_argument("--init-points", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=80)
    parser.add_argument("--eval-repeats", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--collision-penalty", type=float, default=50.0)
    parser.add_argument("--time-weight", type=float, default=5.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--skip-global", "--skip_global", dest="skip_global", action="store_true")
    parser.add_argument(
        "--skip-per-scenario",
        "--skip_per_scenario",
        dest="skip_per_scenario",
        action="store_true",
    )
    parser.add_argument("--out-dir", type=str, default="artifacts/exp/tuning")
    args = parser.parse_args()

    scenarios = sorted(set(int(s) for s in args.scenarios))
    search_space = _load_search_space(args.algo, args.search_space_json)

    def _search_space_for_scenario(base_space: dict[str, tuple[float, float]], sid: int) -> dict[str, tuple[float, float]]:
        space = dict(base_space)
        if sid in SCENARIO_WAYPOINT_BOUNDS:
            space["number_of_waypoints"] = tuple(SCENARIO_WAYPOINT_BOUNDS[sid])
        return space

    def _search_space_for_global(base_space: dict[str, tuple[float, float]], scenario_list: list[int]) -> dict[str, tuple[float, float]]:
        space = dict(base_space)
        # merge waypoint bounds across selected scenarios (min low, max high)
        lows: list[float] = []
        highs: list[float] = []
        for sid in scenario_list:
            if sid in SCENARIO_WAYPOINT_BOUNDS:
                low, high = SCENARIO_WAYPOINT_BOUNDS[sid]
                lows.append(float(low))
                highs.append(float(high))
        if lows and highs:
            space["number_of_waypoints"] = (min(lows), max(highs))
        return space

    if int(args.n_jobs) != 1 and (Parallel is None or delayed is None):
        raise ImportError("joblib is required for n_jobs != 1. Install with `uv pip install joblib`.")


    all_rows: list[dict[str, Any]] = []
    per_scenario_summaries: list[dict[str, Any]] = []
    global_summary: dict[str, Any] | None = None

    # Per-scenario tuning (parallelizable across scenarios)
    if not args.skip_per_scenario:
        if int(args.n_jobs) == 1:
            for scenario_id in tqdm(scenarios, desc="Tuning per scenario"):
                rows, summary = _optimize_for_scenario(
                    scenario_id=scenario_id,
                    algo=args.algo,
                    search_space=_search_space_for_scenario(search_space, scenario_id),
                    init_points=int(args.init_points),
                    n_iter=int(args.n_iter),
                    eval_repeats=int(args.eval_repeats),
                    seed_base=int(args.seed_base),
                    collision_penalty=float(args.collision_penalty),
                    time_weight=float(args.time_weight),
                )
                all_rows.extend(rows)
                per_scenario_summaries.append(summary)
        else:
            print(f"Running per-scenario tuning in parallel with n_jobs={int(args.n_jobs)}")
            with tqdm(total=len(scenarios), desc="Tuning per scenario (parallel)") as pbar:
                with _tqdm_joblib(pbar):
                    results = Parallel(n_jobs=int(args.n_jobs), backend="loky", verbose=0)(
                        delayed(_optimize_for_scenario)(
                            scenario_id=scenario_id,
                            algo=args.algo,
                            search_space=_search_space_for_scenario(search_space, scenario_id),
                            init_points=int(args.init_points),
                            n_iter=int(args.n_iter),
                            eval_repeats=int(args.eval_repeats),
                            seed_base=int(args.seed_base),
                            collision_penalty=float(args.collision_penalty),
                            time_weight=float(args.time_weight),
                        )
                        for scenario_id in scenarios
                    )
            for rows, summary in results:
                all_rows.extend(rows)
                per_scenario_summaries.append(summary)

    # Progress bar for global tuning
    if not args.skip_global:
        with tqdm(total=1, desc="Global tuning") as pbar:
            rows, summary = _optimize_global(
                scenarios=scenarios,
                algo=args.algo,
                search_space=_search_space_for_global(search_space, scenarios),
                init_points=int(args.init_points),
                n_iter=int(args.n_iter),
                eval_repeats=int(args.eval_repeats),
                seed_base=int(args.seed_base),
                collision_penalty=float(args.collision_penalty),
                time_weight=float(args.time_weight),
            )
            all_rows.extend(rows)
            global_summary = summary
            pbar.update(1)

    runs_df = pd.DataFrame(all_rows)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_out = out_dir / "tuning_runs.parquet"
    summary_out = out_dir / "tuning_summary.json"

    runs_df.to_parquet(runs_out, index=False)

    with summary_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "algo": args.algo,
                "scenarios": scenarios,
                "search_space": {k: [float(v[0]), float(v[1])] for k, v in search_space.items()},
                "global": global_summary,
                "per_scenario": per_scenario_summaries,
            },
            handle,
            indent=2,
        )

    print("\n=== Bayesian tuning complete ===")
    print(f"Runs parquet: {runs_out}")
    print(f"Summary JSON: {summary_out}")


if __name__ == "__main__":
    main()
