from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.environment import Environment
from src.PSO.Config import PSOConfig
from src.PSO.PSO import PSO
from src.benchmark.algo_profiles import apply_algo_flags
from src.benchmark.pso_dimVtempVvanilla import _extract_metrics, _scenario_path

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _load_tuned_params(summary_path: Path, strategy: str, scenario_id: int) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    if strategy == "per_scenario":
        per = payload.get("per_scenario", [])
        for item in per:
            if int(item.get("scenario", -1)) == int(scenario_id):
                params = item.get("best_params")
                if isinstance(params, dict):
                    return params
        raise ValueError(f"No per-scenario best params found for scenario {scenario_id} in {summary_path}")

    if strategy == "global":
        global_block = payload.get("global")
        if not isinstance(global_block, dict):
            raise ValueError(f"Missing 'global' section in {summary_path}")
        by_scenario = global_block.get("best_params_by_scenario", {})
        params = by_scenario.get(str(scenario_id))
        if isinstance(params, dict):
            return params
        raise ValueError(f"No global-derived params for scenario {scenario_id} in {summary_path}")

    raise ValueError("Unknown strategy. Use 'global' or 'per_scenario'.")


def _run_one(
    scenario_id: int,
    algo: str,
    config: dict[str, Any],
    run_index: int,
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

    return {
        "run_id": f"s{scenario_id}_{algo}_r{run_index:04d}",
        "scenario": int(scenario_id),
        "algo": str(algo),
        "run_index": int(run_index),
        "seed": int(seed),
        "fitness": float(metrics["fitness"]),
        "length": float(metrics["length"]),
        "smoothness": float(metrics["smoothness"]),
        "collisions": int(metrics["collisions"]),
        "corners": int(metrics["corners"]),
        "time_sec": float(elapsed),
        "is_collision_free": bool(int(metrics["collisions"]) == 0),
        "worker_pid": int(os.getpid()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a tuned algorithm vs vanilla baseline.")
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--mode", type=str, choices=["compare", "vanilla_only"], default="compare")
    parser.add_argument("--params-summary", type=str, default=None)
    parser.add_argument("--params-strategy", type=str, choices=["global", "per_scenario"], default="per_scenario")
    parser.add_argument("--out-dir", type=str, default="artifacts/exp/benchmark")
    args = parser.parse_args()

    if args.n_jobs != 1 and (Parallel is None or delayed is None):
        raise ImportError("joblib is required for n_jobs != 1. Install with `uv pip install joblib`.")

    scenarios = sorted(set(int(s) for s in args.scenarios))
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    tuned_summary_path = Path(args.params_summary) if args.params_summary else None
    if args.mode == "compare" and tuned_summary_path is None:
        raise ValueError("--params-summary is required in compare mode.")

    tasks: list[tuple[int, str, dict[str, Any], int, int]] = []


    # Charger les meilleurs params vanilla depuis tuning_summary.json si mode compare
    vanilla_tuning_summary = None
    if args.mode == "compare":
        # On suppose que le tuning vanilla a été fait dans le même dossier que params-summary
        vanilla_tuning_path = Path(args.params_summary).parent / "tuning_summary.json"
        if vanilla_tuning_path.exists():
            with open(vanilla_tuning_path, "r", encoding="utf-8") as f:
                vanilla_tuning_summary = json.load(f)
        else:
            print(f"[WARN] Vanilla tuning summary not found at {vanilla_tuning_path}, fallback to default config.")

    def get_best_vanilla_params(scenario_id: int) -> dict:
        if vanilla_tuning_summary is not None:
            # Chercher per_scenario
            per = vanilla_tuning_summary.get("per_scenario", [])
            for item in per:
                if int(item.get("scenario", -1)) == int(scenario_id):
                    params = item.get("best_params")
                    if isinstance(params, dict):
                        return params
        return None

    for scenario_id in scenarios:
        base_cfg = _load_base_config(scenario_id)

        # Si mode compare, utiliser les meilleurs params vanilla issus du tuning
        if args.mode == "compare":
            best_vanilla = get_best_vanilla_params(scenario_id)
            if best_vanilla is not None:
                vanilla_cfg = dict(base_cfg)
                vanilla_cfg.update(best_vanilla)
                vanilla_cfg = apply_algo_flags(vanilla_cfg, "vanilla")
            else:
                vanilla_cfg = apply_algo_flags(base_cfg, "vanilla")
        else:
            vanilla_cfg = apply_algo_flags(base_cfg, "vanilla")

        tuned_cfg = None
        if args.mode == "compare":
            tuned_params = _load_tuned_params(
                summary_path=tuned_summary_path,
                strategy=args.params_strategy,
                scenario_id=scenario_id,
            )
            tuned_cfg = dict(base_cfg)
            tuned_cfg.update(tuned_params)
            tuned_cfg = apply_algo_flags(tuned_cfg, args.algo)

        for run_index in range(int(args.runs)):
            seed = int(args.seed_base + scenario_id * 100_000 + run_index)
            tasks.append((scenario_id, "vanilla", vanilla_cfg, run_index, seed))
            if tuned_cfg is not None:
                tasks.append((scenario_id, args.algo, tuned_cfg, run_index, seed))

    print(
        f"Launching benchmark: scenarios={scenarios}, runs={args.runs}, mode={args.mode}, "
        f"total_runs={len(tasks)}, n_jobs={args.n_jobs}"
    )

    def _exec(task: tuple[int, str, dict[str, Any], int, int]) -> dict[str, Any]:
        scenario_id, algo, cfg, run_idx, seed = task
        return _run_one(scenario_id=scenario_id, algo=algo, config=cfg, run_index=run_idx, seed=seed)

    if int(args.n_jobs) == 1:
        rows = [_exec(task) for task in tasks]
    else:
        rows = Parallel(n_jobs=int(args.n_jobs), backend="loky", verbose=10)(
            delayed(_exec)(task) for task in tasks
        )

    runs_df = pd.DataFrame(rows)

    summary_df = (
        runs_df.groupby(["scenario", "algo"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            fitness_mean=("fitness", "mean"),
            fitness_var=("fitness", "var"),
            time_mean=("time_sec", "mean"),
            time_var=("time_sec", "var"),
            collision_free_mean=("is_collision_free", "mean"),
            collision_free_var=("is_collision_free", "var"),
        )
        .sort_values(["scenario", "algo"])
        .reset_index(drop=True)
    )

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_out = out_dir / "benchmark_runs.parquet"
    summary_out = out_dir / "benchmark_summary.json"

    runs_df.to_parquet(runs_out, index=False)

    with summary_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mode": args.mode,
                "algo": args.algo,
                "scenarios": scenarios,
                "runs": int(args.runs),
                "num_rows": int(len(runs_df)),
                "group_metrics": summary_df.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    print("\n=== Benchmark complete ===")
    print(f"Runs parquet: {runs_out}")
    print(f"Summary JSON: {summary_out}")


if __name__ == "__main__":
    main()
