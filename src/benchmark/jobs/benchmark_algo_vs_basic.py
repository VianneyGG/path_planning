from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None

from src.environment import Environment
from src.PSO.pso_config import PSOConfig
from src.PSO.pso_solver import PSO
from src.benchmark.core.algo_profiles import apply_algo_flags, DEFAULT_RESET_NUMBER
from src.benchmark.core.common import scenario_path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ENV_CACHE: dict[int, Environment] = {}


def _resolve_n_jobs(requested_n_jobs: int, num_tasks: int) -> int:
    if requested_n_jobs == 1:
        return 1
    cpu = max(1, int(os.cpu_count() or 1))
    if requested_n_jobs <= 0:
        requested_n_jobs = max(1, cpu - 1)
    return max(1, min(int(requested_n_jobs), cpu, max(1, num_tasks)))


def _get_env(scenario_id: int) -> Environment:
    sid = int(scenario_id)
    env = _ENV_CACHE.get(sid)
    if env is not None:
        return env
    env = Environment()
    env.from_file(str(scenario_path(sid)))
    _ENV_CACHE[sid] = env
    return env


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


def _canonical_basic_summary_path() -> Path:
    return ROOT / "src" / "benchmark" / "artifacts" / "basic" / "tuning" / "tuning_summary.json"


def _load_vanilla_tuning_summary(path_override: str | None) -> dict[str, Any] | None:
    candidate = Path(path_override) if path_override else _canonical_basic_summary_path()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.exists():
        _LOG.warning("Vanilla tuning summary not found at %s; fallback to default config.", candidate)
        return None
    with candidate.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_best_vanilla_params(vanilla_tuning_summary: dict[str, Any] | None, scenario_id: int) -> dict[str, Any] | None:
    if vanilla_tuning_summary is None:
        return None
    per = vanilla_tuning_summary.get("per_scenario", [])
    for item in per:
        if int(item.get("scenario", -1)) == int(scenario_id):
            params = item.get("best_params")
            if isinstance(params, dict):
                return params
    return None


def _run_one(
    scenario_id: int,
    algo: str,
    config: dict[str, Any],
    run_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run a single PSO execution.

    Always returns at least one row:
    - CF run  -> all iteration rows (with ``iteration >= 0``).
    - non-CF run -> single summary row with ``iteration = -1``.

    Every row carries:
    - ``is_collision_free``  - True if the run ultimately ended CF.
    - ``elapsed_s``          - wall-clock seconds of the full run.
    - ``path_length_m``      - final best-path length from ``pso.solution``.
    These are run-level constants, repeated on every row so callers can
    ``drop_duplicates("run_id")`` to get per-run summaries for CF-proportion,
    CPU-time and path-length plots.
    """
    env = _get_env(scenario_id)
    np.random.seed(int(seed))
    pso = PSO(env, config=config)

    run_id = f"s{scenario_id}_{algo}_r{run_index:04d}"
    iter_rows: list[dict[str, Any]] = []

    def _on_iteration(payload: dict[str, Any]) -> None:
        iter_rows.append(
            {
                "run_id":            run_id,
                "scenario":          int(scenario_id),
                "algo":              str(algo),
                "iteration":         int(payload.get("iteration", -1)),
                "best_fitness":      float(payload.get("best_fitness", float("inf"))),
                "is_collision_free": bool(payload.get("is_collision_free", False)),
                # placeholders - filled in after pso.run() completes
                "elapsed_s":         float("nan"),
                "path_length_m":     float("nan"),
            }
        )

    t0 = time.perf_counter()
    pso.run(progress=False, verbose=False, iteration_callback=_on_iteration)
    elapsed_s = float(time.perf_counter() - t0)

    path_length_m: float = float("nan")
    try:
        if pso.solution is not None:
            path_length_m = float(pso.solution.total_length())
    except Exception:
        pass

    run_ended_cf = any(r["is_collision_free"] for r in iter_rows)

    if run_ended_cf:
        # Back-fill the run-level constants into every collected row.
        for r in iter_rows:
            r["elapsed_s"] = elapsed_s
            r["path_length_m"] = path_length_m
        return iter_rows

    # Non-CF run: return a single summary sentinel row so the overall run is
    # represented in the parquet (needed for CF-proportion computation).
    last_fitness = iter_rows[-1]["best_fitness"] if iter_rows else float("inf")
    return [
        {
            "run_id":            run_id,
            "scenario":          int(scenario_id),
            "algo":              str(algo),
            "iteration":         -1,
            "best_fitness":      last_fitness,
            "is_collision_free": False,
            "elapsed_s":         elapsed_s,
            "path_length_m":     path_length_m,
        }
    ]


def _run_chunk(
    scenario_id: int,
    algo: str,
    config: dict[str, Any],
    run_start: int,
    run_end: int,
    seed_base: int,
) -> list[dict[str, Any]]:
    """Run a chunk of PSO executions; accumulate rows from _run_one."""
    iter_rows: list[dict[str, Any]] = []
    for run_index in range(int(run_start), int(run_end)):
        seed = int(seed_base + scenario_id * 100_000 + run_index)
        iter_rows.extend(
            _run_one(
                scenario_id=int(scenario_id),
                algo=str(algo),
                config=config,
                run_index=int(run_index),
                seed=int(seed),
            )
        )
    return iter_rows


def _effective_chunk_size(
    *,
    scenario_id: int,
    algo: str,
    base_chunk_size: int,
    adaptive_chunking: bool,
) -> int:
    if not adaptive_chunking:
        return max(1, int(base_chunk_size))

    factor = 1.0
    sid = int(scenario_id)
    if sid >= 4:
        factor *= 0.50
    elif sid == 3:
        factor *= 0.65
    elif sid == 2:
        factor *= 0.80

    if str(algo) != "vanilla":
        factor *= 0.80

    proposed = int(round(float(base_chunk_size) * factor))
    min_chunk = max(4, int(base_chunk_size) // 3)
    return max(min_chunk, proposed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark a tuned algorithm vs vanilla baseline.")
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--adaptive-chunking", dest="adaptive_chunking", action="store_true")
    parser.add_argument("--no-adaptive-chunking", dest="adaptive_chunking", action="store_false")
    parser.set_defaults(adaptive_chunking=True)
    parser.add_argument("--mode", type=str, choices=["compare", "vanilla_only"], default="compare")
    parser.add_argument("--params-summary", type=str, default=None)
    parser.add_argument("--vanilla-params-summary", type=str, default=None)
    parser.add_argument("--params-strategy", type=str, choices=["global", "per_scenario"], default="per_scenario")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--out-dir", type=str, default="artifacts/exp/benchmark")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    scenarios = sorted(set(int(s) for s in args.scenarios))
    if int(args.runs) < 1:
        raise ValueError("--runs must be >= 1")
    if int(args.chunk_size) < 1:
        raise ValueError("--chunk-size must be >= 1")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)


    tuned_summary_path = Path(args.params_summary) if args.params_summary else None
    if args.mode == "compare" and tuned_summary_path is None:
        raise ValueError("--params-summary is required in compare mode.")

    force_single_thread_fitness = int(args.n_jobs) != 1

    def _sanitize_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
        out = dict(cfg)
        if force_single_thread_fitness:
            out["parallel_fitness_workers"] = 1
            out["reuse_fitness_thread_pool"] = True
        return out

    vanilla_tuning_summary = None
    if args.mode == "compare":
        vanilla_tuning_summary = _load_vanilla_tuning_summary(args.vanilla_params_summary)

    configs_by_scenario: dict[int, dict[str, dict[str, Any]]] = {}
    for scenario_id in scenarios:
        base_cfg = _load_base_config(scenario_id)

        best_vanilla = _get_best_vanilla_params(vanilla_tuning_summary, scenario_id)
        if best_vanilla is not None:
            vanilla_cfg = dict(base_cfg)
            vanilla_cfg.update(best_vanilla)
            vanilla_cfg = apply_algo_flags(vanilla_cfg, "vanilla")
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

        configs_by_scenario[int(scenario_id)] = {
            "vanilla": _sanitize_cfg(vanilla_cfg),
            "tuned": _sanitize_cfg(tuned_cfg) if tuned_cfg is not None else None,
        }

    chunk_tasks: list[tuple[int, str, dict[str, Any], int, int, int]] = []
    for scenario_id in scenarios:
        stream_algos: list[tuple[str, dict[str, Any] | None]] = [("vanilla", configs_by_scenario[scenario_id]["vanilla"])]
        tuned_cfg = configs_by_scenario[scenario_id]["tuned"]
        if tuned_cfg is not None:
            stream_algos.append((str(args.algo), tuned_cfg))

        for algo_name, cfg in stream_algos:
            if cfg is None:
                continue
            eff_chunk = _effective_chunk_size(
                scenario_id=int(scenario_id),
                algo=str(algo_name),
                base_chunk_size=int(args.chunk_size),
                adaptive_chunking=bool(args.adaptive_chunking),
            )
            run_start = 0
            while run_start < int(args.runs):
                run_end = min(int(args.runs), run_start + int(eff_chunk))
                chunk_tasks.append((scenario_id, str(algo_name), cfg, run_start, run_end, int(eff_chunk)))
                run_start = run_end

    chunk_tasks.sort(key=lambda t: (int(t[5]), int(t[0])), reverse=False)

    n_jobs = _resolve_n_jobs(int(args.n_jobs), num_tasks=len(chunk_tasks))
    if n_jobs != 1 and (Parallel is None or delayed is None):
        raise ImportError("joblib is required for n_jobs != 1. Install with `uv pip install joblib`.")

    _LOG.info(
        "Launching benchmark: scenarios=%s, runs=%s, mode=%s, "
        "chunks=%d, chunk_size=%s, adaptive_chunking=%s, n_jobs=%d",
        scenarios, args.runs, args.mode,
        len(chunk_tasks), args.chunk_size, args.adaptive_chunking, n_jobs,
    )

    def _exec(task: tuple[int, str, dict[str, Any], int, int, int]) -> list[dict[str, Any]]:
        scenario_id, algo, cfg, run_start, run_end, _eff_chunk = task
        return _run_chunk(
            scenario_id=scenario_id,
            algo=algo,
            config=cfg,
            run_start=run_start,
            run_end=run_end,
            seed_base=int(args.seed_base),
        )

    all_iter_rows: list[list[dict[str, Any]]] = []

    if n_jobs == 1:
        for task in chunk_tasks:
            all_iter_rows.append(_exec(task))
    else:
        try:
            gen = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, return_as="generator_unordered")(
                delayed(_exec)(task) for task in chunk_tasks
            )
            for chunk_result in gen:
                all_iter_rows.append(chunk_result)
        except TypeError:
            all_iter_rows = list(
                Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                    delayed(_exec)(task) for task in chunk_tasks
                )
            )

    loss_rows = [row for chunk_result in all_iter_rows for row in chunk_result]
    loss_df = pd.DataFrame(loss_rows)

    loss_out = out_dir / "benchmark_loss_curves.parquet"
    if not loss_df.empty:
        loss_df.to_parquet(loss_out, index=False)

    _LOG.info("=== Benchmark complete ===")
    if not loss_df.empty:
        _LOG.info("Loss curves parquet: %s (%d rows)", loss_out, len(loss_df))
    else:
        _LOG.info("No collision-free iterations recorded; loss curves file not written.")


if __name__ == "__main__":
    main()

