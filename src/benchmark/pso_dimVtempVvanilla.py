from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

from src.environment import Environment
from src.PSO.Config import PSOConfig
from src.PSO.PSO import PSO


DEFAULT_SCENARIOS = [0, 1, 2, 3, 4]
RESET_FALSE_SCENARIOS = {0, 1}

CASE_FLAGS: dict[str, dict[str, bool]] = {
	"vanilla": {
		"dimensional_learning": False,
		"simulated_annealing": False,
		"controlled_cooling": False,
	},
	"dim_only": {
		"dimensional_learning": True,
		"simulated_annealing": False,
		"controlled_cooling": False,
	},
	"sa_no_cc": {
		"dimensional_learning": False,
		"simulated_annealing": True,
		"controlled_cooling": False,
	},
	"dim_plus_sa_no_cc": {
		"dimensional_learning": True,
		"simulated_annealing": True,
		"controlled_cooling": False,
	},
}


@dataclass(frozen=True)
class RunTask:
	scenario: int
	case: str
	run_index: int
	seed: int


def _scenario_path(scenario_id: int) -> Path:
	path = ROOT / "scenarios" / f"scenario{scenario_id}.txt"
	if not path.exists():
		raise FileNotFoundError(f"Missing scenario file: {path}")
	return path


def _scenario_best_json_path(scenario_id: int) -> Path:
	path = ROOT / "src" / "benchmarking" / f"Ob_scenario{scenario_id}" / f"scenario{scenario_id}_best.json"
	if not path.exists():
		raise FileNotFoundError(
			f"Missing best-params JSON for scenario {scenario_id}: {path}"
		)
	return path


def _reset_policy_for_scenario(scenario_id: int) -> bool:
	return scenario_id not in RESET_FALSE_SCENARIOS


def _strip_json_comments(text: str) -> str:
	result: list[str] = []
	i = 0
	in_string = False
	escape = False
	length = len(text)

	while i < length:
		char = text[i]

		if in_string:
			result.append(char)
			if escape:
				escape = False
			elif char == "\\":
				escape = True
			elif char == '"':
				in_string = False
			i += 1
			continue

		if char == '"':
			in_string = True
			result.append(char)
			i += 1
			continue

		if char == "/" and i + 1 < length:
			next_char = text[i + 1]
			if next_char == "/":
				i += 2
				while i < length and text[i] not in "\r\n":
					i += 1
				continue
			if next_char == "*":
				i += 2
				while i + 1 < length and not (text[i] == "*" and text[i + 1] == "/"):
					i += 1
				i += 2
				continue

		result.append(char)
		i += 1

	return "".join(result)


def _load_json_relaxed(path: Path) -> dict[str, Any]:
	text = path.read_text(encoding="utf-8")
	try:
		data = json.loads(text)
		if not isinstance(data, dict):
			raise ValueError(f"Top-level JSON must be an object in {path}")
		return data
	except json.JSONDecodeError:
		cleaned = _strip_json_comments(text)
		cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
		data = json.loads(cleaned)
		if not isinstance(data, dict):
			raise ValueError(f"Top-level JSON must be an object in {path}")
		return data


def _load_base_config_per_scenario(scenario_id: int) -> dict[str, Any]:
	base = asdict(PSOConfig())
	best_json_path = _scenario_best_json_path(scenario_id)
	payload = _load_json_relaxed(best_json_path)

	best_params = payload.get("best_params")
	if not isinstance(best_params, dict):
		raise ValueError(f"Invalid or missing 'best_params' in {best_json_path}")

	base.update(best_params)

	base["reset_waypoints"] =bool(_reset_policy_for_scenario(scenario_id))
	base["pre_heat"] = False
	base["controlled_cooling"] = False

	return base


def _extract_metrics(pso: PSO, env: Environment) -> dict[str, Any]:
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


def _build_tasks(
	scenarios: list[int],
	cases: list[str],
	runs_per_case: int,
	seed_base: int,
	) -> list[RunTask]:
	tasks: list[RunTask] = []
	for scenario_id in scenarios:
		for case in cases:
			for run_index in range(runs_per_case):
				seed = int(seed_base + scenario_id * 100_000 + run_index)
				tasks.append(
					RunTask(
						scenario=scenario_id,
						case=case,
						run_index=run_index,
						seed=seed,
					)
				)
	return tasks


def _run_one(task: RunTask, scenario_base_config: dict[str, Any]) -> dict[str, Any]:
	scenario_id = int(task.scenario)
	case = task.case
	flags = CASE_FLAGS[case]

	config = dict(scenario_base_config)
	config.update(flags)
	config["controlled_cooling"] = False
	config["pre_heat"] = False

	env = Environment()
	env.from_file(str(_scenario_path(scenario_id)))

	np.random.seed(int(task.seed))

	pso = PSO(env, config=config)
	t0 = time.perf_counter()
	pso.run(progress=False, verbose=False)
	elapsed = float(time.perf_counter() - t0)

	metrics = _extract_metrics(pso, env)

	run_id = f"s{scenario_id}_{case}_r{task.run_index:04d}"
	row = {
		"run_id": run_id,
		"scenario": scenario_id,
		"case": case,
		"run_index": int(task.run_index),
		"seed": int(task.seed),
		"time_sec": elapsed,
		"fitness": float(metrics["fitness"]),
		"length": float(metrics["length"]),
		"smoothness": float(metrics["smoothness"]),
		"collisions": int(metrics["collisions"]),
		"corners": int(metrics["corners"]),
		"is_collision_free": bool(int(metrics["collisions"]) == 0),
		"worker_pid": int(os.getpid()),
		"number_of_particules": int(config["number_of_particules"]),
		"number_of_iterations": int(config["number_of_iterations"]),
		"number_of_waypoints": int(config["number_of_waypoints"]),
		"reset_waypoints": bool(config["reset_waypoints"]),
		"reset_number": int(config["reset_number"]),
		"inertia_weight": float(config["inertia_weight"]),
		"best_position_acceleration": float(config["best_position_acceleration"]),
		"global_best_position_acceleration": float(config["global_best_position_acceleration"]),
		"length_weight": float(config["length_weight"]),
		"smoothness_weight": float(config["smoothness_weight"]),
		"collision_weight": float(config["collision_weight"]),
		"corner_weight": float(config["corner_weight"]),
		"dimensional_learning": bool(config["dimensional_learning"]),
		"simulated_annealing": bool(config["simulated_annealing"]),
		"controlled_cooling": bool(config["controlled_cooling"]),
		"pre_heat": bool(config["pre_heat"]),
	}

	return {"row": row, "coords": metrics["coords"]}


def _enforce_dtypes(runs_df: pd.DataFrame) -> pd.DataFrame:
	int_columns = [
		"scenario",
		"run_index",
		"seed",
		"collisions",
		"corners",
		"worker_pid",
		"number_of_particules",
		"number_of_iterations",
		"number_of_waypoints",
		"reset_number",
	]
	float_columns = [
		"time_sec",
		"fitness",
		"length",
		"smoothness",
		"inertia_weight",
		"best_position_acceleration",
		"global_best_position_acceleration",
		"length_weight",
		"smoothness_weight",
		"collision_weight",
		"corner_weight",
	]
	bool_columns = [
		"is_collision_free",
		"reset_waypoints",
		"dimensional_learning",
		"simulated_annealing",
		"controlled_cooling",
		"pre_heat",
	]

	for column in int_columns:
		if column in runs_df.columns:
			runs_df[column] = pd.to_numeric(runs_df[column], errors="coerce").astype("Int64")

	for column in float_columns:
		if column in runs_df.columns:
			runs_df[column] = pd.to_numeric(runs_df[column], errors="coerce")

	for column in bool_columns:
		if column in runs_df.columns:
			runs_df[column] = runs_df[column].astype("boolean")

	return runs_df


def _build_best_paths_points_df(
	runs_df: pd.DataFrame,
	coords_by_run_id: dict[str, np.ndarray],
    ) -> pd.DataFrame:
	if runs_df.empty:
		return pd.DataFrame(
			columns=["scenario", "case", "run_id", "waypoint_idx", "x", "y"]
		)

	winners = (
		runs_df.sort_values(["scenario", "case", "fitness", "time_sec", "run_id"], ascending=[True, True, True, True, True])
		.drop_duplicates(subset=["scenario", "case"], keep="first")
		.reset_index(drop=True)
	)

	rows: list[dict[str, Any]] = []
	for _, winner in winners.iterrows():
		run_id = str(winner["run_id"])
		coords = coords_by_run_id.get(run_id)
		if coords is None or coords.size == 0:
			continue

		for waypoint_idx, point in enumerate(coords):
			rows.append(
				{
					"scenario": int(winner["scenario"]),
					"case": str(winner["case"]),
					"run_id": run_id,
					"waypoint_idx": int(waypoint_idx),
					"x": float(point[0]),
					"y": float(point[1]),
				}
			)

	return pd.DataFrame(rows)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Benchmark PSO on 4 cases (vanilla, DL, SA no CC, DL+SA no CC) across scenarios."
	)
	parser.add_argument("--scenarios", type=int, nargs="+", default=None)
	parser.add_argument(
		"--cases",
		type=str,
		nargs="+",
		default=None,
		help=(
			"Cases to run among: "
			"vanilla, dim_only, sa_no_cc, dim_plus_sa_no_cc. "
			"Default: all cases."
		),
	)
	parser.add_argument("--runs-per-case", type=int, default=100)
	parser.add_argument("--seed-base", type=int, default=42)
	parser.add_argument("--n-jobs", type=int, default=-1)
	parser.add_argument("--out-dir", type=str, default="benchmark_dim_sa_cases")
	parser.add_argument("--show-progress", action="store_true")
	args = parser.parse_args()

	scenarios = args.scenarios if args.scenarios is not None else DEFAULT_SCENARIOS
	scenarios = sorted(set(int(s) for s in scenarios))

	if args.cases is None:
		selected_cases = list(CASE_FLAGS.keys())
	else:
		selected_cases = [str(case).strip() for case in args.cases]
		invalid_cases = [case for case in selected_cases if case not in CASE_FLAGS]
		if invalid_cases:
			raise ValueError(
				f"Unknown case(s): {invalid_cases}. Allowed: {list(CASE_FLAGS.keys())}"
			)
		selected_cases = list(dict.fromkeys(selected_cases))

	if args.runs_per_case < 1:
		raise ValueError("--runs-per-case must be >= 1")

	if args.n_jobs != 1 and (Parallel is None or delayed is None):
		raise ImportError("joblib is required for n_jobs != 1. Install with `pip install joblib`.")

	for scenario_id in scenarios:
		_scenario_path(scenario_id)
		_scenario_best_json_path(scenario_id)

	scenario_base_configs = {
		scenario_id: _load_base_config_per_scenario(scenario_id) for scenario_id in scenarios
	}

	tasks = _build_tasks(
		scenarios=scenarios,
		cases=selected_cases,
		runs_per_case=int(args.runs_per_case),
		seed_base=int(args.seed_base),
	)

	total_expected = len(scenarios) * len(selected_cases) * int(args.runs_per_case)
	print(
		f"Launching benchmark: scenarios={scenarios}, cases={selected_cases}, "
		f"runs_per_case={args.runs_per_case}, total_runs={total_expected}, n_jobs={args.n_jobs}"
	)

	if args.n_jobs == 1:
		results = [_run_one(task, scenario_base_configs[int(task.scenario)]) for task in tasks]
	else:
		results = Parallel(n_jobs=int(args.n_jobs), backend="loky", verbose=10 if args.show_progress else 0)(
			delayed(_run_one)(task, scenario_base_configs[int(task.scenario)]) for task in tasks
		)

	run_rows = [item["row"] for item in results]
	coords_by_run_id = {str(item["row"]["run_id"]): np.asarray(item["coords"], dtype=float) for item in results}

	runs_df = _enforce_dtypes(pd.DataFrame(run_rows))
	best_paths_points_df = _build_best_paths_points_df(runs_df, coords_by_run_id)

	out_dir = ROOT / args.out_dir
	out_dir.mkdir(parents=True, exist_ok=True)

	runs_out = out_dir / "pso_runs.parquet"
	best_paths_out = out_dir / "best_paths_points.parquet"
	summary_out = out_dir / "summary.json"

	runs_df.to_parquet(runs_out, index=False)
	best_paths_points_df.to_parquet(best_paths_out, index=False)

	per_group = (
		runs_df.groupby(["scenario", "case"], as_index=False)
		.agg(
			runs=("run_id", "count"),
			fitness_min=("fitness", "min"),
			fitness_mean=("fitness", "mean"),
			time_sec_mean=("time_sec", "mean"),
			collision_free_runs=("is_collision_free", "sum"),
		)
	)

	summary = {
		"scenarios": scenarios,
		"cases": selected_cases,
		"runs_per_case": int(args.runs_per_case),
		"total_runs": int(len(runs_df)),
		"collision_free_runs": int(runs_df["is_collision_free"].sum()) if "is_collision_free" in runs_df.columns else 0,
		"reset_policy": {
			"false_scenarios": sorted(RESET_FALSE_SCENARIOS),
			"true_scenarios": [s for s in scenarios if s not in RESET_FALSE_SCENARIOS],
		},
		"group_metrics": per_group.to_dict(orient="records"),
	}

	with summary_out.open("w", encoding="utf-8") as handle:
		json.dump(summary, handle, indent=2)

	print("\n=== Benchmark complete ===")
	print(f"Runs parquet: {runs_out}")
	print(f"Best paths parquet: {best_paths_out}")
	print(f"Summary JSON: {summary_out}")


if __name__ == "__main__":
	main()
