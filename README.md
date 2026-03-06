# path_planing

2D path planning project with RRT and PSO algorithms.

## Requirements

- Python 3.13
- Scenario files in `scenarios/`

## Setup (choose one)

### Option A: uv (recommended)

```bash
uv python install 3.13
uv venv --python 3.13
uv pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate path_planning
```

### Option C: venv + pip

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

Run one of the available modes:

```bash
python main.py rrt
python main.py pso
python main.py multi
```

Optional animation:

```bash
python main.py rrt 0 --animate
```

## Benchmark pipeline

Run tuning + benchmarking + reporting + plots:

```bash
python -m src.benchmark.run_pipeline --algo RS_SA_noCC_DL --scenarios 0 1 2 3 4
```

Outputs are written under:

- `src/benchmark/artifacts/<ALGO>/tuning/`
- `src/benchmark/artifacts/<ALGO>/benchmark/`
- `src/benchmark/artifacts/<ALGO>/plots/`

### `run_pipeline` CLI reference

Main command:

```bash
python -m src.benchmark.run_pipeline --algo RS_SA_PH --mode compare --exp-id RS_SA_PH
```

Key arguments:

- `--algo`: tuned algorithm key (ex: `RS`, `RS_SA_noCC`, `RS_SA_PH`, `RS_SA_CC_DL`)
- `--scenarios`: list of scenarios (default: `0 1 2 3 4`)
- `--runs`: benchmark runs per algo/scenario
- `--n-jobs`: outer parallel workers
- `--chunk-size`: benchmark chunk size (default `24`, safe baseline)
- `--adaptive-chunking` / `--no-adaptive-chunking`: scenario-aware chunk sizing (enabled by default)
- `--mode`: `compare` (vanilla + tuned) or `vanilla_only`
- `--exp-id`: output experiment folder under `src/benchmark/artifacts/`

Tuning/HPO arguments:

- `--hpo-backend`: `optuna` (default) or `bayes_opt`  
  (Optuna runs are now quiet; per‑trial “Trial … finished” messages are suppressed)
- `--init-points`: warm/random startup budget
- `--n-iter`: optimization iterations
- `--eval-repeats`: repeats per trial for robustness
- `--grid-warmstart-points`: lightweight grid points before main HPO
- `--grid-focus-params`: number of priority params for grid warm-start
- The pipeline runs tuning in `per_scenario` mode only (global tuning is disabled in `run_pipeline`)

Penalty and confidence arguments:

- `--disable-auto-penalties`: disable per-scenario auto scaling
- `--penalty-calibration-runs`: baseline runs used to scale penalties
- `--collision-penalty`, `--non-collision-free-penalty`, `--collision-free-weight`, `--no-feasible-penalty`
- `--confidence-level`: confidence level used in loss curve plotting (ex: `0.95`)

Loss and diagnostic logs:

- `--log-loss-curves` / `--no-log-loss-curves` (tuning only, disabled by default in `run_pipeline`)
- `--log-test-loss-curves` / `--no-log-test-loss-curves` (benchmark/test only, enabled by default)
- `--log-pso-curves`
- `--plot-loss-curves` / `--no-plot-loss-curves`

Baseline (Basic) behavior in compare mode:

- `--vanilla-params-summary` defaults to `src/benchmark/artifacts/basic/tuning/tuning_summary.json`
- If this file is missing, the pipeline prints a warning and falls back to default vanilla config

### Where are benchmark test logs?

For each pipeline run (`--exp-id <ID>`), benchmark logs are stored in:

- `src/benchmark/artifacts/<ID>/benchmark/benchmark_run_log.jsonl` (chunk-level execution logs)
- `src/benchmark/artifacts/<ID>/benchmark/benchmark_summary.json` (aggregated metrics + metadata)
- `src/benchmark/artifacts/<ID>/benchmark/benchmark_runs.parquet` (per-run raw metrics)

Tuning logs (if enabled) are stored in:

- `src/benchmark/artifacts/<ID>/tuning/tuning_loss_curves.parquet`
- `src/benchmark/artifacts/<ID>/tuning/tuning_pso_curves.parquet`

Benchmark test-loss logs (if enabled) are stored in:

- `src/benchmark/artifacts/<ID>/benchmark/benchmark_test_loss_curves.parquet`

## Project layout

- `main.py`: main CLI entry point
- `src/environment.py`: environment and rendering
- `src/PSO/`: PSO implementation
- `src/RRT/`: RRT implementation
- `src/benchmark/core/`: shared benchmark primitives (`common.py`, `algo_profiles.py`)
- `src/benchmark/jobs/`: executable benchmark jobs (`run_pipeline`, `tune_algo_bayes`, `benchmark_algo_vs_basic`)
- `src/benchmark/viz/`: reports and plotting (`performance`, `plot_all_algos`, `plot_tuning_curves`)
- `src/benchmark/archive/`: legacy/ad-hoc benchmark scripts kept for reference
- `src/benchmark/`: compatibility wrappers for historical imports / `python -m src.benchmark.<module>`
- `scenarios/`: input maps

## Notes

- If `python` points to another interpreter, use your environment-specific launcher (`uv run`, activated `conda`, or activated `.venv`).
- Keep dependencies synchronized between `requirements.txt`, `environment.yml`, and `pyproject.toml` when updating packages.
