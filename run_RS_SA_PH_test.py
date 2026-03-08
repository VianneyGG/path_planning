import subprocess
import sys


configs = [
    {"algo": "RS",  "exp_id": "RS"},
]

base_args = [
    "--init-points", "50",
    "--n-iter", "200",
    "--eval-repeats", "6",
    "--grid-warmstart-points", "100",
    "--runs", "500",
    "--seed", "47",
    "--n-jobs", "20",
    "--mode", "compare",
]

for cfg in configs:
    cmd = [
        "uv", "run", "python", "-m", "src.benchmark.run_pipeline",
        "--algo", cfg["algo"],
        *base_args,
        "--exp-id", cfg["exp_id"],
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
