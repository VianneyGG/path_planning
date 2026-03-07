"""Run predefined benchmark configurations for RS/SA-PH style experiments."""

from __future__ import annotations

import subprocess
import sys


CONFIGS = [
    {"algo": "RS",  "exp_id": "RS"},
    {"algo": "RS_SA_PH",     "exp_id": "RS_SA_PH"},
    {"algo": "RS_SA_noCC",   "exp_id": "SA_noCC"},
    {"algo": "RS_SA_noCC_DL","exp_id": "SA_noCC_DL"},
    {"algo": "RS_SA_CC_DL",  "exp_id": "SA_CC_DL"},
    {"algo": "RS_SA_CC",     "exp_id": "SA_CC"},
]

BASE_ARGS = [
    "--init-points", "50",
    "--n-iter", "200",
    "--eval-repeats", "6",
    "--grid-warmstart-points", "100",
    "--runs", "500",
    "--seed", "47",
    "--n-jobs", "20",
    "--mode", "compare",
]


def main() -> int:
    """Execute configured benchmark runs sequentially.

    Returns:
        Process exit code.
    """
    for cfg in CONFIGS:
        cmd = [
            "uv", "run", "python", "-m", "src.benchmark.run_pipeline",
            "--algo", cfg["algo"],
            *BASE_ARGS,
            "--exp-id", cfg["exp_id"],
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return int(result.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
