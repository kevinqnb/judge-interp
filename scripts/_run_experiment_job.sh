#!/bin/bash -l
# scripts/_run_experiment_job.sh <id>
#
# The qsub job body (submitted by scripts/submit.sh). Runs on the assigned GPU
# node. Module loads live here, not in submit.sh, because they only take effect
# inside the batch session.
set -euo pipefail

ID="${1:?usage: _run_experiment_job.sh <id>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# TODO(smoke): confirm the CUDA module a GPU node needs for torch, then pin it
# here and record it in CLAUDE.local.md. API-only smoke never exercised this.
# module load cuda/<ver>

exec uv run --python 3.12 python scripts/run_experiment.py "configs/${ID}.yaml"
