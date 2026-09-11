#!/bin/bash -l
# scripts/_run_experiment_job.sh <id>
#
# The qsub job body (submitted by scripts/submit.sh). Runs on the assigned GPU
# node. Module loads live here, not in submit.sh, because they only take effect
# inside the batch session.
set -euo pipefail

ID="${1:?usage: _run_experiment_job.sh <id>}"
# Not derived from BASH_SOURCE: SGE stages this script into a per-host spool
# dir and runs it from there, so that would resolve to the spool copy's
# location, not the repo (see submit.sh). submit.sh passes the real path.
REPO_ROOT="${JOB_REPO_ROOT:?JOB_REPO_ROOT not set -- must be passed via qsub -v from scripts/submit.sh}"
cd "$REPO_ROOT"

# TODO(smoke): confirm the CUDA module a GPU node needs for torch, then pin it
# here and record it in CLAUDE.local.md. API-only smoke never exercised this.
# module load cuda/<ver>

# CLAUDE.local.md's HF_HOME (this account's cache on the project allocation,
# not $HOME). Not sourced from the shell profile: it's only set there via a
# manual `hfhome` alias, not an unconditional export like TMPDIR/RUNS_ROOT/
# UV_CACHE_DIR right next to it, so `bash -l` never picks it up on a compute
# node -- job 7520682 (2026-09-10) fell back to ~/.cache/huggingface and blew
# the $HOME disk quota re-downloading the model.
export HF_HOME=/projectnb/mcnet/kevin/cache-hf

exec uv run --python 3.12 python scripts/run_experiment.py "configs/${ID}.yaml"
