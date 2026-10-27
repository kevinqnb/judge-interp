#!/bin/bash -l
# experiments/_run_experiment_job.sh <id>
#
# The qsub job body (submitted by experiments/submit.sh). Runs on the assigned GPU
# node. Module loads live here, not in submit.sh, because they only take effect
# inside the batch session.
set -euo pipefail

ID="${1:?usage: _run_experiment_job.sh <id>}"
# Not derived from BASH_SOURCE: SGE stages this script into a per-host spool
# dir and runs it from there, so that would resolve to the spool copy's
# location, not the repo (see submit.sh). submit.sh passes the real path.
REPO_ROOT="${JOB_REPO_ROOT:?JOB_REPO_ROOT not set -- must be passed via qsub -v from experiments/submit.sh}"
cd "$REPO_ROOT"
# Repo-relative config path, resolved from the id by submit.sh (the config lives in a group subdirectory).
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH not set -- must be passed via qsub -v from experiments/submit.sh}"

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

# CPU-only analysis job (submit.sh --analysis <script>): ANALYSIS_SCRIPT is always
# passed by submit.sh, empty for a normal experiment. BLAS threads follow the config's
# params.thread_cap (no default: a missing key fails here), at most the granted slots.
if [ -n "${ANALYSIS_SCRIPT}" ]; then
    : "${NSLOTS:?NSLOTS not set -- not running under SGE}"
    THREADS="$(uv run --python 3.12 python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['params']['thread_cap'])" "${CONFIG_PATH}")"
    [[ "$THREADS" =~ ^[1-9][0-9]*$ ]] && [ "$THREADS" -le "$NSLOTS" ] \
        || { echo "error: params.thread_cap=${THREADS} must be an integer in [1, NSLOTS=${NSLOTS}]" >&2; exit 1; }
    export OMP_NUM_THREADS="$THREADS"
    # numpy and scipy each bundle an OpenBLAS (and others ship MKL/numexpr/Accelerate), and
    # OMP_NUM_THREADS alone does not cap all of them, so pin every pool. Pinning to the slot count
    # still overran it on compute nodes (CPU/wall 10-18x on 8-16 slots; 8098591 and, 2026-10-26,
    # 8289218 / 8289220 were SIGKILLed), hence a cap below the slots in the config.
    export OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS"
    # Log which pools exist and their thread counts, to verify the caps took effect.
    uv run --python 3.12 --extra analysis python -c "import numpy, scipy, sklearn; from threadpoolctl import threadpool_info; [print('threadpool:', d['internal_api'], d['num_threads'], d['filepath']) for d in threadpool_info()]"
    exec uv run --python 3.12 --extra analysis python "${ANALYSIS_SCRIPT}" "${CONFIG_PATH}"
fi

# API-backed task (submit.sh --api): no GPU; the key (TYPESAFE_API_KEY) comes from the repo's .env.
# API_JOB is always passed by submit.sh.
if [ "${API_JOB}" = 1 ]; then
    exec uv run --python 3.12 --env-file .env python experiments/run_experiment.py "${CONFIG_PATH}"
fi

exec uv run --python 3.12 python experiments/run_experiment.py "${CONFIG_PATH}"
