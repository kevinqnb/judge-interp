#!/bin/bash -l
# scripts/submit.sh <id>
#
# One qsub per id. Runs scripts/run_experiment.py under the project venv on a
# single GPU. Every resource value is READ FROM configs/<id>.yaml's `params`
# block and passed through to qsub unmodified — this wrapper invents nothing.
# Required params keys: gpu_type, gpu_memory, gpu_c, omp, walltime.
#
# SGE job names cannot start with a digit and contract ids always do
# (YYYY-MM-DD-slug-NN), so `qsub -N` gets "x<id>". run.json's job_id comes from
# SGE's numeric $JOB_ID, not from -N.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: bash scripts/submit.sh <id>" >&2
    exit 1
fi
ID="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/$ID.yaml"
[ -f "$CONFIG" ] || { echo "error: no such config: $CONFIG" >&2; exit 1; }

: "${SGE_PROJECT:?SGE_PROJECT is not set -- export it in your cluster profile}"

read -r GPU_TYPE GPU_MEMORY GPU_C OMP WALLTIME < <(
  uv run --python 3.12 python - "$CONFIG" <<'PY'
import sys, yaml
p = yaml.safe_load(open(sys.argv[1]))["params"]
print(p["gpu_type"], p["gpu_memory"], p["gpu_c"], p["omp"], p["walltime"])
PY
)

mkdir -p "$REPO_ROOT/scripts/out"
JOB_NAME="x${ID}"

# REPO_ROOT is passed through as JOB_REPO_ROOT rather than re-derived inside
# _run_experiment_job.sh: SGE stages the submitted script into a per-host
# spool dir (/var/spool/sge/<host>/...) and runs it from there, so a
# BASH_SOURCE-based lookup on the compute node resolves to the spool copy's
# location, not this repo (job 7520650, 2026-09-10, confirmed this).

set -x
qsub -N "$JOB_NAME" -P "$SGE_PROJECT" -j y \
     -o "$REPO_ROOT/scripts/out/${ID}.log" -m e \
     -l "h_rt=${WALLTIME}" -pe omp "$OMP" \
     -l gpus=1 -l "gpu_type=${GPU_TYPE}" \
     -l "gpu_memory=${GPU_MEMORY}" -l "gpu_c=${GPU_C}" \
     -v "ID=${ID},JOB_REPO_ROOT=${REPO_ROOT}" \
     "$REPO_ROOT/scripts/_run_experiment_job.sh" "$ID"
