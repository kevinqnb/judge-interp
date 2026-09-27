#!/bin/bash -l
# scripts/submit.sh <id>
#
# One qsub per id. Runs scripts/run_experiment.py under the project venv on a
# single GPU. Every resource value is READ FROM configs/<id>.yaml's `params`
# block and passed through to qsub unmodified — this wrapper invents nothing.
# Required params keys: gpu_type, gpu_memory, gpu_c, omp, walltime. gpu_type
# must be present but may be left blank (null) to drop the qsub `-l
# gpu_type=...` flag entirely, i.e. let SGE schedule on any GPU type that
# satisfies gpu_memory/gpu_c. A commented-out/missing gpu_type key is still a
# hard error -- that's a different thing from an explicit blank value.
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

# One value per line (not space-separated) so a blank gpu_type doesn't shift
# the remaining fields when read back below.
{
  IFS= read -r GPU_TYPE
  IFS= read -r GPU_MEMORY
  IFS= read -r GPU_C
  IFS= read -r OMP
  IFS= read -r WALLTIME
} < <(
  uv run --python 3.12 python - "$CONFIG" <<'PY'
import sys, yaml
p = yaml.safe_load(open(sys.argv[1]))["params"]
if "gpu_type" not in p:
    raise KeyError("gpu_type key missing from params -- comment it out is not "
                    "enough, it must be present (blank is fine) per submit.sh")
gpu_type = p["gpu_type"]
print(gpu_type if gpu_type is not None else "")
print(p["gpu_memory"])
print(p["gpu_c"])
print(p["omp"])
print(p["walltime"])
PY
)

mkdir -p "$REPO_ROOT/scripts/out"
JOB_NAME="x${ID}"

# REPO_ROOT is passed through as JOB_REPO_ROOT rather than re-derived inside
# _run_experiment_job.sh: SGE stages the submitted script into a per-host
# spool dir (/var/spool/sge/<host>/...) and runs it from there, so a
# BASH_SOURCE-based lookup on the compute node resolves to the spool copy's
# location, not this repo (job 7520650, 2026-09-10, confirmed this).

GPU_TYPE_FLAG=()
if [ -n "$GPU_TYPE" ]; then
    GPU_TYPE_FLAG=(-l "gpu_type=${GPU_TYPE}")
fi

set -x
qsub -N "$JOB_NAME" -P "$SGE_PROJECT" -j y \
     -o "$REPO_ROOT/scripts/out/${ID}.log" -m e \
     -l "h_rt=${WALLTIME}" -pe omp "$OMP" \
     -l gpus=1 "${GPU_TYPE_FLAG[@]}" \
     -l "gpu_memory=${GPU_MEMORY}" -l "gpu_c=${GPU_C}" \
     -v "ID=${ID},JOB_REPO_ROOT=${REPO_ROOT}" \
     "$REPO_ROOT/scripts/_run_experiment_job.sh" "$ID"
