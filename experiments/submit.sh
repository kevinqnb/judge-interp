#!/bin/bash -l
# experiments/submit.sh [--analysis <script> | --api] <id>
#
# --analysis <script>: CPU-only job that runs `python <script> <config path>`
# (e.g. analysis/entity_detection.py) instead of run_experiment.py. Requests no GPU
# and reads only `omp` and `walltime` from params. Explicit flag, never inferred.
#
# --api: CPU-only job that runs run_experiment.py for a task whose model is called over an API
# (the jev_* tasks). Requests no GPU, reads only `omp` and `walltime`, and runs with the repo's
# .env (TYPESAFE_API_KEY). Explicit flag, never inferred; mutually exclusive with --analysis.
#
# One qsub per id. Runs experiments/run_experiment.py under the project venv on a
# single GPU. Every resource value is READ FROM the id's config's `params`
# block and passed through to qsub unmodified — this wrapper invents nothing.
# The config is experiments/experiment-configs/<group>/<id>.yaml, or analysis/configs/<type>/<id>.yaml
# for --analysis: the id is looked up across the subdirectories of the one root that applies and must match exactly one file (none or several is an error). The repo-relative
# path is passed to the job, which does not search again.
#
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

ANALYSIS_SCRIPT=""
API_JOB=0
if [ "${1:-}" = "--analysis" ]; then
    ANALYSIS_SCRIPT="${2:?--analysis needs a script path}"
    shift 2
elif [ "${1:-}" = "--api" ]; then
    API_JOB=1
    shift
fi
if [ $# -ne 1 ]; then
    echo "usage: bash experiments/submit.sh [--analysis <script> | --api] <id>" >&2
    exit 1
fi
ID="$1"
[ -z "$ANALYSIS_SCRIPT" ] || [ -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/$ANALYSIS_SCRIPT" ] \
    || { echo "error: no such analysis script: $ANALYSIS_SCRIPT" >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "$ANALYSIS_SCRIPT" ]; then CONFIG_ROOT="$REPO_ROOT/analysis/configs"; else CONFIG_ROOT="$REPO_ROOT/experiments/experiment-configs"; fi
mapfile -t MATCHES < <(find "$CONFIG_ROOT" -type f -name "${ID}.yaml")
[ "${#MATCHES[@]}" -eq 1 ] || { echo "error: expected exactly one ${ID}.yaml under $CONFIG_ROOT, found ${#MATCHES[@]}: ${MATCHES[*]:-}" >&2; exit 1; }
CONFIG="${MATCHES[0]}"
CONFIG_PATH="${CONFIG#"$REPO_ROOT"/}"

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
  uv run --python 3.12 python - "$CONFIG" "$ANALYSIS_SCRIPT" "$API_JOB" <<'PY'
import sys, yaml
p = yaml.safe_load(open(sys.argv[1]))["params"]
if sys.argv[2] or sys.argv[3] == "1":  # --analysis / --api: CPU-only, no GPU keys read
    for _ in range(3):
        print("")
    print(p["omp"])
    print(p["walltime"])
    sys.exit(0)
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

mkdir -p "$REPO_ROOT/experiments/out"
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

GPU_FLAGS=(-l gpus=1 "${GPU_TYPE_FLAG[@]}" -l "gpu_memory=${GPU_MEMORY}" -l "gpu_c=${GPU_C}")
if [ -n "$ANALYSIS_SCRIPT" ] || [ "$API_JOB" = 1 ]; then
    GPU_FLAGS=()
fi

set -x
qsub -N "$JOB_NAME" -P "$SGE_PROJECT" -j y \
     -o "$REPO_ROOT/experiments/out/${ID}.log" -m e \
     -l "h_rt=${WALLTIME}" -pe omp "$OMP" \
     "${GPU_FLAGS[@]}" \
     -v "ID=${ID},JOB_REPO_ROOT=${REPO_ROOT},CONFIG_PATH=${CONFIG_PATH},ANALYSIS_SCRIPT=${ANALYSIS_SCRIPT},API_JOB=${API_JOB}" \
     "$REPO_ROOT/experiments/_run_experiment_job.sh" "$ID"
