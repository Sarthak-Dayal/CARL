#!/usr/bin/env bash
# run_parallel.sh — run a command in parallel, once per GPU.
#
# Usage:  run_parallel.sh [-g GPUS] [-l LOG_DIR] -- CMD [ARGS...]
#   -g GPUS  comma/space-separated GPU IDs (default: auto-detect via nvidia-smi)
#   -l DIR   log directory (default: ./logs)
#
# Spawns one process per GPU in its own session, with CUDA_VISIBLE_DEVICES set.
# Each process logs to a unique file. SIGINT/SIGTERM tears down all process
# groups (5s grace, then SIGKILL).
#
# Examples:
#   ./run_parallel.sh -- wandb agent ENT/PROJ/SWEEP
set -Eeuo pipefail

GPUS=""; LOG_DIR="./logs"
while getopts "g:l:" opt; do
  case "$opt" in
    g) GPUS="$OPTARG" ;;
    l) LOG_DIR="$OPTARG" ;;
  esac
done
shift $((OPTIND - 1))
(($# >= 1)) || { echo "usage: $0 [-g GPUS] [-l LOG_DIR] -- CMD [ARGS...]" >&2; exit 1; }

# Resolve GPUs: explicit -g, else nvidia-smi, else "0".
if [[ -z "$GPUS" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
fi
GPUS="${GPUS:-0}"
IFS=', ' read -r -a GPU_ARR <<< "$GPUS"

mkdir -p "$LOG_DIR"

# Build prefix from SLURM info when available, else fall back.
node="${SLURMD_NODENAME:-$(hostname -s 2>/dev/null || echo host)}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  job="${SLURM_JOB_ID:-local}"
fi
name="${SLURM_JOB_NAME:-job}"
name="${name//[^A-Za-z0-9._-]/_}"   # sanitize: job names can have spaces/slashes
PREFIX="${name}__${job}__${node}"

declare -a PIDS=()
cleanup() {
  echo "[orch] stopping ${#PIDS[@]} process group(s)..." >&2
  for p in "${PIDS[@]}"; do kill -TERM "-$p" 2>/dev/null || true; done
  sleep 5
  for p in "${PIDS[@]}"; do
    kill -0 "$p" 2>/dev/null && kill -KILL "-$p" 2>/dev/null || true
  done
}
trap cleanup INT TERM

echo "[orch] gpus=(${GPU_ARR[*]}) cmd: $*"
for gpu in "${GPU_ARR[@]}"; do
  log="${LOG_DIR}/${PREFIX}__gpu-${gpu}.log"
  (
    exec </dev/null >"$log" 2>&1
    export CUDA_VISIBLE_DEVICES="$gpu"
    sleep "0.$((RANDOM % 1000))"   # small jitter to avoid thundering herd
    exec setsid "$@"
  ) &
  pid=$!
  PIDS+=("$pid")
  echo "[orch] gpu=$gpu pid=$pid log=$log"
done

for p in "${PIDS[@]}"; do wait "$p" || true; done
