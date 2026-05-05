#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <env_name> <algorithm> [extra args passed through to python]
       $0 --env_name=<env_name> --algorithm=<algorithm> [--key=value ...]

Both positional and --flag=value forms are accepted; any unrecognized
--flag=value args (including those injected by a wandb sweep agent) are
appended to the resolved python command.

Environment:
  HYPERPARAMS_FILE   Path to hyperparams.sh (default: ./hyperparameters.sh)
  VENV_PATH          Path to venv activate script (default: ../.venv/bin/activate)
EOF
    exit 1
}

ENV_NAME=""
ALGO=""
PASSTHROUGH=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --env_name=*)  ENV_NAME="${1#*=}" ;;
        --env_name)    shift; ENV_NAME="${1:-}" ;;
        --algorithm=*) ALGO="${1#*=}" ;;
        --algorithm)   shift; ALGO="${1:-}" ;;
        --algo=*)      ALGO="${1#*=}" ;;
        --algo)        shift; ALGO="${1:-}" ;;
        -h|--help)     usage ;;
        --*)           PASSTHROUGH+=("$1") ;;
        *)
            if [ -z "$ENV_NAME" ]; then
                ENV_NAME="$1"
            elif [ -z "$ALGO" ]; then
                ALGO="$1"
            else
                PASSTHROUGH+=("$1")
            fi
            ;;
    esac
    shift
done

if [ -z "$ENV_NAME" ] || [ -z "$ALGO" ]; then
    echo "Error: env_name and algorithm are both required" >&2
    usage
fi

ALGO="${ALGO^^}"

HYPERPARAMS_FILE="${HYPERPARAMS_FILE:-./hyperparameters.sh}"
VENV_PATH="${VENV_PATH:-../.venv/bin/activate}"

if [ ! -f "$HYPERPARAMS_FILE" ]; then
    echo "Error: hyperparams file not found: $HYPERPARAMS_FILE" >&2
    exit 1
fi

# Find a header line matching "# <env_name> (<ALGO>)" and return the next line.
CMD=$(awk -v env="$ENV_NAME" -v algo="$ALGO" '
    BEGIN { target = "# " env " (" algo ")" }
    found { print; exit }
    $0 == target { found = 1 }
' "$HYPERPARAMS_FILE")

if [ -z "$CMD" ]; then
    echo "Error: no entry found for env='$ENV_NAME' algorithm='$ALGO' in $HYPERPARAMS_FILE" >&2
    echo "Header expected: '# $ENV_NAME ($ALGO)'" >&2
    exit 1
fi

unset LD_LIBRARY_PATH

# shellcheck disable=SC1090
source "$VENV_PATH"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

if [ "${#PASSTHROUGH[@]}" -gt 0 ]; then
    EXTRA="$(printf ' %q' "${PASSTHROUGH[@]}")"
    CMD="$CMD$EXTRA"
fi

echo "+ $CMD"
eval "$CMD"
