#!/usr/bin/env bash
# Source the verified Ascend environment before importing torch or SGLang.
set -eo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash run.sh prefill|decode [options]; see README.md' >&2
  exit 2
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../../.." && pwd)
cann_env=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
original_pythonpath=${PYTHONPATH:-}
[[ -f "$cann_env" ]] || { echo "CANN environment not found: $cann_env" >&2; exit 2; }
source "$cann_env"
if [[ -n ${ATB_ENV:-} ]]; then source "$ATB_ENV"; fi
export PYTHONPATH="$repo/python${KERNEL_PYTHON_PATH:+:$KERNEL_PYTHON_PATH}${PYTHONPATH:+:$PYTHONPATH}${original_pythonpath:+:$original_pythonpath}"
exec "${PYTHON_BIN:-python3}" "$script_dir/run.py" "$@"
