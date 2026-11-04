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
export PYTHONPATH="$repo/python${KERNEL_PYTHON_PATH:+:$KERNEL_PYTHON_PATH}${MF_PYTHON_PATH:+:$MF_PYTHON_PATH}${PYTHONPATH:+:$PYTHONPATH}${original_pythonpath:+:$original_pythonpath}"
python_bin=${PYTHON_BIN:-python3}
if [[ -n ${MF_PYTHON_PATH:-} ]]; then
  torch_libs=$("$python_bin" - <<'PY'
import importlib.util
from pathlib import Path

paths = []
for name in ("torch", "torch_npu"):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        raise RuntimeError(f"Required package is missing: {name}")
    paths.append(str(Path(spec.origin).parent / "lib"))
print(":".join(paths))
PY
)
  export LD_LIBRARY_PATH="$MF_PYTHON_PATH/memfabric_hybrid/lib:$torch_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$python_bin" "$script_dir/run.py" "$@"
