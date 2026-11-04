#!/usr/bin/env bash
# Reproduce the measured native-KV BS6 case in the prepared 209/217 containers.
set -eo pipefail
role=${1:-}
case "$role" in
  prefill|decode) ;;
  *) echo 'Usage: bash run_native_bs6_209_217.sh prefill|decode' >&2; exit 2 ;;
esac
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=${SPLITKV_WORKSPACE:-/workspace/splitkv-pd-live}
export CANN_ENV=${CANN_ENV:-/usr/local/Ascend/cann-9.1.0/set_env.sh}
export KERNEL_PYTHON_PATH=${KERNEL_PYTHON_PATH:-$workspace/prefix}
export MF_PYTHON_PATH=${MF_PYTHON_PATH:-$workspace/mf-prefix}
if [[ -z ${ATB_ENV:-} && -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  export ATB_ENV=/usr/local/Ascend/nnal/atb/set_env.sh
fi
unset CUDA_VISIBLE_DEVICES TORCH_DEVICE_BACKEND_AUTOLOAD
exec bash "$script_dir/run.sh" "$role" \
  --model "${MODEL_PATH:-/home/weights/GLM-5.2-w4a8}" \
  --prefill-host 192.168.25.209 --decode-host 192.168.25.217 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --interface lo \
  --tp 16 --kv-mode native --bs 6 --graph-bs 8 \
  --max-total-tokens 163840 --mem-fraction 0.92 \
  --output-dir "$workspace/runs"
