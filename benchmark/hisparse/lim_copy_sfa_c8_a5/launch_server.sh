#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${LIM_EVAL_MODE:-lim_copy_sfa}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the A5-compatible checkpoint}
TP_SIZE=${TP_SIZE:-1}
EP_SIZE=${EP_SIZE:-$TP_SIZE}
ATTN_DP_SIZE=${ATTN_DP_SIZE:-1}
export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=0
export SGLANG_NPU_ENABLE_COPY_SFA=0
export SGLANG_NPU_SPARSE_KV_INDEXER=legacy
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_NPU_SPARSE_KV_CACHE_SIZE=${CACHE_TOKENS:-2048}
case "$MODE" in
  baseline) ;;
  lim_copy_sfa)
    export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
    export SGLANG_NPU_ENABLE_COPY_SFA=1
    export SGLANG_NPU_SPARSE_KV_INDEXER=lim
    ;;
  *) echo "Unknown LIM_EVAL_MODE=$MODE" >&2; exit 2 ;;
esac
python3 "$SCRIPT_DIR/preflight.py" --mode "$MODE"
extra=()
if [[ -n ${WEIGHT_QUANTIZATION:-} ]]; then
  extra+=(--quantization "$WEIGHT_QUANTIZATION")
fi
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "${SERVED_MODEL_NAME:-c8-a5}" \
  --attention-backend ascend --device npu --dtype bfloat16 --kv-cache-dtype fp8_e4m3 \
  --tp-size "$TP_SIZE" --ep-size "$EP_SIZE" --attn-dp-size "$ATTN_DP_SIZE" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-4}" --cuda-graph-bs-decode "${GRAPH_BATCH_SIZE:-8}" \
  --context-length "${CONTEXT_LENGTH:-36864}" --page-size 128 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-2048}" --max-prefill-tokens "${MAX_PREFILL_TOKENS:-32768}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" --disable-radix-cache \
  --trust-remote-code --random-seed 42 --host "${HOST:-127.0.0.1}" --port "${PORT:-8810}" \
  "${extra[@]}" "$@"
