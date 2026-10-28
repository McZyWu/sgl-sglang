#!/usr/bin/env bash
# opt217 Q1 LIM + Copy-SFA. Give decode one round after each prefill batch.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT=${SGLANG_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the GLM-5.2 W4A8 checkpoint}
PREFILL_DECODE_INTERVAL=${PREFILL_DECODE_INTERVAL:-1}
TP_SIZE=${TP_SIZE:-16}
ATTN_DP_SIZE=${ATTN_DP_SIZE:-2}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-18}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-589824}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
CACHE_TOKENS=${CACHE_TOKENS:-8192}
PORT=${PORT:-18888}
GRAPH_BATCHES=${GRAPH_BATCHES:-8 16}

for name in PREFILL_DECODE_INTERVAL TP_SIZE ATTN_DP_SIZE MAX_RUNNING_REQUESTS MAX_TOTAL_TOKENS CONTEXT_LENGTH CHUNKED_PREFILL_SIZE CACHE_TOKENS PORT; do
  value=${!name}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a nonnegative integer, got: $value" >&2
    exit 2
  fi
done
if (( TP_SIZE < 1 || ATTN_DP_SIZE < 1 || TP_SIZE % ATTN_DP_SIZE != 0 || MAX_RUNNING_REQUESTS < ATTN_DP_SIZE )); then
  echo 'Invalid TP, attention DP, or running request capacity' >&2
  exit 2
fi
if (( CACHE_TOKENS < 2048 || CACHE_TOKENS > 32640 || CACHE_TOKENS % 128 != 0 )); then
  echo 'CACHE_TOKENS must be a multiple of 128 in [2048, 32640]' >&2
  exit 2
fi
read -r -a graph_batches <<< "$GRAPH_BATCHES"
for value in "${graph_batches[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid GRAPH_BATCHES' >&2; exit 2; }
done

args=(python3 -m sglang.launch_server
  --model-path "$MODEL_PATH" --served-model-name GLM-5.2-w4a8
  --attention-backend ascend --device npu
  --tp-size "$TP_SIZE" --ep-size "$TP_SIZE" --attn-dp-size "$ATTN_DP_SIZE"
  --nnodes 1 --trust-remote-code --quantization modelslim
  --dtype bfloat16 --kv-cache-dtype bfloat16 --page-size 128
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens 32768
  --prefill-decode-interval "$PREFILL_DECODE_INTERVAL"
  --max-running-requests "$MAX_RUNNING_REQUESTS" --max-total-tokens "$MAX_TOTAL_TOKENS"
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "${MEM_FRACTION_STATIC:-0.88}"
  --cuda-graph-bs-decode "${graph_batches[@]}"
  --moe-a2a-backend deepep --deepep-mode auto --load-balance-method round_robin
  --disable-shared-experts-fusion --disable-radix-cache
  --random-seed 42 --host 127.0.0.1 --port "$PORT")

# Interval 0 is the matched prefill-priority control. Keep chunk size, cache,
# graph and model configuration identical between the two server launches.
# Mixed chunk is deliberately left disabled: fairness uses ordinary Q1 decode.
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi
if (( $# != 0 )); then
  echo 'Use environment variables to configure this launcher; only --dry-run is accepted' >&2
  exit 2
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
unset SGLANG_ENABLE_SPEC_V2 SGLANG_NPU_LIM_LIBRARY VLLM_ASCEND_SRC CUSTOM_OP_ROOT
# Preserve ASCEND_RT_VISIBLE_DEVICES supplied by the device owner.
export PYTHONPATH="${KERNEL_PYTHON_ROOT:?Set KERNEL_PYTHON_ROOT}:$SOURCE_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_SET_CPU_AFFINITY=1 SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export STREAMS_PER_DEVICE=32 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1000} HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600
export DEEPEP_HYBRID_DEPLOYMENT=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1 DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1 SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100
export SGLANG_NPU_PROFILING=0 TRANSFORMERS_VERBOSITY=error
export SGLANG_NPU_SPARSE_KV_INDEXER=lim SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
export SGLANG_NPU_ENABLE_COPY_SFA=1 SGLANG_NPU_SPARSE_KV_CACHE_SIZE="$CACHE_TOKENS"
export SGLANG_NPU_FUSED_LIM_PRE_METADATA=1 SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA=1
export TORCHINDUCTOR_CACHE_DIR="${TASK_ROOT:-/task}/cache/inductor"
export TRITON_CACHE_DIR="${TASK_ROOT:-/task}/cache/triton"
printf 'prefill_decode_interval=%s cache_tokens=%s visible_devices=%s\n' \
  "$PREFILL_DECODE_INTERVAL" "$CACHE_TOKENS" "${ASCEND_RT_VISIBLE_DEVICES:-all}"
printf '%q ' "${args[@]}"
printf '\n'
exec "${args[@]}"
