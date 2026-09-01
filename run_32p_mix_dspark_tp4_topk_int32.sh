#!/usr/bin/env bash

# Reproducible four-node Kimi-K3 DSpark graph launcher for a 2x2 isolation:
#
#   shared-expert TP8 + TopK int64:
#     SHARED_EXPERTS_ATTN_TP_SIZE=8 ENABLE_DEEPEP_TOPK_INT32=0
#   shared-expert TP8 + TopK int32:
#     SHARED_EXPERTS_ATTN_TP_SIZE=8 ENABLE_DEEPEP_TOPK_INT32=1
#   shared-expert TP4 + TopK int64:
#     SHARED_EXPERTS_ATTN_TP_SIZE=4 ENABLE_DEEPEP_TOPK_INT32=0
#   shared-expert TP4 + TopK int32 (historical combined candidate):
#     SHARED_EXPERTS_ATTN_TP_SIZE=4 ENABLE_DEEPEP_TOPK_INT32=1
#   latest-main default control (omit both new options):
#     CASE=default
#
# Run this file once on every node. NODE_RANK can be supplied explicitly; when
# it is omitted, the launcher resolves the rank from NODE_IPS and hostname -I.
# CONFIG_ONLY=1 prints the resolved command without starting the server.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/home/weights/Kimi-K3-DSpark}"
SGL_KERNEL_NPU_ROOT="${SGL_KERNEL_NPU_ROOT:-/home/q00886407/k3-a5/sgl-kernel-npu}"
NODE_IPS="${NODE_IPS:-192.168.25.209,192.168.25.212,192.168.25.216,192.168.25.217}"
DIST_PORT="${DIST_PORT:-15110}"
PORT="${PORT:-15010}"
NET_IFACE="${NET_IFACE:-enp196s0f0}"
TP_SIZE="${TP_SIZE:-64}"
DP_SIZE="${DP_SIZE:-4}"
CP_SIZE="${CP_SIZE:-1}"
DCP_SIZE="${DCP_SIZE:-1}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-15}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
CASE="${CASE:-matrix}"
SHARED_EXPERTS_ATTN_TP_SIZE="${SHARED_EXPERTS_ATTN_TP_SIZE:-4}"
ENABLE_DEEPEP_TOPK_INT32="${ENABLE_DEEPEP_TOPK_INT32:-1}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

IFS=',' read -r -a NODE_IP_ARRAY <<< "${NODE_IPS}"
if (( ${#NODE_IP_ARRAY[@]} != 4 )); then
    echo "NODE_IPS must contain exactly four addresses." >&2
    exit 2
fi
if (( TP_SIZE != 64 || DP_SIZE != 4 || CP_SIZE != 1 || DCP_SIZE != 1 )); then
    echo "This measured launcher requires TP=64, DP=4, CP=1, and DCP=1." >&2
    exit 2
fi
if (( DSPARK_BLOCK_SIZE != 15 )); then
    echo "This isolated launcher requires DSpark block size 15." >&2
    exit 2
fi
if [[ "${CASE}" != "matrix" && "${CASE}" != "default" ]]; then
    echo "CASE must be matrix or default." >&2
    exit 2
fi
if [[ "${CASE}" == "matrix" ]]; then
    if [[ "${SHARED_EXPERTS_ATTN_TP_SIZE}" != "4" && "${SHARED_EXPERTS_ATTN_TP_SIZE}" != "8" ]]; then
        echo "SHARED_EXPERTS_ATTN_TP_SIZE must be 4 or 8." >&2
        exit 2
    fi
    if [[ "${ENABLE_DEEPEP_TOPK_INT32}" != "0" && "${ENABLE_DEEPEP_TOPK_INT32}" != "1" ]]; then
        echo "ENABLE_DEEPEP_TOPK_INT32 must be 0 or 1." >&2
        exit 2
    fi
fi
if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" || "${ENABLE_NPU_GRAPH:-1}" == "0" ]]; then
    echo "This launcher is graph-only; eager decode is not supported." >&2
    exit 2
fi
if [[ "${SGLANG_RAGGED_VERIFY_MODE:-static}" != "static" ]]; then
    echo "Kimi-K3 DSpark requires SGLANG_RAGGED_VERIFY_MODE=static." >&2
    exit 2
fi

GRAPH_BS=(1 2 4 8)
if [[ -n "${CUDA_GRAPH_BS_DECODE:-}" ]]; then
    GRAPH_BS_TEXT="${CUDA_GRAPH_BS_DECODE//,/ }"
    read -r -a REQUESTED_GRAPH_BS <<< "${GRAPH_BS_TEXT}"
    if [[ "${REQUESTED_GRAPH_BS[*]}" != "${GRAPH_BS[*]}" ]]; then
        echo "CUDA_GRAPH_BS_DECODE must resolve to: ${GRAPH_BS[*]}." >&2
        exit 2
    fi
fi

if [[ -z "${NODE_RANK:-}" ]]; then
    LOCAL_IPS=" $(hostname -I 2>/dev/null || true) "
    for rank in 0 1 2 3; do
        if [[ "${LOCAL_IPS}" == *" ${NODE_IP_ARRAY[$rank]} "* ]]; then
            NODE_RANK="${rank}"
            break
        fi
    done
fi
if [[ ! "${NODE_RANK:-}" =~ ^[0-3]$ ]]; then
    echo "Set NODE_RANK to 0, 1, 2, or 3 when the host IP cannot be detected." >&2
    exit 2
fi

for env_script in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh; do
    if [[ "${CONFIG_ONLY}" == "0" && -f "${env_script}" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${env_script}"
        set -u
    fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_PYTHON="${SGL_KERNEL_NPU_ROOT}/python/sgl_kernel_npu"
export PYTHONPATH="${REPO_ROOT}/python:${KERNEL_PYTHON}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${KERNEL_PYTHON}/sgl_kernel_npu/lib:${LD_LIBRARY_PATH:-}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset ASCEND_USE_FIA DEEPEP_HCCL_BUFFSIZE ASCEND_CUSTOM_OPP_PATH
unset ENABLE_PROFILING SGLANG_NPU_FUSED_MOE_MODE
unset SGLANG_K3_SHARED_EXPERTS_ATTN_TP SGLANG_K3_SHARED_EXPERTS_TP_SIZE
unset SGLANG_K3_DENSE_MLP_ATTN_TP SGLANG_NPU_DEEPEP_TOPK_INT32
unset SGLANG_K3_AR_FUSION SGLANG_K3_SP_COLLECTIVE SGLANG_K3_SP_ATTN_RES
unset SGLANG_K3_GEMM_AR SGLANG_K3_RADIX4_TOPK SGLANG_OPT_FUSED_KDA_VERIFY
unset SGLANG_NPU_USE_TRITON_KV_CACHE_STORE SGLANG_NPU_USE_MULTI_STREAM
unset SGLANG_NPU_QUANT_SHARED_AG SGLANG_DSPARK_FUSED_LOCAL_TOP1
unset SGLANG_NPU_FUSED_KDA_VERIFY_GATES SGLANG_NPU_FUSED_KDA_RAGGED_IO
unset SGLANG_NPU_FUSED_KDA_ONORM SGLANG_NPU_REUSE_KDA_VERIFY_METADATA
unset SGLANG_NPU_K3_MERGED_QKVGB SGLANG_NPU_K3_MERGED_QKVGBFA
unset DEEPEP_LOW_LATENCY_COMBINE_INT8
unset SGLANG_SPECULATIVE_FUSED_DP_MLP_SYNC
unset SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH

export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DSPARK_FOLDED_PROPOSAL=1
export SGLANG_DSPARK_FOLDED_SAMPLING=1
export SGLANG_DSPARK_STACKED_CTX_KV=1
export SGLANG_DSPARK_EMBED_IN_GRAPH=1
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_FAST_SAMPLING=1
export SGLANG_DSPARK_OPT_MARKOV_W2_BF16=1
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV=0
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_SOCKET_IFNAME="${NET_IFACE}"
export GLOO_SOCKET_IFNAME="${NET_IFACE}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2000}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-64}"
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-512}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1

EXPERIMENT_ARGS=()
if [[ "${CASE}" == "matrix" ]]; then
    EXPERIMENT_ARGS+=(
        --shared-experts-attn-tp-size "${SHARED_EXPERTS_ATTN_TP_SIZE}"
    )
    if [[ "${ENABLE_DEEPEP_TOPK_INT32}" == "1" ]]; then
        EXPERIMENT_ARGS+=(--enable-deepep-topk-int32)
    fi
    DISPLAY_SHARED_TP="${SHARED_EXPERTS_ATTN_TP_SIZE}"
    DISPLAY_TOPK_INT32="${ENABLE_DEEPEP_TOPK_INT32}"
else
    DISPLAY_SHARED_TP="default-full-attn-tp"
    DISPLAY_TOPK_INT32="default-int64"
fi

SERVER_ARGS=(
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --dist-init-addr "${NODE_IP_ARRAY[0]}:${DIST_PORT}"
    --nnodes 4
    --node-rank "${NODE_RANK}"
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${MODEL_PATH}"
    --trust-remote-code
    --attention-backend ascend
    --device npu
    --quantization modelslim
    --dtype bfloat16
    --tp-size "${TP_SIZE}"
    --enable-dp-attention
    --dp-size "${DP_SIZE}"
    --enable-dp-lm-head
    --enable-shared-experts-attn-tp
    "${EXPERIMENT_ARGS[@]}"
    --enable-dense-mlp-attn-tp
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --reasoning-parser kimi_k3
    --linear-attn-verify-backend triton
    --moe-a2a-backend deepep
    --deepep-mode auto
    --cuda-graph-bs-decode "${GRAPH_BS[@]}"
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
    --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}"
    --speculative-draft-attention-backend ascend
    --speculative-eagle-topk 1
    --speculative-draft-model-quantization unquant
    --watchdog-timeout 9000
    --host 0.0.0.0
    --port "${PORT}"
)

echo "Kimi-K3 DSpark case=${CASE} rank=${NODE_RANK}/4 TP=${TP_SIZE} DP=${DP_SIZE} block=${DSPARK_BLOCK_SIZE} graph=[${GRAPH_BS[*]}] shared_tp=${DISPLAY_SHARED_TP} topk_int32=${DISPLAY_TOPK_INT32}"
if [[ "${CONFIG_ONLY}" == "1" ]]; then
    printf 'python3 -m sglang.launch_server'
    printf ' %q' "${SERVER_ARGS[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${LOG_DIR:-${REPO_ROOT}/logs}"
python3 -m sglang.launch_server "${SERVER_ARGS[@]}" 2>&1 | \
    tee "${LOG_DIR:-${REPO_ROOT}/logs}/k3_dspark_${CASE}_tp${DISPLAY_SHARED_TP}_topk${DISPLAY_TOPK_INT32}_rank${NODE_RANK}_$(date +%Y-%m-%d_%H-%M-%S).log"
