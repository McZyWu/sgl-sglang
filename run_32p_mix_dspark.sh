#!/usr/bin/env bash

# Reproducible four-node Kimi-K3 DSpark graph launcher for the KDA/DSpark
# hot-path bundle. Run this file once on every node.
#
# HOTPATH_BUNDLE=1 enables all six optimized paths. HOTPATH_BUNDLE=0 keeps the
# same framework/kernel revisions and launch parameters for an isolated A/B.
# NODE_RANK can be supplied explicitly; when omitted, it is resolved from
# NODE_IPS and hostname -I. CONFIG_ONLY=1 prints the command without launching.

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
HOTPATH_BUNDLE="${HOTPATH_BUNDLE:-1}"
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
if [[ "${HOTPATH_BUNDLE}" != "0" && "${HOTPATH_BUNDLE}" != "1" ]]; then
    echo "HOTPATH_BUNDLE must be 0 or 1." >&2
    exit 2
fi
if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" || "${ENABLE_NPU_GRAPH:-1}" == "0" ]]; then
    echo "This launcher is graph-only; eager decode is not supported." >&2
    exit 2
fi
if [[ "${SGLANG_RAGGED_VERIFY_MODE:-static}" != "static" ]]; then
    echo "Kimi-K3 DSpark requires SGLANG_RAGGED_VERIFY_MODE=static." >&2
    exit 2
fi

if [[ -n "${CUDA_GRAPH_BS_DECODE:-}" ]]; then
    GRAPH_BS_TEXT="${CUDA_GRAPH_BS_DECODE//,/ }"
    read -r -a GRAPH_BS <<< "${GRAPH_BS_TEXT}"
elif (( DSPARK_BLOCK_SIZE == 15 )); then
    # The measured optimization changes gamma 7 -> 15 and, with a verify
    # width aligned to attention-TP16, captures true bs=1 replay buckets.
    GRAPH_BS=(1 2 4 8)
elif (( DSPARK_BLOCK_SIZE == 7 )); then
    # B1 is retained by the TP-local generic draft and filtered from target
    # verify when B*verify_width does not meet attention-TP alignment.
    GRAPH_BS=(1 2 4 8 16)
else
    echo "Set CUDA_GRAPH_BS_DECODE explicitly for block size ${DSPARK_BLOCK_SIZE}." >&2
    exit 2
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

if [[ "${CONFIG_ONLY}" == "0" && "${HOTPATH_BUNDLE}" == "1" ]]; then
    for required_kernel_file in \
        "${KERNEL_PYTHON}/sgl_kernel_npu/dspark/top1.py" \
        "${KERNEL_PYTHON}/sgl_kernel_npu/fla/kda_ragged.py"; do
        if [[ ! -f "${required_kernel_file}" ]]; then
            echo "Missing paired pull/1 kernel source: ${required_kernel_file}" >&2
            echo "Point SGL_KERNEL_NPU_ROOT at the matching sgl-kernel-npu pull/1 checkout." >&2
            exit 2
        fi
    done
fi

export PYTHONPATH="${REPO_ROOT}/python:${KERNEL_PYTHON}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${KERNEL_PYTHON}/sgl_kernel_npu/lib:${LD_LIBRARY_PATH:-}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset ASCEND_USE_FIA DEEPEP_HCCL_BUFFSIZE ASCEND_CUSTOM_OPP_PATH
unset ENABLE_PROFILING SGLANG_NPU_FUSED_MOE_MODE
unset SGLANG_K3_SHARED_EXPERTS_ATTN_TP SGLANG_K3_DENSE_MLP_ATTN_TP
unset SGLANG_K3_AR_FUSION SGLANG_K3_SP_COLLECTIVE SGLANG_K3_SP_ATTN_RES
unset SGLANG_K3_GEMM_AR SGLANG_K3_RADIX4_TOPK SGLANG_OPT_FUSED_KDA_VERIFY
unset SGLANG_NPU_USE_TRITON_KV_CACHE_STORE SGLANG_NPU_USE_MULTI_STREAM
unset SGLANG_NPU_QUANT_SHARED_AG
unset SGLANG_DSPARK_FUSED_LOCAL_TOP1
unset SGLANG_NPU_FUSED_KDA_VERIFY_GATES
unset SGLANG_NPU_FUSED_KDA_RAGGED_IO
unset SGLANG_NPU_FUSED_KDA_ONORM
unset SGLANG_NPU_REUSE_KDA_VERIFY_METADATA
unset SGLANG_NPU_KDA_DENSE_CONV3D
unset SGLANG_NPU_K3_MERGED_QKVGB SGLANG_NPU_K3_MERGED_QKVGBFA

if [[ "${HOTPATH_BUNDLE}" == "1" ]]; then
    export SGLANG_DSPARK_FUSED_LOCAL_TOP1=1
    export SGLANG_NPU_FUSED_KDA_VERIFY_GATES=1
    export SGLANG_NPU_FUSED_KDA_RAGGED_IO=1
    export SGLANG_NPU_FUSED_KDA_ONORM=1
    export SGLANG_NPU_REUSE_KDA_VERIFY_METADATA=1
    export SGLANG_NPU_KDA_DENSE_CONV3D=1
fi

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

echo "Kimi-K3 DSpark rank=${NODE_RANK}/4 TP=${TP_SIZE} DP=${DP_SIZE} block=${DSPARK_BLOCK_SIZE} graph=[${GRAPH_BS[*]}] hotpath_bundle=${HOTPATH_BUNDLE}"
if [[ "${CONFIG_ONLY}" == "1" ]]; then
    printf 'python3 -m sglang.launch_server'
    printf ' %q' "${SERVER_ARGS[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${LOG_DIR:-${REPO_ROOT}/logs}"
python3 -m sglang.launch_server "${SERVER_ARGS[@]}" 2>&1 | \
    tee "${LOG_DIR:-${REPO_ROOT}/logs}/k3_dspark_hotpath${HOTPATH_BUNDLE}_rank${NODE_RANK}_$(date +%Y-%m-%d_%H-%M-%S).log"
