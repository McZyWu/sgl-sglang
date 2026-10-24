#!/usr/bin/env bash
# Run inside an existing A5 SGLang container with its matching CANN/torch stack.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$ACTION" == help || "$ACTION" == --help ]]; then
  cat <<'EOF'
export C8_CONFIG=/workspace/c8-a5.env  # copy/edit container.env.example first
bash container.sh build              # selected DeepEP, then C8 kernel wheel
bash container.sh build-kernel       # rebuild/install sgl-kernel-npu only
bash container.sh build-deepep       # rebuild/install DeepEP only
bash container.sh test [STAGE]       # default all; see validate.sh
bash container.sh deepep-test        # optional multi-rank normal/LL alternation
bash container.sh serve [server args...]  # foreground; Ctrl-C to stop this server
bash container.sh smoke              # second terminal, same C8_CONFIG
bash container.sh gsm8k              # full 1319-question few-shot accuracy eval
bash container.sh info               # source revisions and A5 preflight
EOF
  exit 0
fi
case "$ACTION" in
  build|build-kernel|build-deepep|test|deepep-test|serve|smoke|gsm8k|info) ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
: "${C8_CONFIG:?Copy container.env.example and export C8_CONFIG=/path/to/your.env}"
[[ -f "$C8_CONFIG" ]] || { echo "Missing config: $C8_CONFIG" >&2; exit 2; }
set -a
source "$C8_CONFIG"
set +a
source "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f ${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh} ]]; then
  source "${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
fi
set -u
SGLANG_SOURCE_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
export PYTHONPATH="$SGLANG_SOURCE_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export KERNEL_SOURCE_DIR=${KERNEL_SOURCE_DIR:?Set KERNEL_SOURCE_DIR in C8_CONFIG}
KERNEL_SOURCE_DIR=$(cd "$KERNEL_SOURCE_DIR" && pwd)
export MOE_A2A_BACKEND=${MOE_A2A_BACKEND:-none}
export C8_RESULTS_DIR=${C8_RESULTS_DIR:-/tmp/sglang-c8-a5}
mkdir -p "$C8_RESULTS_DIR"
C8_RESULTS_DIR=$(cd "$C8_RESULTS_DIR" && pwd)
RUN_DIR="$C8_RESULTS_DIR/$ACTION-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$RUN_DIR"
trap 'echo "FAILED: $ACTION; see $RUN_DIR" >&2' ERR
cd "$SGLANG_SOURCE_DIR"

logged() { local name=$1; shift; "$@" 2>&1 | tee "$RUN_DIR/$name.log"; }

configure_deepep() {
  [[ "$MOE_A2A_BACKEND" == deepep ]] || return 0
  local package
  package=$(python3 - <<'PY'
from importlib.metadata import distribution
print(distribution("deep_ep").locate_file("deep_ep"))
PY
  )
  [[ -d "$package/vendors/hwcomputing" ]] || {
    echo "DeepEP vendor payload missing; run container.sh build-deepep" >&2; return 2;
  }
  # Prefer the wheel's matching deep_ep_cpp, without a stale top-level symlink.
  export PYTHONPATH="$package:$PYTHONPATH"
  export LD_LIBRARY_PATH="$package/vendors/hwcomputing/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export ASCEND_CUSTOM_OPP_PATH="$package/vendors/hwcomputing${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
  logged deepep-import python3 -c \
    'import deep_ep, deep_ep_cpp; print(deep_ep.__file__); print(deep_ep_cpp.__file__)'
}

build_wheel() {
  local target=$1 pattern=$2 stamp
  stamp=$(mktemp "$RUN_DIR/wheel-start.XXXXXX")
  logged "build-$target" bash -c 'cd "$1" && bash build.sh -a "$2" Ascend950' \
    bash "$KERNEL_SOURCE_DIR" "$target"
  local wheels=()
  mapfile -d '' wheels < <(find "$KERNEL_SOURCE_DIR/output" -maxdepth 1 -type f \
    -name "$pattern" -newer "$stamp" -print0)
  [[ ${#wheels[@]} -eq 1 ]] || {
    echo "Expected exactly one newly built $pattern; found ${#wheels[@]}" >&2; return 2;
  }
  logged "install-$target" python3 -m pip install --no-deps --force-reinstall "${wheels[0]}"
  sha256sum "${wheels[0]}" | tee "$RUN_DIR/$target-wheel.sha256"
  rm -f "$stamp"
}

if [[ "$ACTION" != smoke ]]; then
  {
    for repo in "$SGLANG_SOURCE_DIR" "$KERNEL_SOURCE_DIR"; do
      echo "Repository: $repo"
      git -C "$repo" rev-parse HEAD
      git -C "$repo" status --short
    done
    python3 --version
  } | tee "$RUN_DIR/sources.log"
fi
case "$ACTION" in
  build|build-kernel|build-deepep)
    logged environment python3 "$SCRIPT_DIR/preflight.py" --mode baseline
    [[ -f "$KERNEL_SOURCE_DIR/csrc/a5_c8_registration.cpp" ]] || {
      echo "Kernel checkout lacks A5 C8; use feat/hisparse-lim-copysfa-c8-a5" >&2; exit 2;
    }
    logged submodules git -C "$KERNEL_SOURCE_DIR" submodule update --init --recursive
    if [[ "$ACTION" == build-deepep || ( "$ACTION" == build && "$MOE_A2A_BACKEND" == deepep ) ]]; then
      build_wheel deepep 'deep_ep-*.whl'
    fi
    if [[ "$ACTION" != build-deepep ]]; then
      build_wheel kernels 'sgl_kernel_npu-*.whl'
    fi
    configure_deepep
    if [[ "$ACTION" != build-deepep ]]; then
      logged preflight python3 "$SCRIPT_DIR/preflight.py"
    fi
    ;;
  test)
    export C8_TEST_RESULTS_DIR="$RUN_DIR"
    bash "$SCRIPT_DIR/validate.sh" "${1:-all}"
    ;;
  deepep-test)
    logged environment python3 "$SCRIPT_DIR/preflight.py" --mode baseline
    export MOE_A2A_BACKEND=deepep
    configure_deepep
    export WORLD_SIZE=1 RANK=0
    export DEEPEP_HYBRID_DEPLOYMENT=1
    export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:?Set HCCL_BUFFSIZE in C8_CONFIG}
    logged deepep-alternation timeout --kill-after=30s "${C8_TEST_TIMEOUT:-3600}s" \
      python3 "$KERNEL_SOURCE_DIR/tests/python/deepep/test_normal_and_low_latency.py" \
      --num-processes "${DEEPEP_TEST_RANKS:-${EP_SIZE:-8}}" \
      --normal-num-tokens 256 --low-latency-num-tokens 16 \
      --hidden "${DEEPEP_TEST_HIDDEN:-7168}" --num-experts "${DEEPEP_TEST_EXPERTS:-256}" \
      --num-topk "${DEEPEP_TEST_TOPK:-8}" --test-loop "${DEEPEP_TEST_LOOPS:-10}"
    ;;
  serve)
    configure_deepep
    logged server bash "$SCRIPT_DIR/launch_server.sh" "$@"
    ;;
  smoke)
    long_tokens=${C8_SMOKE_LONG_TOKENS:-$(( ${CACHE_TOKENS:-2048} + 256 ))}
    if (( long_tokens < 4096 )); then long_tokens=4096; fi
    logged liveness python3 "$SCRIPT_DIR/check_liveness.py" \
      --url "${C8_SERVER_URL:-http://127.0.0.1:${PORT:-8810}}" \
      --model "${SERVED_MODEL_NAME:-c8-a5}" --rounds "${C8_SMOKE_ROUNDS:-10}" \
      --concurrency "${C8_SMOKE_CONCURRENCY:-8}" --long-tokens "$long_tokens" \
      --cache-tokens "${CACHE_TOKENS:-2048}" --new-tokens "${C8_SMOKE_NEW_TOKENS:-32}" \
      --timeout "${C8_REQUEST_TIMEOUT:-180}" --output "$RUN_DIR/liveness.json"
    ;;
  gsm8k)
    gsm8k_extra=()
    if [[ -n ${GSM8K_TRAIN_DATA_PATH:-} ]]; then
      gsm8k_extra+=(--train-path "$GSM8K_TRAIN_DATA_PATH")
    fi
    logged gsm8k python3 "$SCRIPT_DIR/evaluate_gsm8k.py" \
      --url "${C8_SERVER_URL:-http://127.0.0.1:${PORT:-8810}}" \
      --data-path "${GSM8K_DATA_PATH:-/data/wzy/test.jsonl}" \
      --num-shots "${GSM8K_NUM_SHOTS:-5}" \
      --max-new-tokens "${GSM8K_MAX_NEW_TOKENS:-512}" \
      --parallel "${GSM8K_PARALLEL:-8}" --output "$RUN_DIR" \
      "${gsm8k_extra[@]}" "$@"
    ;;
  info)
    logged preflight python3 "$SCRIPT_DIR/preflight.py"
    configure_deepep
    ;;
esac
echo "Completed $ACTION; logs: $RUN_DIR"
