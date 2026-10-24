#!/usr/bin/env bash
set -eo pipefail
source "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f ${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh} ]]; then
  source "${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
fi
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_SOURCE_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
export PYTHONPATH="$SGLANG_SOURCE_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
KERNEL_SOURCE_DIR=${KERNEL_SOURCE_DIR:?Set KERNEL_SOURCE_DIR to the matching sgl-kernel-npu checkout}
STAGE=${1:-all}
RESULTS=${C8_TEST_RESULTS_DIR:-${C8_RESULTS_DIR:-/tmp/sglang-c8-a5}/test-$(date +%Y%m%d-%H%M%S)-$$}
case "$STAGE" in
  all|cpu|meta|lim|copy-sfa|pair|framework) ;;
  *) echo "Stage must be all, cpu, meta, lim, copy-sfa, pair, or framework" >&2; exit 2 ;;
esac
mkdir -p "$RESULTS"
RESULTS=$(cd "$RESULTS" && pwd)
cd "$SGLANG_SOURCE_DIR"

pytest_stage() {
  local name=$1; shift
  timeout --kill-after=30s "${C8_TEST_TIMEOUT:-3600}s" python3 -m pytest -q -ra \
    --junitxml="$RESULTS/$name.xml" "$@" 2>&1 | tee "$RESULTS/$name.log"
  python3 - "$RESULTS/$name.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

cases = ET.parse(sys.argv[1]).getroot().findall(".//testcase")
bad = [c.attrib.get("name") for c in cases if any(c.find(k) is not None for k in ("skipped", "failure", "error"))]
if not cases or bad:
    raise SystemExit(f"Incomplete A5 validation: cases={len(cases)}, skipped/failed={bad}")
print(f"PASS: {len(cases)} executed cases, zero skips: {sys.argv[1]}")
PY
}

# Fail before the CPU suite when the checkout or installed wheel is stale.
if [[ "$STAGE" != cpu ]]; then
  python3 "$SCRIPT_DIR/preflight.py" --kernel-source "$KERNEL_SOURCE_DIR" 2>&1 | tee "$RESULTS/preflight.log"
fi
if [[ "$STAGE" == all || "$STAGE" == cpu ]]; then
  pytest_stage cpu \
    test/registered/unit/hardware_backend/test_npu_lim_state.py \
    test/manual/npu/test_sparse_kv_offload_cache.py \
    test/registered/unit/kernels/test_kernels_namespace.py
fi
if [[ "$STAGE" == meta ]]; then
  pytest_stage meta "$KERNEL_SOURCE_DIR/tests/test_fused_quant_lightning_indexer_manage_a5.py" -k meta_contract
fi
if [[ "$STAGE" == all || "$STAGE" == lim ]]; then
  pytest_stage quant-lim "$KERNEL_SOURCE_DIR/tests/test_fused_quant_lightning_indexer_manage_a5.py"
fi
if [[ "$STAGE" == all || "$STAGE" == copy-sfa ]]; then
  pytest_stage quant-copy-sfa "$KERNEL_SOURCE_DIR/tests/test_fused_quant_scatter_copy_sparse_flash_attention.py"
fi
if [[ "$STAGE" == all || "$STAGE" == pair ]]; then
  pytest_stage paired-c8 "$KERNEL_SOURCE_DIR/tests/test_lim_copy_sfa_c8.py"
fi
if [[ "$STAGE" == all || "$STAGE" == framework ]]; then
  timeout --kill-after=30s "${C8_TEST_TIMEOUT:-3600}s" python3 \
    "$SCRIPT_DIR/../check_lim_copy_sfa_c8.py" \
    --output "${OUTPUT_JSON:-$RESULTS/framework.json}" 2>&1 | tee "$RESULTS/framework.log"
fi
echo "PASS: $STAGE; results: $RESULTS"
