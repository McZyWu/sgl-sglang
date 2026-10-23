#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
KERNEL_SOURCE_DIR=${KERNEL_SOURCE_DIR:?Set KERNEL_SOURCE_DIR to the matching sgl-kernel-npu checkout}
python3 "$SCRIPT_DIR/preflight.py"
python3 -m pytest -q \
  "$KERNEL_SOURCE_DIR/tests/test_fused_quant_lightning_indexer_manage_a5.py" \
  "$KERNEL_SOURCE_DIR/tests/test_fused_quant_scatter_copy_sparse_flash_attention.py" \
  "$KERNEL_SOURCE_DIR/tests/test_lim_copy_sfa_c8.py"
python3 "$SCRIPT_DIR/../check_lim_copy_sfa_c8.py" --output "${OUTPUT_JSON:-c8-framework-validation.json}"
