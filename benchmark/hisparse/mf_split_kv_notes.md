# MF-backed split KV for Ascend fused HiSparse

The fused manager now owns a `MemFabricKVPool`. Its MF SHARED entity reserves
one address space per attention TP rank, with physical DRAM allocated only on
rank 0. Each layer has separate, 2 MiB-aligned CKV (512) and KPE (64) buffers.
Other ranks wrap the broadcast GVAs without allocating another host payload.
The physical token IDs and 128-token page tables remain the native allocator's.

New KV is written with `offload.sparse_copy`. A device broadcast orders TP0's
writes before the other ranks read the shared pool, including captured graphs.
Accepted-token relocation stages all source rows before writing destinations,
preserving overlapping/permuted move semantics for both planes.

PD prefill keeps the native split HBM cache even when the fused offload flags
are inherited. Decode exports MF CKV/KPE GVAs in the same page order as prefill,
followed by device-resident indexer buffers. The existing Ascend transfer engine
therefore sends prefill HBM pages directly into decode DRAM; no full-size
intermediate decode HBM KV allocation is introduced.

Resident HBM KV and Copy-SFA inputs are also split. The matching kernel change
is in `/Users/justin/Documents/Work/sgl-kernel-npu`: host validation and Torch
schema accept two HBM tensors and two DRAM pointers; first-fill and attention
use the split-storage templates. The cube RoPE page reader uses a 64-element
stride rather than inheriting the CKV stride. Rebuild/install this kernel wheel
together with the SGLang change; the Python loader rejects the old ABI.

## Validation in this checkout

- Six isolated CPU numerical/contract checks passed: shared GVA aliases,
  allocation alignment, masked split transfers, overlapping moves, PD page
  offsets, invalid buffers, and MF error propagation (some share a test).
  MF and NPU allocation are simulated at their external boundaries. The local
  harness bypasses serving-only `CustomTestCase` infrastructure.
- Four Copy-SFA Python adapter checks passed.
- The standalone C++ `test_copy_sfa_address_math.cpp` compiled and passed.
- Python AST, selected Ruff rules, and `git diff --check` passed in both repos.
- Full registered-unit collection is unavailable locally: the serving test
  helper requires additional packages (`aiohttp` is the first missing one).
- The full repository registry checker reports the existing unregistered file
  `test/registered/unit/hardware_backend/test_npu_sparse_dispatch.py`.
  Both registration records touched by this change parse successfully.

There is no NPU/CANN environment on this machine, so kernel compilation, real
MF SHARED behavior, HCCL graph ordering, numerical NPU replay, and end-to-end PD
transfer have not been verified. The updated manual manager tests and kernel
fixtures cover the split interface for execution on an Ascend worker.

Relevant checks on an equipped worker:

```bash
# From the SGLang checkout, with the matching kernel wheel installed:
python -m pytest test/registered/unit/hardware_backend/npu/test_memfabric_sparse_kv.py
python -m pytest test/registered/unit/npu/test_sparsity_driven_kv_offload_config.py
python -m pytest test/manual/npu/test_fused_host_kv.py
python -m pytest test/manual/npu/test_lim_short_dynamic.py
python -m pytest test/manual/npu/test_lim_mtp_manager.py

# From the sgl-kernel-npu checkout:
python -m pytest tests/test_copy_sfa_combined.py tests/test_copy_sfa_dynamic_tnd.py \
  tests/test_copy_sfa_resident_refresh.py tests/test_lim_mtp_full_visible.py
```

## Long-only execution

All-long eager batches skip residency staging after MF publication. Decode
graphs capture separate resident/mixed and long-only attention variants. The
long-only variant contains no history/current metadata or unidex staging calls;
NPU replay selects its graph key using CPU sequence lengths. Missing CPU
lengths and unlabelled manual captures retain the general path. MTP committed
CPU lengths are used conservatively as a lower bound.

The existing Copy-SFA metadata kernel invalidates dense resident extents for
sparse requests, so rollback to a short sequence restores history correctly
after sparse attention has overwritten HBM slots. Twelve additional isolated
CPU dispatch tests passed; NPU dual-graph replay and transition validation still
require an Ascend worker.
