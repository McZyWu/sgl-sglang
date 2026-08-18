# Qwen3-Next NPU Mega GDN runtime workaround

This note records a runtime/test-only workaround for
`test_npu_qwen3_next_models.py` on the CANN 9.1 / `triton-ascend 3.2.2`
container. It does not require changing SGLang framework source code.

## Background

With `sgl_kernel_npu 2026.6.1`, the default common FLA GDN path enters the
Triton implementation in `wy_fast.py`. On the tested A3 container,
`triton-ascend 3.2.2` can hang during long/variable-length prefill in
`recompute_w_u_fwd_npu`.

`GDN_USE_MEGA_GDN=1` selects the AscendC Mega GDN implementation and avoids
that Triton kernel. Mega GDN uses a fixed chunk size of 128, so SGLang's
Mamba cache and the test expectations must use the same 128-token grid.

## Environment

```bash
export ASCEND_USE_FIA=1
export GDN_USE_MEGA_GDN=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export SGLANG_NPU_USE_MULTI_STREAM=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

Set the visible NPU devices as appropriate for the host. The validation used:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7
```

## Test-only changes

In `test/registered/npu/basic_function/mambacache/test_npu_qwen3_next_models.py`,
change the expected cache chunk size, Mamba tracking interval, and page size to
128:

```diff
 class TestQwen3Next(...):
-    cache_chunk_size = 64
+    cache_chunk_size = 128
@@
         "--mamba-track-interval",
-        "16",
+        "128",
         "--page-size",
-        "16",
+        "128",
```

No `--json-model-override-args` is needed for this variant. In the unmodified
framework, the effective Mamba cache chunk is:

```text
max(model mamba chunk 64, page size 128) = 128
```

The test's `cache_chunk_size` only controls the prefix-cache assertion; it does
not configure the server. `--mamba-track-interval 128` keeps tracking boundaries
aligned with both the 128-token page and Mega GDN chunk.

## Validation

Validation target:

- Container: `sglwmc-813`
- SGLang: community main `53621818`, extracted before the framework patch
- CANN: 9.1.0
- Python: 3.12.13
- `triton-ascend`: 3.2.2
- `sgl_kernel_npu`: 2026.6.1
- NPU devices: 4, 5, 6, 7

The source copy used for validation contains no Mega-GDN-specific changes in
`server_args.py`. All four tests passed:

| Test | Result |
| --- | --- |
| `test_gsm8k` | PASS, score `0.96` |
| `test_input_output_logprobs_match_decode_cache_hit` | PASS, average KL `0.0003437970` |
| `test_input_output_logprobs_match_prefill_cache_hit` | PASS, average KL `0.0004961259` |
| `test_prefix_cache_branching` | PASS, expected/actual cached tokens `256` |

Final result:

```text
Ran 4 tests in 795.812s
OK
```

Full log in the validation container:

```text
/data/wzy/qwen3_next_813_no_framework_json128_full_page128_20260818.log
```

## Trade-off

Page size 128 is a test/runtime workaround with no framework modification, but
it also changes the full-attention KV/Radix-cache page granularity from 16 to
128. This can increase unused tokens in partially filled pages. A framework-side
chunk alignment allows page size 16 while using Mega GDN chunk size 128 and has
less impact on KV-cache granularity.
