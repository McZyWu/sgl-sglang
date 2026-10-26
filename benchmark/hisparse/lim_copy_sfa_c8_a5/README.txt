A5 C8 LIM + Copy-SFA integration

For the complete in-container build/install/test/serve workflow, read
MANUAL_CONTAINER_ZH.txt and copy container.env.example outside the checkout.
The entry point is: C8_CONFIG=/path/to/your.env bash container.sh <action>.
Use container.sh gsm8k for the complete 1319-question repository few-shot eval;
GSM8K_FULL_ZH.txt documents data paths, protocol and saved outputs.
Use container.sh perf MODE and perf-compare BASE OPT for serving performance;
PERFORMANCE_ZH.txt gives the two-terminal A/B procedure and metric definitions.
perf-configs generates four configs for 128K/64K/16K input -> 1K output with
C2048/C4096 and matching baselines. Scan client concurrency 2/4/8/16 with
context_length=135168, graph buckets [1,2,4,8,16], and automatic token pools.
The objective is maximum measured whole-model throughput near 50 ms mean TPOT;
the default acceptance ceiling is <60 ms. Each mode may choose its own best
concurrency. OFFLOAD_TRACE_ZH.txt describes optional GSM8K transfer counters;
keep these diagnostics disabled during performance measurements.

Both repositories now use feat/hisparse-lim-copysfa-c8-a5-10-26. The original
feat/hisparse-lim-copysfa-c8-a5 branches are preserved. The new branches merge
opt-217 framework 6e37ddf7 and kernel 0664b242 while retaining the A5 C8 paths.
PORT_OPT_217_20261026.txt records the adaptation and current validation boundary.
Install matching SGLang and Ascend950
sgl-kernel-npu code/wheel together.

A5 execution results reported by the user on 2026-10-24 are recorded in
VALIDATION_A5.txt and validation_a5.json: 267 pytest cases pass with zero skips,
and the fixed-projection framework check passes. Tested revisions are SGLang
375e05bd and kernel 988ad6d. A subsequent user-reported model smoke completed
160 requests / 5120 generated tokens; its console excerpt did not repeat revisions.
The user also reported a full 1319-question GSM8K run at client revision e429072e:
93.858984% accuracy, zero request errors/truncations, 1415.451 s, 93.271 output
tokens/s. This uses the requested legacy 5-shot test-prefix/last-number protocol;
the console does not reprint the server mode/C, and no matched baseline score
or performance speedup is claimed. See the validation records for provenance.
FIX_PREFILL_INDEXER_ABI_20261024.txt records the subsequent startup-warmup crash:
custom_ops overwrites the torch_npu quant-indexer alias. Prefill now explicitly
uses the native torch.ops.npu entry; the kernel wheel remains 988ad6d.

The integration selects TopK over the full causal-visible range. Stable-prefix
sources use the persistent C-token cache; selected tail sources hit an additional
prefix-relative tail region at C+(source-P). The framework reserves 256 slots;
Q1 uses at most one 128-token page. C8 KV is an opaque 656-byte record, while
index query/key are FP8 with real FP32 scales and attention query/output stay BF16.

Initial framework scope:
  - Ascend950/A5, BF16 model compute and --kv-cache-dtype fp8_e4m3;
  - native LIM + Copy-SFA, K=2048, page size 128;
  - Q1, PP=1, DCP=1, prefill CP=1, colocated prefill/decode;
  - --disable-radix-cache, because host KV is owned by each request;
  - C is a multiple of 128 in [2048,32384], plus 256 caller-managed tail slots;
  - aligned physical source capacity, including allocation padding, <=262144.
  - each layer's host/HBM byte-copy extent must fit uint32 addressing (<4 GiB).

Build and install the matching kernel on an Ascend950 build environment:
  bash build.sh -a kernels Ascend950
  python3 -m pip install --no-deps --force-reinstall output/sgl_kernel_npu-*.whl

Run A5 operator and framework validation:
  KERNEL_SOURCE_DIR=/path/to/sgl-kernel-npu \
    bash benchmark/hisparse/lim_copy_sfa_c8_a5/validate.sh

The operator suites include donor TopK/scale/stride/lifecycle checks and packed
CPU attention/payload oracles. The paired suite covers dense/sparse/padding,
tail slots, request reuse and graph replay at C2048/4096/8192. test framework
runs all three capacities, saving framework.json and framework-c4096/8192.json.
check_lim_copy_sfa_c8.py runs the real
SRT manager/registered host buffers and IndexShare payloads with fixed projections.

Launch a server after choosing an A5-compatible model and parallel topology:
  MODEL_PATH=/models/your-model TP_SIZE=8 EP_SIZE=8 ATTN_DP_SIZE=1 \
    LIM_EVAL_MODE=lim_copy_sfa \
    bash benchmark/hisparse/lim_copy_sfa_c8_a5/launch_server.sh

Optional WEIGHT_QUANTIZATION supplies --quantization. Other server arguments may
be appended. Launch the same command with LIM_EVAL_MODE=baseline for ordinary C8
LI/SFA without offload. Both modes use the ordinary Q/K producer (MLAPO disabled)
so the projection path stays the same. The launcher checks hardware/APIs before
allocating model weights. No task-specific container, PID or model path is fixed.

The normal-wheel port retains the donor vector management path; its SIMT
postprocess is disabled until the direct launcher can provide the donor's user-UB
reservation. Operator workspaces, allocator overhead and graph pools still require
headroom outside the persistent sparse-manager budget.

217 is an A3 machine. Its CPU tests, cross-compilation and A3 rejection checks
must be reported separately from A5 numerical/model/performance execution.
This directory does not claim a C8 GPQA score or a fixed-length A/B speedup.

GSM8K_HIT_RATE_ZH.txt explains the persistent cache versus selected KV workspace,
and summarize_offload_trace.py reports token-weighted hit rates from existing
logs or a GSM8K log window, with optional actual request-length analysis.
