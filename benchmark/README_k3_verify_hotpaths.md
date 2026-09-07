# Kimi-K3 verify gate and DSpark top1 A/B

Framework [PR31](https://github.com/zhaozx-cn/sglang/pull/31) is paired with
updated [kernel PR3](https://github.com/zhaozx-cn/sgl-kernel-npu/pull/3). Use both
source checkouts and a kernel library containing the current A5 chunk-KDA
prefill operator. The former PR3 head `fd4e17d` predates that base update and
the new parallel-gate arguments.

## Gate dispatch

The default path activates the log gate in FP32 with `fused_kda_gate_npu`,
activates beta with `dense_b.float().sigmoid()`, and calls target verify with
`gates_are_preactivated=True`. The legacy
`SGLANG_NPU_FUSED_KDA_VERIFY_GATES` setting is ignored by this framework.

The experimental path passes raw gates with `precompute_raw_gates=True` and
`gates_are_preactivated=False`. The kernel activates all verify positions as
a `[next_power_of_2(steps), K]` tile before loading the recurrent state. The
state updates and per-position snapshots remain sequential. The K3 lower bound
is applied exactly once in either mode. This specialization requires at most
16 verify positions and Triton-Ascend `cann.extract_slice`.

| Setting | Default | Meaning |
|---|---|---|
| `SGLANG_NPU_KDA_VERIFY_PARALLEL_GATES` | `0` | `0`: standalone FP32 activation; `1`: experimental token-vectorized activation inside verify |
| `SGLANG_NPU_KDA_VERIFY_VALUE_BLOCK_SIZE` | `0` | `0`: keep kernel default (up to 64); explicit `32`, `64`, or `128`: select the V tile independently of gate mode |

Set the variables identically on every rank **before starting the service**.
Restart between variants so graphs are captured with the intended path.

```bash
# Reference: retain the verified framework dispatch.
export SGLANG_NPU_KDA_VERIFY_PARALLEL_GATES=0
export SGLANG_NPU_KDA_VERIFY_VALUE_BLOCK_SIZE=64

# Candidate: change only this setting, retaining BV=64.
export SGLANG_NPU_KDA_VERIFY_PARALLEL_GATES=1
```

Run the paired kernel's `benchmark/bench_kda_verify_parallel_gates.py` first.
It compares standalone activation, original per-token raw activation, and
token-vectorized activation using the same inputs and revision, including all
gate preparation in the timing. Output and every state snapshot are checked
before and after graph replay. Test BV=32/64/128 separately; a larger tile can
reduce duplicated work but also increase local-memory pressure.

The old profiles measured 59.559 us/layer for separate activation plus verify
and 75.083 us/layer for raw fused verify. These are historical observations,
not measured performance of the new specialization. Compare against the
same-run standalone baseline, then measure full-round time, acceptance length,
and TPOT on the same model, request set, sampling mode, and software stack.

## Preserved optional operators

Metadata reuse, the once-per-forward int64 padding mask, and the shared `-1`
state sentinel remain. Cache slot 0 remains valid. PR3 retains its local/global
top1 and ragged input/output/onorm operators, plus fixed-width convolution
coverage. PR31 retains their dispatch and the dense Conv3D option. The supplied
dense/static profiles did not exercise all of these paths; no new TPOT benefit
is claimed for them.

To select the DSpark fused top1 path:

```bash
export SGLANG_DSPARK_FOLDED_PROPOSAL=1
export SGLANG_DSPARK_FOLDED_SAMPLING=0
export SGLANG_DSPARK_FUSED_LOCAL_TOP1=1
export SGLANG_DSPARK_OPT_MARKOV_W2_BF16=1
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_FP32_LM_HEAD=0
```

`FOLDED_PROPOSAL=0` prevents construction of the sampler containing this
top1 entry point. `FOLDED_SAMPLING=0` selects greedy-only folding; it does not
disable folded proposal. W2 BF16 and TP sharding default to true in this code.
The batch must use greedy sampling and hit a draft graph for the folded result
to be used. `--speculative-eagle-topk 1` is not a replacement for greedy request
sampling; for a controlled greedy test use request `temperature=0`.

Check for `_select_local_top1_after_add_kernel`, candidate AllGather, and
`_select_global_top1_kernel` in the draft sampling region. A startup message
that proposal is folded does not alone prove the specialized top1 was selected.
The target embedding AllReduce is a separate, unchanged operation.

For a top1-only comparison, keep folded proposal on and folded sampling off
on both sides, toggling only `FUSED_LOCAL_TOP1`. Do not attribute changes from
enabling graph folding and top1 together solely to top1.

The repository `run_32p_mix_dspark.sh` still targets its original TP64/DP4
layout. It prints the gate settings and does not enable experimental gates
via `HOTPATH_BUNDLE`. For the supplied TP32/DP1/block-7 workload, add the flags
above to the original serving script instead of treating that launcher as an
equivalent workload.

## Validation limits

Framework CPU contract tests check both dispatch modes, gate precision and
lower-bound placement, all V-tile overrides, metadata reuse, Conv2D/Conv3D,
padding-index sharing, and graph-bucket alignment. Kernel CPU semantic tests
check the actual kernel body with tensor/pointer adapters. Neither test mode
compiles Triton or runs an NPU graph. NPU correctness, compiler memory usage,
acceptance length, and TPOT remain to be measured before changing defaults.
