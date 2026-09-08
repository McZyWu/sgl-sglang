# Kimi-K3 verify gates and DSpark folded sampling

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

For the supplied TP32/DP1, BS32, K=V=128 profile, use **BV64**. Across
69 matched target layers, the BV32 run increased recurrent-kernel time from
50.118 to 64.929 microseconds per layer and doubled the grid from 192 to
384 programs. This is an observed stacked-run regression, not an isolated
tile-only benchmark. Updated kernel PR10 removes its forced BV32/two-warp
retile and retains the dense SiTU change; kernel PR3 already defaults to
BV64/one warp. The launcher here now defaults to 64. An explicit exported
`SGLANG_NPU_KDA_VERIFY_VALUE_BLOCK_SIZE=32` still takes precedence, so change
it to 64 in the actual serving script. Keep chunk-KDA precision and canonical
state-layout fixes on both sides of any performance comparison.

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

The earlier raw-gate excerpt is not the user's corrected baseline, which
already uses standalone activation. It cannot establish a speedup for this PR.
Compare against a same-run standalone baseline, then measure full-round time,
acceptance length, and TPOT on the same model, request set, sampling mode, and
software stack.

## Enabled NPU folded proposal and sampling

For the K3 `DSparkDraftModel` with a vanilla Markov head, retain both folds:

```bash
export SGLANG_DSPARK_FOLDED_PROPOSAL=1
export SGLANG_DSPARK_FOLDED_SAMPLING=1
```

On torch_npu 2.10.0/CANN 9.0, the previous `exponential_()` inside the captured
tail fails with `Cannot call ...philox_engine_inputs during NPU graph capture`.
The NPU path now stages independent `[batch, gamma, vocab]` noise outside the
graph immediately before replay. Greedy batches generate no noise. The graph
still performs every Markov step, selects proposals, and synchronizes tokens.
No new environment variable or sampling fallback is required.

Adapted from the first commit of upstream PR34944, stochastic selection uses
`argmax(logits - temperature * log(exponential_noise))`. Greedy rows use direct
logit argmax, preserving near-tie order. A two-pass reduction uses 8192-element
tiles to amortize A3 vector tasks; a 16384-element trial exceeded the A3 192 KiB
UB budget. The partial kernel directly writes gamma-strided corrected logits
for sampling batches, eliminating the intermediate stack and copy. Greedy
batches skip these stores. The public greedy mask remains boolean for mixed
target acceptance; the kernel loads it without per-step casts.

At DP1, each draft batch-size bucket now captures two variants in the same
graph memory pool: native ArgMax for an all-greedy batch, and the existing
two-pass sampler for a mixed or stochastic batch. Staging the request's
`sampling_info.is_all_greedy` selects the appropriate recorded graph before
replay, without a device-to-host read or recapture. Merely branching on that
host flag inside the captured tail would freeze the first sampling mode and
be incorrect on later requests. Both variants retain the same TP sync site,
and graph input updates finish on the selected variant before replay.

This targets the supplied profile's greedy ArgMax+Cast of 13.405 microseconds
versus 34.065 microseconds for the two-pass sampler per Markov step. It does
not establish the new full-graph or TPOT improvement; compare native ArgMax
and two-pass replay using the benchmark below. DP sizes above one retain the
single graph with device sampling inputs, because independent DP batches may
have different sampling modes. CUDA and greedy-only folding are unchanged.

Sampling parameters and noise remain persistent graph inputs. Every Markov
step receives independent noise; one `[batch, vocab]` draw reused across steps
would change the joint proposal distribution. AUTO's memory estimate includes
all gamma noise planes. BS32/gamma7/vocab163840 requires 140 MiB of noise and
70 MiB of BF16 corrected logits, plus capture headroom. DP1 records two graphs
per bucket, increasing capture time and graph metadata/output storage; the
sampling buffers and graph pool are shared. Check peak capture memory with the
full model in addition to replay latency.

After restarting to recapture, look for these startup messages at DP1:

```text
DSpark draft proposal (greedy + sampling) folded into the draft cuda graph.
DSpark NPU folded sampling: per-step noise staged before replay; greedy skips RNG and corrected-logit stores.
DSpark NPU DP1 captures separate greedy ArgMax and mixed sampling graphs for each batch size.
```

In profiling, DP1 greedy replay should use native ArgMax without
`_sample_partial_kernel` / `_sample_combine_kernel`, exponential RNG, or
corrected-logit stack/copy. Mixed/stochastic replay retains the two sampling
kernels and has one noise refresh before the
draft graph, with a distinct plane consumed at each step. Setting an environment
variable alone is not proof that the graph was selected.

If the integration also contains PR29, do not pass
`--speculative-dspark-draft-prefetch` for this comparison: that implementation
skips construction of the folded sampler. The K3 generic vanilla Markov head
also does not read the DSv4 W2 TP-sharding switches below; those exports do not
remove its full-logit AllGather.

Run the native replay test and the isolated tail benchmark:

```bash
PYTHONPATH=python python3 test/registered/unit/npu/speculative/test_npu_dspark_folded_sampling.py
PYTHONPATH=python python3 benchmark/bench_dspark_npu_folded_sampling.py --bs 32 --gamma 7 --vocab 163840
```

The benchmark checks proposal IDs and times captured sampling tails, including
native ArgMax for greedy batches and the original corrected-logit stack/copy
versus the new direct store. Inputs are contiguous per-step logits, matching
the Markov add output. All
tails consume the same precomputed noise; it excludes RNG, Markov/model
computation, communication, and acceptance. The original complete folded
sampler cannot capture on this CANN RNG implementation. These measurements
therefore are not an old-versus-new complete graph or a TPOT measurement.

On a shared A3 device with torch/torch_npu 2.10.0 and CANN 9.0, the updated
benchmark (BS32/gamma7/vocab163840, BF16 contiguous step logits, five
alternating rounds of 30 replays) measured 0.397206 ms for the existing
8192-element two-pass greedy tail and 0.189489 ms for native ArgMax: a
0.207717 ms reduction, or 52.3%. Proposal IDs matched. The stochastic
two-pass tail measured 0.428563 ms and remains the selected stochastic path.
These timings include the token stack but exclude model/Markov computation,
RNG, TP synchronization, and acceptance; no full-model TPOT is implied.

## Speculative scheduler synchronization at DP1

PR31 includes the PR27 fused scheduler-sync prerequisite and repairs its DP1
local finalization. Keep `SGLANG_SPECULATIVE_FUSED_DP_MLP_SYNC=1` when testing
this combination. No additional environment switch is needed for the repair.

Without DP attention, no collective is needed, but both scheduling decisions
must still be published: whether prefill takes priority and whether the decode
probe remains valid. Previously the local path left these at `False` and `True`.
That could discard prefill work or skip a required metadata refresh after a
finished/retracted request. Stale `is_extend_in_batch` then routes a nonempty
DSpark decode batch into prefill and can trigger
`extend-idle conversion expects an empty rank` in hybrid KDA padding.

Local finalization now copies both decisions. DP1 still performs no scheduler
collective, stable decode retains the fused fast path, and an invalid probe
requests the existing post-update metadata refresh. The hybrid empty-rank
assertion and both folded graph modes remain enabled.

## Preserved optional operators

Metadata reuse, the once-per-forward int64 padding mask, and the shared `-1`
state sentinel remain. Cache slot 0 remains valid. PR3 retains its local/global
top1 and ragged input/output/onorm operators, plus fixed-width convolution
coverage. PR31 retains their dispatch and the dense Conv3D option. The supplied
dense/static profiles did not exercise all of these paths; no new TPOT benefit
is claimed for them.

To select the DSv4 TP-sharded fused top1 path on a compatible Markov head:

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
layout. It defaults the V tile to 64, prints the gate settings, and does not
enable experimental gates via `HOTPATH_BUNDLE`. For the supplied
TP32/DP1/block-7 workload, add the flags
above to the original serving script instead of treating that launcher as an
equivalent workload.

## Validation limits

Framework CPU contract tests check both gate dispatch modes, gate precision and
lower-bound placement, all V-tile overrides, metadata reuse, Conv2D/Conv3D,
padding-index sharing, and graph-bucket alignment. Kernel CPU semantic tests
check the actual kernel body with tensor/pointer adapters. Neither test mode
compiles Triton or runs an NPU graph for KDA. The separate folded-sampling NPU
tests compile and capture the actual sampling kernels, verify BS32/gamma7 with
the production vocabulary, greedy/mixed/shrinking batches, independent random
draws across repeated replay, near-tie/tail handling, and selection of native
greedy versus mixed graphs through `NPUCudaGraphBackend`. The four tests passed
on A3 with torch/torch_npu 2.10.0 and CANN 9.0 after the variant change,
including the new backend variant test. Peak allocated memory was 495 MiB
(778 MiB reserved) with a 1.5% per-process allocator limit on a shared device.
The 87 CPU tests also passed: 11 sampler, 3 graph routing/order, 66 KDA
contracts, and 7 bucket alignment cases. These checks use a small
synthetic Markov model, not loaded K3 weights. Full-model acceptance, TPOT, and
four-machine performance still require an isolated serving benchmark.
