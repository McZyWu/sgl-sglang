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

Sampling mode and noise values are persistent graph inputs, so greedy/mixed
transitions and smaller live batches reuse the captured graph. Every Markov
step receives independent noise; one `[batch, vocab]` draw reused across steps
would change the joint proposal distribution. AUTO's memory estimate includes
all gamma noise planes. BS32/gamma7/vocab163840 requires 140 MiB of noise and
70 MiB of BF16 corrected logits, plus capture headroom.

After restarting to recapture, look for both startup messages:

```text
DSpark draft proposal (greedy + sampling) folded into the draft cuda graph.
DSpark NPU folded sampling: per-step noise staged before replay; greedy skips RNG and corrected-logit stores.
```

In profiling, `_sample_partial_kernel` and `_sample_combine_kernel` should
appear in the draft graph. Greedy replay should have no exponential RNG or
corrected-logit stack/copy. Non-greedy replay has one noise refresh before the
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
the original corrected-logit stack/copy versus the new direct store. Both
tails consume the same precomputed noise; it excludes RNG, Markov/model
computation, communication, and acceptance. The original complete folded
sampler cannot capture on this CANN RNG implementation. These measurements
therefore are not an old-versus-new complete graph or a TPOT measurement.

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
layout. It prints the gate settings and does not enable experimental gates
via `HOTPATH_BUNDLE`. For the supplied TP32/DP1/block-7 workload, add the flags
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
draws across repeated replay, and near-tie/tail handling. They use a small
synthetic Markov model, not loaded K3 weights. Full-model acceptance, TPOT, and
four-machine performance still require an isolated serving benchmark.
