# Experimental current-row metadata reuse

Status: design only. This branch contains no implementation of the proposed
reuse and has no device correctness, memory, or performance result for it.

## Branch identity

The experiment uses an independent branch under
`exp/prefillpacked-pd-20261106/`. It does not update the branches of SGLang
PR #47383 or sgl-kernel-npu PR #909.

The implementation baseline is C1 commit
`63115f0d51e18be987212735b74185c73f75e55b`. This document is the only added
content on the C3 design branch. C3 is not an additional implemented or measured
optimization on top of C1.

The companion branches identify separate snapshots:

| Branch suffix | Contents |
| --- | --- |
| `a0-baseline` | Common PD bridge without the original packed-prefill increment. |
| `b0-prefill` | A0 plus the original prefill increment. |
| `c1-sender-indices` | B0 plus the bounded sender index buffer and last-key reuse. |
| `c2-ack-wait` | C1 plus the bounded scheduling-yield candidate for scatter completion. |
| `c3-row-metadata-design` | C1 plus this design document; no C3 runtime code. |

## Goal and existing call chain

The candidate would reduce repeated construction of row metadata in the D-side
non-ordinary-prefill path:

```text
DSA indexer / actual anchor select
  -> LIMSelection stored in ForwardMetadata
  -> fused attention for each layer
  -> manager.offload_v2
       -> fresh combined KV payload for the current layer
       -> row positions, source/destination indices, and token validity
       -> current layer Host publication
  -> CopySFA metadata and combined CopySFA
```

Potentially reusable values are source indices, physical destination indices,
token positions, and the complete token-level validity mask. The current layer
KV payload, Host writes, CopySFA computation, resident state, and required
invalidation remain layer dependent and must still execute.

The proposal is distinct from C1's PD sender index cache and from reusing the
entire CopySFA metadata result. Sharing selected source IDs does not prove that
different layers have identical residency or misses.

## Required correctness boundary

Reuse requires evidence that all consumers belong to the same forward and
anchor, access is serialized, all inputs remain unchanged, and the row-preparation
and consumer streams have the existing valid ordering. A cache key based only
on tensor shape, plan object identity, layer number, or ForwardBatch identity is
insufficient: inputs and shared adapter buffers can change in place.

The metadata must preserve the existing mode-specific query width, request and
position checks, physical out-cache locations, draft acceptance lengths, TP
padding, and destination-capacity bounds. A per-request valid flag cannot replace
the per-token mask. Unknown ownership, missing anchor evidence, reset, failure,
interleaving, or unsupported stream behavior must use the existing preparation.

## Unresolved scope and graph questions

The first layer's indexer runs before its attention function. A scope created
only when that attention function starts therefore cannot label the earlier
select. Rebuilding on the first attention call alone does not establish a fresh
select for anchorless or carried plans, and increasing layer numbers do not
prove that another forward has not interleaved.

Graph warmup, capture, and replay have different Python lifetimes. Capture must
record the actual preparation for each anchor rather than inherit cached
warmup tensors. Replay cannot depend on a Python generation check that does not
run during replay. Multiple graph sizes and anchors must not share stale rows
or outlive their storage. No implementation is approved by this document until
the owner and lifetime boundary is established.

## Validation required before claiming a result

- Compare the complete helper outputs against the existing preparation for
  Q1/Q4, B1/B16, short/long/mixed contexts, cache boundaries, padding, invalid
  slots, physical-location changes, and draft acceptance cases.
- Check two layers with different payloads, consecutive forwards, anchor
  changes, request-slot reset/reuse, cancellation, errors, and in-place inputs.
- Verify current-layer Host contents, CopySFA outputs, and resident state;
  include interleaving and stream ordering, with fallback where unsupported.
- Validate warmup/capture/replay separately before enabling any graph reuse.
- Measure actual allocated/reserved/peak memory, graph-pool lifetime, native KV
  capacity, and device traces. Tensor element counts alone are not a peak-memory
  proof.
- Run matched whole-model accuracy and repeated PD benchmarks before claiming
  a performance gain. Existing A0/C1 evidence does not validate this candidate.

The historical old/new PD results also changed operator interfaces and the
bridge. They are not a controlled attribution of any regression to this proposal.
