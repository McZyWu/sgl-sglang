"""Isolated sampling-tail benchmark; excludes the model and RNG generation.

The original full folded graph cannot capture on this CANN RNG implementation.
This benchmark supplies identical precomputed per-step noise to both tails.
It includes the original corrected-logit stack/copy versus the new fused store.
Shared-device results are diagnostic, not end-to-end performance evidence.
"""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401

from sglang.kernels.ops.speculative.dspark import dspark_draft_sampling_npu as npu_ops
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    sample_step_tokens_triton,
)


@torch.inference_mode()
def main(args):
    torch.npu.set_device(0)
    if args.memory_fraction is not None:
        torch.npu.set_per_process_memory_fraction(args.memory_fraction)
    bs, gamma, vocab = args.bs, args.gamma, args.vocab
    logits = torch.randn(bs, gamma, vocab, device="npu", dtype=torch.bfloat16)
    noise = torch.empty(bs, gamma, vocab, device="npu").exponential_()
    original_noise = [noise[:, step].contiguous() for step in range(gamma)]
    temperatures = torch.ones(bs, device="npu")
    mask = torch.ones(bs, dtype=torch.bool, device="npu")
    corrected = torch.empty_like(logits)
    write_corrected = torch.zeros((), dtype=torch.int32, device="npu")

    def original():
        tokens = [
            sample_step_tokens_triton(
                step_logits=logits[:, step],
                temperatures=temperatures,
                greedy_mask=mask,
                exp_noise=original_noise[step],
            )
            for step in range(gamma)
        ]
        corrected.copy_(torch.stack([logits[:, step] for step in range(gamma)], dim=1))
        return torch.stack(tokens, dim=1)

    def optimized():
        return torch.stack(
            [
                npu_ops.sample_step_tokens_npu(
                    step_logits=logits[:, step],
                    temperatures=temperatures,
                    greedy_mask=mask,
                    exp_noise=noise[:, step],
                    corrected_logits_out=corrected[:, step],
                    write_corrected_logits=write_corrected,
                )
                for step in range(gamma)
            ],
            dim=1,
        )

    def capture(fn):
        fn()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = fn()
        return graph, output

    def measure(graph):
        repeats = args.repeats
        for _ in range(5):
            graph.replay()
        torch.npu.synchronize()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
            enable_timing=True
        )
        begin.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) / repeats

    baseline, before = capture(original)
    variants = {}
    for tile in args.tiles:
        npu_ops._BLOCK_V = tile
        variants[tile] = capture(optimized)
    report = {
        "bs": bs,
        "gamma": gamma,
        "vocab": vocab,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "scope": "sampling tail, precomputed noise; no model or RNG timing",
        "cases": {},
    }
    for greedy in (True, False):
        mask.fill_(int(greedy))
        write_corrected.fill_(int(not greedy))
        baseline.replay()
        torch.npu.synchronize()
        samples = {"original": []} | {str(tile): [] for tile in variants}
        for tile, (graph, after) in variants.items():
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        for round_idx in range(args.rounds):
            pairs = [("original", baseline)] + [
                (str(t), g) for t, (g, _) in variants.items()
            ]
            if round_idx % 2:
                pairs.reverse()
            for label, graph in pairs:
                samples[label].append(measure(graph))
        report["cases"]["greedy" if greedy else "random"] = {
            label: {"median_ms": statistics.median(values), "samples_ms": values}
            for label, values in samples.items()
        }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--gamma", type=int, default=7)
    parser.add_argument("--vocab", type=int, default=163840)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--tiles", type=int, nargs="+", default=[8192])
    parser.add_argument("--memory-fraction", type=float, default=None)
    args = parser.parse_args()
    if min(args.bs, args.gamma, args.vocab, args.repeats, args.rounds) < 1:
        parser.error("dimensions and repeat counts must be positive")
    if any(tile < 1 or tile & (tile - 1) for tile in args.tiles):
        parser.error("tiles must be positive powers of two")
    main(args)
