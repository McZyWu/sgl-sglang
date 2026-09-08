"""Benchmark the production MLA verify layout branch without model weights.

Run from a source checkout on an idle NPU, for example:
    python benchmark/kernels/bench_npu_mla_verify_layout.py --nz --output nz.json
    python benchmark/kernels/bench_npu_mla_verify_layout.py --nz --strided --output nz-strided.json

Uses one shared preallocated workspace for serial fixed-length FIA calls.
The measurements include query/output copies but exclude graph.update and HCCL.
For live-length graph correctness, run test_npu_mla_verify_bsnd.py separately.
"""

import argparse
import faulthandler
import gc
import importlib.util
import json
import runpy
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nz", action="store_true")
    parser.add_argument("--kv", type=int, default=128000)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--strided", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(45, repeat=True)
    torch.npu.set_device(args.device)
    torch.manual_seed(39)
    b, h, s, d, r, page = (32, 4, 8, 512, 64, 128)
    pages = max(1563, (args.kv + page - 1) // page)
    q = torch.randn(
        b, s, h, d * (2 if args.strided else 1), dtype=torch.bfloat16, device="npu"
    )[..., :d]
    qr = torch.randn(
        b, s, h, r * (2 if args.strided else 1), dtype=torch.bfloat16, device="npu"
    )[..., :r]
    k = torch.randn(pages, 1, page, d, dtype=torch.bfloat16)
    kr = torch.randn(pages, 1, page, r, dtype=torch.bfloat16)
    if args.nz:
        k = k.reshape(pages, 1, page, d // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
        kr = kr.reshape(pages, 1, page, r // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    k = k.to("npu")
    kr = kr.to("npu")
    table = torch.arange(pages, dtype=torch.int32).repeat(b, 1).to("npu")
    mask = torch.ones(2048, 2048, dtype=torch.bool).triu(1).to("npu")
    common = dict(
        key_rope=kr,
        num_query_heads=h,
        num_key_value_heads=1,
        softmax_scale=(d + r) ** (-0.5),
        block_table=table,
        block_size=page,
        sparse_mode=3,
        atten_mask=mask,
        actual_seq_qlen=[s] * b,
        actual_seq_kvlen=[args.kv] * b,
        pre_tokens=2147483647,
        next_tokens=0,
    )
    qb = q.transpose(1, 2).contiguous()
    qrb = qr.transpose(1, 2).contiguous()
    torch.npu.synchronize()
    torch.npu.empty_cache()
    workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
        qb, k, k, query_rope=qrb, input_layout="BNSD", **common
    )
    del qb, qrb
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "sglang.test.ci.ci_register", root / "python/sglang/test/ci/ci_register.py"
    )
    ci = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ci
    spec.loader.exec_module(ci)
    tests = runpy.run_path(
        str(root / "test/registered/unit/npu/attention/test_npu_mla_verify_bsnd.py")
    )
    source_call = tests["load_verify_call"]()

    def call_with_workspace(x, key, value, **kwargs):
        out = torch_npu._npu_fused_infer_attention_score_v2_infer_output(
            x,
            value,
            input_layout=kwargs["input_layout"],
            num_query_heads=h,
            num_key_value_heads=1,
            query_rope=kwargs["query_rope"],
            block_table=table,
        )
        return torch_npu.npu_fused_infer_attention_score_v2.out(
            x, key, value, **kwargs, workspace=workspace, out=list(out)
        )

    source_call.__globals__["torch_npu"] = SimpleNamespace(
        npu_fused_infer_attention_score_v2=call_with_workspace
    )
    owner = SimpleNamespace(
        speculative_num_draft_tokens=s,
        kv_lora_rank=d,
        qk_rope_head_dim=r,
        page_size=page,
        mtp_mask=mask,
    )
    layer = SimpleNamespace(tp_k_head_num=1, scaling=(d + r) ** (-0.5))
    graphs = {}
    outputs = {}
    samples = {}
    for layout in ("BNSD", "BSND"):

        def call():
            owner.mla_verify_bsnd = layout == "BSND"
            return source_call(
                owner,
                q.reshape(b * s, h, d),
                qr.reshape(b * s, h, r),
                k,
                kr,
                layer,
                table,
                [args.kv] * b,
            )

        print("WARMUP", layout, flush=True)
        reference = call()
        torch.npu.synchronize()
        print("CAPTURE", layout, flush=True)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(args.layers):
                final = call()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(final, reference, rtol=0, atol=0)
        outputs[layout] = final
        graphs[layout] = graph
        samples[layout] = []
        del reference, call
    torch.testing.assert_close(outputs["BNSD"], outputs["BSND"], rtol=0, atol=0)
    print("CROSS LAYOUT BITWISE EQUAL", flush=True)
    for trial in range(7):
        for layout in ("BNSD", "BSND") if trial % 2 == 0 else ("BSND", "BNSD"):
            graph = graphs[layout]
            graph.replay()
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                graph.replay()
            end.record()
            end.synchronize()
            samples[layout].append(start.elapsed_time(end) / 3)
        print("TRIAL", trial, {k: v[-1] for k, v in samples.items()}, flush=True)
    result = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "kv": args.kv,
        "nz": args.nz,
        "layers": args.layers,
        "pages": pages,
        "source_branch": True,
        "strided_query": args.strided,
        "single_path_benchmark_not_end_to_end": True,
        "fixed_length_excludes_graph_update": True,
        "includes_query_and_output_transpose": True,
        "cross_layout_bitwise_equal": True,
        "samples_ms": samples,
        "median_ms": {k: statistics.median(v) for k, v in samples.items()},
        "peak_tensor_bytes": torch.npu.max_memory_allocated(),
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    del graphs, graph, outputs, final
    gc.collect()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
