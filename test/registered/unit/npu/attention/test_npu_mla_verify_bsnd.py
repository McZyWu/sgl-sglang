"""Exercise the production MLA verify branch with both FIA query layouts."""

import ast
import gc
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="stage-a-unit-test-npu")


def load_verify_call():
    root = Path(__file__).resolve().parents[5]
    source = root / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
    tree = ast.parse(source.read_text())
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.use_fias_v2_bsnd"
        and "FIAS V2 target verify requires" in ast.unparse(node)
    )
    wrapper = ast.parse(
        "def call(self, q_nope, q_rope, c_kv_cache, k_rope_cache, layer, block_table, actual_seq_lengths_kv):\n"
        "    num_query_heads = q_nope.shape[1]\n"
    )
    wrapper.body[0].body.extend(branch.body)
    wrapper.body[0].body.append(
        ast.Return(value=ast.Name(id="attn_output", ctx=ast.Load()))
    )
    ast.fix_missing_locations(wrapper)
    scope = {
        "torch": torch,
        "torch_npu": torch_npu,
        "FULL_ATTENTION_WINDOW": 2147483647,
    }
    exec(compile(wrapper, str(source), "exec"), scope)
    return scope["call"]


def pack_nz(value):
    blocks, heads, page, dim = value.shape
    return (
        value.reshape(blocks, heads, page, dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )


class TestMLAVerifyBSND(unittest.TestCase):
    @torch.inference_mode()
    def test_empty_batch_keeps_flat_output(self):
        call = load_verify_call()
        for enabled in (False, True):
            owner = SimpleNamespace(
                speculative_num_draft_tokens=8, mla_verify_bsnd=enabled
            )
            q = torch.empty(0, 4, 512, device="npu", dtype=torch.bfloat16)
            result = call(owner, q, None, None, None, None, None, [])
            self.assertEqual(result.shape, q.shape)

    @torch.inference_mode()
    def test_paged_layouts_strides_and_live_lengths(self):
        call = load_verify_call()
        for bs in (1, 32):
            for nz in (False, True):
                for strided in (False, True):
                    with self.subTest(bs=bs, nz=nz, strided=strided):
                        self._case(call, bs, nz, strided)
                        gc.collect()
                        torch.npu.empty_cache()

    def _case(self, call, bs, nz, strided):
        h, s, d, r, page = 4, 8, 512, 64, 128
        q_storage = torch.randn(
            bs * s, h, d * (2 if strided else 1), device="npu", dtype=torch.bfloat16
        )
        qr_storage = torch.randn(
            bs * s, h, r * (2 if strided else 1), device="npu", dtype=torch.bfloat16
        )
        q, qr = q_storage[..., :d], qr_storage[..., :r]
        k = torch.randn(16, 1, page, d, dtype=torch.bfloat16)
        kr = torch.randn(16, 1, page, r, dtype=torch.bfloat16)
        if nz:
            k, kr = pack_nz(k), pack_nz(kr)
        k, kr = k.to("npu"), kr.to("npu")
        # Different block-table rows, with shared pages allowed by prefix caching.
        table = (
            torch.stack([torch.arange(16).roll(row % 16) for row in range(bs)])
            .int()
            .to("npu")
        )
        owner = SimpleNamespace(
            speculative_num_draft_tokens=s,
            kv_lora_rank=d,
            qk_rope_head_dim=r,
            page_size=page,
            mtp_mask=torch.ones(2048, 2048, dtype=torch.bool).triu(1).to("npu"),
            mla_verify_bsnd=False,
        )
        layer = SimpleNamespace(tp_k_head_num=1, scaling=(d + r) ** -0.5)
        lengths = [1024] * bs
        graphs, outputs = {}, {}
        for enabled in (False, True):
            owner.mla_verify_bsnd = enabled
            warm = call(owner, q, qr, k, kr, layer, table, lengths)
            torch.npu.synchronize()
            del warm
            torch.npu.empty_cache()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, auto_dispatch_capture=True):
                outputs[enabled] = call(owner, q, qr, k, kr, layer, table, lengths)
            graphs[enabled] = graph
        for update in (0, 1, 2):
            q.normal_()
            qr.normal_()
            table.copy_(
                torch.stack(
                    [torch.arange(16).roll((row + update) % 16) for row in range(bs)]
                ).int()
            )
            lengths = [8 + ((row * 131 + update * 73) % 1024) for row in range(bs)]
            if bs > 1:
                lengths[-1] = 0
            for graph in graphs.values():
                graph.update(cpu_update_input=[{"actual_seq_kvlen": lengths}])
                graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(outputs[True], outputs[False], rtol=0, atol=0)
        del graphs, graph, outputs


if __name__ == "__main__":
    unittest.main()
