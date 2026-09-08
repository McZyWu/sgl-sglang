"""Run the real context-RoPE methods without loading target weights or HCCL."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from sgl_kernel_npu.norm.fused_rope_qk_mqa import fused_rope_qk_mqa

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=20, suite="stage-a-unit-test-npu")


def _method(relative_path, class_name, method_name):
    root = Path(__file__).resolve().parents[5] / "python/sglang/srt"
    tree = ast.parse((root / relative_path).read_text())
    owner = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    scope = {
        "torch": torch,
        "F": F,
        "_is_npu": True,
        "fused_rope_qk_mqa": fused_rope_qk_mqa,
    }
    exec(compile(module, str(root / relative_path), "exec"), scope)
    return scope[method_name]


def _rotary(head_dim, *, neox=True):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angle = torch.outer(torch.arange(512).float(), inv)
    owner = SimpleNamespace(
        head_size=head_dim,
        rotary_dim=head_dim,
        is_neox_style=neox,
        cos_sin_cache=torch.cat((angle.cos(), angle.sin()), -1).to("npu"),
    )
    forward = _method(
        "layers/rotary_embedding/base.py", "RotaryEmbedding", "forward_npu"
    )
    return lambda p, q, k: forward(owner, p, q, k)


class TestDSparkContextRoPE(unittest.TestCase):
    @torch.inference_mode()
    def test_kv_only_flattened_and_strided_graph_replay(self):
        apply = _method("models/dflash.py", "DFlashAttention", "apply_k_rope")
        for heads in (1, 2, 5):
            for neox in (False, True):
                with self.subTest(heads=heads, neox=neox):
                    tokens, dim = 5, 64
                    owner = SimpleNamespace(
                        head_dim=dim, rotary_emb=_rotary(dim, neox=neox)
                    )
                    backing = torch.randn(
                        tokens, heads * dim * 2, dtype=torch.bfloat16, device="npu"
                    )
                    flat = backing[:, : heads * dim]
                    positions = torch.arange(tokens, dtype=torch.int64, device="npu")
                    apply(owner, positions, flat)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        result = apply(owner, positions, flat)
                    for shift in (0, 117):
                        flat.normal_()
                        positions.copy_(torch.arange(tokens, dtype=torch.int64) + shift)
                        graph.replay()
                        shaped = flat.reshape(tokens, heads, dim)
                        _, expected = owner.rotary_emb(
                            positions, torch.empty_like(shaped), shaped
                        )
                        torch.testing.assert_close(
                            result, expected.reshape_as(flat), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            apply(owner, positions, shaped), expected, rtol=0, atol=0
                        )
                    del graph

    @torch.inference_mode()
    def test_stacked_context_matches_per_layer_and_live_inputs(self):
        apply = _method(
            "models/dspark.py", "DSparkDraftMixin", "_project_ctx_kv_stacked"
        )
        for layers in (2, 3, 5):
            with self.subTest(layers=layers):
                tokens, hidden_dim, heads, dim = 5, 128, 2, 64
                kv = heads * dim
                rotary = _rotary(dim)
                attn = SimpleNamespace(
                    kv_size=kv, head_dim=dim, num_kv_heads=heads, rotary_emb=rotary
                )
                owner = SimpleNamespace(
                    layers=[SimpleNamespace(self_attn=attn) for _ in range(layers)]
                )
                hidden = torch.randn(
                    tokens, hidden_dim, dtype=torch.bfloat16, device="npu"
                )
                positions = torch.arange(tokens, dtype=torch.int64, device="npu")
                weights = torch.randn(
                    layers, 2 * kv, hidden_dim, dtype=torch.bfloat16, device="npu"
                )
                norms = torch.randn(layers, dim, device="npu")
                stacked = dict(
                    weight=weights.flatten(0, 1),
                    bias=None,
                    k_norm_weight=norms,
                    eps=1e-6,
                )
                apply(owner, ctx_hidden=hidden, positions=positions, stacked=stacked)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    actual_k, actual_v = apply(
                        owner, ctx_hidden=hidden, positions=positions, stacked=stacked
                    )
                for shift in (0, 123):
                    hidden.normal_()
                    positions.copy_(torch.arange(tokens, dtype=torch.int64) + shift)
                    graph.replay()
                    for layer in range(layers):
                        key, value = F.linear(hidden, weights[layer]).split(kv, -1)
                        key = key.reshape(tokens, heads, dim).float()
                        key *= torch.rsqrt(
                            key.square().mean(-1, keepdim=True) + stacked["eps"]
                        )
                        key = (key * norms[layer]).to(hidden.dtype)
                        _, key = rotary(positions, torch.empty_like(key), key)
                        torch.testing.assert_close(
                            actual_k[layer], key, rtol=0.02, atol=0.02
                        )
                        torch.testing.assert_close(
                            actual_v[layer],
                            value.reshape(tokens, heads, dim),
                            rtol=0.02,
                            atol=0.02,
                        )
                del graph


if __name__ == "__main__":
    unittest.main()
