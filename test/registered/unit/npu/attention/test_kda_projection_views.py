"""NPU regressions for views produced by the TP32 QKVGBFA projection."""

import unittest

import torch
import torch_npu  # noqa: F401

from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.fla.kda_verify_beta_npu import (
    cast_strided_kda_beta_to_fp32,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=20, suite="base-b-test-1-npu-a3")


class TestKDAProjectionViews(unittest.TestCase):
    @torch.no_grad()
    def test_strided_gate_matches_contiguous_gate_exactly(self):
        for tokens, heads, dim in (
            (1, 3, 128),
            (32, 3, 128),
            (256, 3, 128),
            (257, 5, 64),
        ):
            for activation in ("sigmoid", "swish", "silu"):
                with self.subTest(shape=(tokens, heads, dim), activation=activation):
                    storage = torch.randn(
                        tokens, heads * dim + 1288, device="npu", dtype=torch.bfloat16
                    )
                    gate = storage[:, 17 : 17 + heads * dim].unflatten(-1, (heads, dim))
                    x = torch.randn(
                        1, tokens, heads, dim, device="npu", dtype=torch.bfloat16
                    )
                    weight = torch.randn(dim, device="npu", dtype=torch.bfloat16)
                    expected = rms_norm_gated(
                        x.clone(), gate.contiguous(), weight, None, activation
                    )
                    actual = rms_norm_gated(x.clone(), gate, weight, None, activation)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.no_grad()
    def test_noncontiguous_head_stride_and_residual(self):
        storage = torch.randn(33, 6, 128, device="npu", dtype=torch.bfloat16)
        gate = storage[:, ::2, :]
        x = torch.randn(1, 33, 3, 128, device="npu", dtype=torch.bfloat16)
        residual = torch.randn_like(x, dtype=torch.float32)
        weight = torch.randn(128, device="npu", dtype=torch.bfloat16)
        kwargs = dict(activation="sigmoid", residual=residual, prenorm=True)
        actual = rms_norm_gated(x.clone(), gate, weight, None, **kwargs)
        expected = rms_norm_gated(x.clone(), gate.contiguous(), weight, None, **kwargs)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @torch.no_grad()
    def test_beta_cast_and_sigmoid_are_exact(self):
        for tokens, heads in ((1, 3), (32, 3), (256, 3), (257, 5)):
            storage = torch.randn(tokens, 1672, device="npu", dtype=torch.bfloat16)
            beta = storage[:, 1536 : 1536 + heads].unsqueeze(0)
            cast = cast_strided_kda_beta_to_fp32(beta)
            self.assertTrue(cast.is_contiguous())
            torch.testing.assert_close(cast, beta.float(), rtol=0, atol=0)
            torch.testing.assert_close(
                cast.sigmoid(), beta.float().sigmoid(), rtol=0, atol=0
            )

    @torch.no_grad()
    def test_projection_views_read_new_values_on_graph_replay(self):
        storage = torch.randn(256, 1672, device="npu", dtype=torch.bfloat16)
        beta = storage[:, 1536:1539].unsqueeze(0)
        gate = storage[:, 1152:1536].unflatten(-1, (3, 128))
        source = torch.randn(1, 256, 3, 128, device="npu", dtype=torch.bfloat16)
        x = torch.empty_like(source)
        weight = torch.randn(128, device="npu", dtype=torch.bfloat16)

        def run():
            x.copy_(source)
            return (
                rms_norm_gated(x, gate, weight, None, "sigmoid"),
                cast_strided_kda_beta_to_fp32(beta).sigmoid(),
            )

        run()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual_norm, actual_beta = run()
        for _ in range(3):
            storage.copy_(torch.randn_like(storage))
            source.copy_(torch.randn_like(source))
            expected_norm = rms_norm_gated(
                source.clone(), gate.contiguous(), weight, None, "sigmoid"
            )
            expected_beta = beta.float().sigmoid()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)
            torch.testing.assert_close(actual_beta, expected_beta, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
