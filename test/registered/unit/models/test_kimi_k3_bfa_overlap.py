"""KDA bfa side-stream overlap: forward_qkvbfg_fused must produce outputs
bit-identical to the serial path, both eager and under CUDA graph
capture/replay (the overlap only engages in capture mode)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models.kimi_k3 import (
    KimiK3DeltaAttention,
    KimiK3LinearForCausalLM,
    _get_k3_dense_weight,
    _get_k3_qkvgb_merge_flags,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

_H = 7168
_QKVG = 6144  # q,k,v,g slices per rank at TP8
_N_FA = 128
_N_B = 12
_BFA_W_ROWS = 144  # [f_a | b] padded to 8 rows like _merge_bfa_weights


class TestKimiK3QkvgbMetadata(unittest.TestCase):
    def test_npu_opt_in_flag_matrix(self):
        cases = (
            (False, True, True, True, (False, False)),
            (True, False, False, True, (False, False)),
            (True, False, True, True, (False, False)),
            (True, True, False, True, (True, False)),
            (True, True, True, True, (True, True)),
            (True, True, True, False, (False, False)),
        )
        for is_npu, qkvgb, qkvgbfa, full_rank, expected in cases:
            with (
                self.subTest(
                    is_npu=is_npu,
                    qkvgb=qkvgb,
                    qkvgbfa=qkvgbfa,
                    full_rank=full_rank,
                ),
                patch("sglang.srt.models.kimi_k3._is_npu", is_npu),
                patch.object(
                    envs.SGLANG_NPU_K3_MERGED_QKVGB,
                    "get",
                    return_value=qkvgb,
                ),
                patch.object(
                    envs.SGLANG_NPU_K3_MERGED_QKVGBFA,
                    "get",
                    return_value=qkvgbfa,
                ),
            ):
                self.assertEqual(_get_k3_qkvgb_merge_flags(full_rank), expected)

    def test_text_entry_advertises_fused_qkvg_mapping(self):
        self.assertEqual(
            KimiK3LinearForCausalLM.packed_modules_mapping["fused_qkvg_proj"],
            ["q_proj", "k_proj", "v_proj", "g_proj"],
        )

    def test_text_entry_preserves_existing_quant_mappings(self):
        quant_config = SimpleNamespace(
            packed_modules_mapping={"sentinel": ["existing_proj"]}
        )

        def update(mapping):
            quant_config.packed_modules_mapping = mapping

        quant_config.update_packed_modules_mapping = update
        config = SimpleNamespace(vocab_size=8, hidden_size=4)
        with (
            patch("sglang.srt.models.kimi_k3._is_npu", False),
            patch(
                "sglang.srt.models.kimi_k3.KimiK3LinearModel",
                return_value=torch.nn.Identity(),
            ),
            patch(
                "sglang.srt.models.kimi_k3.get_pp_group",
                return_value=SimpleNamespace(is_last_rank=False),
            ),
            patch(
                "sglang.srt.models.kimi_k3.PPMissingLayer",
                return_value=torch.nn.Identity(),
            ),
            patch(
                "sglang.srt.models.kimi_k3.LogitsProcessor",
                return_value=torch.nn.Identity(),
            ),
        ):
            KimiK3LinearForCausalLM(config, quant_config)

        self.assertEqual(
            quant_config.packed_modules_mapping["sentinel"],
            ["existing_proj"],
        )
        self.assertEqual(
            quant_config.packed_modules_mapping["fused_qkvg_proj"],
            ["q_proj", "k_proj", "v_proj", "g_proj"],
        )

    def test_text_entry_merges_modelslim_nested_mapping(self):
        from sglang.srt.layers.quantization.modelslim.modelslim import (
            ModelSlimConfig,
        )

        quant_config = object.__new__(ModelSlimConfig)
        quant_config.packed_modules_mapping = {
            "model": {"sentinel": ["existing_proj"]},
            "visual": {"qkv_proj": ["qkv"]},
        }
        loader_mutated_class_mapping = {
            "fused_qkvg_proj": ["q_proj", "k_proj", "v_proj", "g_proj"],
            "model": {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
            "visual": {"qkv_proj": ["qkv"]},
        }
        config = SimpleNamespace(vocab_size=8, hidden_size=4)
        with (
            patch("sglang.srt.models.kimi_k3._is_npu", True),
            patch.object(
                KimiK3LinearForCausalLM,
                "packed_modules_mapping",
                loader_mutated_class_mapping,
            ),
            patch(
                "sglang.srt.models.kimi_k3.KimiK3LinearModel",
                return_value=torch.nn.Identity(),
            ),
            patch(
                "sglang.srt.models.kimi_k3.get_pp_group",
                return_value=SimpleNamespace(is_last_rank=False),
            ),
            patch(
                "sglang.srt.models.kimi_k3.PPMissingLayer",
                return_value=torch.nn.Identity(),
            ),
            patch(
                "sglang.srt.models.kimi_k3.LogitsProcessor",
                return_value=torch.nn.Identity(),
            ),
        ):
            KimiK3LinearForCausalLM(config, quant_config)

        self.assertEqual(
            quant_config.packed_modules_mapping["model"]["sentinel"],
            ["existing_proj"],
        )
        self.assertEqual(
            quant_config.packed_modules_mapping["model"]["fused_qkvg_proj"],
            ["q_proj", "k_proj", "v_proj", "g_proj"],
        )
        self.assertEqual(
            quant_config.packed_modules_mapping["visual"]["qkv_proj"],
            ["qkv"],
        )

    def test_qkvgb_merge_is_idempotent(self):
        qkvg = SimpleNamespace(
            weight=torch.nn.Parameter(torch.arange(20).view(4, 5).float())
        )
        beta = SimpleNamespace(
            weight=torch.nn.Parameter(torch.arange(5).view(1, 5).float())
        )
        owner = SimpleNamespace(
            _npu_merged_qkvgb=True,
            _npu_merged_qkvgb_fa=False,
            _qkvgb_w=None,
            _bfa_uses_block_fp8=False,
            fused_qkvg_proj=qkvg,
            b_proj=beta,
        )

        KimiK3DeltaAttention._merge_qkvgb_weights(owner)
        merged = owner._qkvgb_w
        self.assertEqual(tuple(merged.shape), (8, 5))
        torch.testing.assert_close(merged[:4], torch.arange(20).view(4, 5).float())
        torch.testing.assert_close(merged[4], torch.arange(5).float())
        self.assertTrue(torch.equal(merged[5:], torch.zeros(3, 5)))
        self.assertEqual(owner._qkvgb_qkvg_size, 4)
        self.assertEqual(owner._qkvgb_b_size, 1)
        self.assertEqual(owner._qkvgb_fa_size, 0)
        self.assertEqual(
            qkvg.weight.untyped_storage().data_ptr(),
            merged.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            beta.weight.untyped_storage().data_ptr(),
            merged.untyped_storage().data_ptr(),
        )

        KimiK3DeltaAttention._merge_qkvgb_weights(owner)
        self.assertIs(owner._qkvgb_w, merged)
        self.assertEqual(tuple(qkvg.weight.shape), (4, 5))
        self.assertEqual(tuple(beta.weight.shape), (1, 5))

    def test_block_fp8_checkpoint_reload_rebuilds_once(self):
        def fp8_module(rows, value, scale):
            return SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.full(
                        (rows, 4),
                        value,
                        dtype=torch.float8_e4m3fn,
                    ),
                    requires_grad=False,
                ),
                weight_scale_inv=torch.nn.Parameter(
                    torch.full((rows // 2, 2), scale),
                    requires_grad=False,
                ),
                quant_method=SimpleNamespace(weight_block_size=[2, 2]),
                params_dtype=torch.bfloat16,
            )

        qkvg = fp8_module(rows=4, value=1.0, scale=2.0)
        beta = fp8_module(rows=2, value=3.0, scale=4.0)
        owner = SimpleNamespace(
            _npu_merged_qkvgb=True,
            _npu_merged_qkvgb_fa=False,
            _qkvgb_w=None,
            _bfa_uses_block_fp8=True,
            fused_qkvg_proj=qkvg,
            b_proj=beta,
        )

        KimiK3DeltaAttention._merge_qkvgb_weights(owner)
        first = owner._qkvgb_w
        torch.testing.assert_close(
            first[:4], torch.full((4, 4), 2.0, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            first[4:6], torch.full((2, 4), 12.0, dtype=torch.bfloat16)
        )

        # A duplicate post-load rebuilds in place from the retained raw FP8
        # sources; it must not apply block scales to dense values a second
        # time or allocate another serving buffer.
        KimiK3DeltaAttention._merge_qkvgb_weights(owner)
        self.assertIs(owner._qkvgb_w, first)
        torch.testing.assert_close(
            first[:4], torch.full((4, 4), 2.0, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            first[4:6], torch.full((2, 4), 12.0, dtype=torch.bfloat16)
        )

        # A serialized checkpoint reload updates the retained raw parameters;
        # the next post-load rebuilds and dequantizes exactly once.
        qkvg.weight.data.fill_(5.0)
        qkvg.weight_scale_inv.data.fill_(2.0)
        beta.weight.data.fill_(7.0)
        beta.weight_scale_inv.data.fill_(4.0)
        KimiK3DeltaAttention._merge_qkvgb_weights(owner)
        reloaded = owner._qkvgb_w
        self.assertIs(reloaded, first)
        torch.testing.assert_close(
            reloaded[:4], torch.full((4, 4), 10.0, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            reloaded[4:6], torch.full((2, 4), 28.0, dtype=torch.bfloat16)
        )

        # A partial reload is also safe: untouched raw sources retain their
        # previous checkpoint values and scales.
        qkvg.weight.data.fill_(11.0)
        KimiK3DeltaAttention._merge_qkvgb_weights(owner)
        self.assertIs(owner._qkvgb_w, reloaded)
        torch.testing.assert_close(
            reloaded[:4], torch.full((4, 4), 22.0, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            reloaded[4:6], torch.full((2, 4), 28.0, dtype=torch.bfloat16)
        )

    def test_unsupported_quantized_projection_fails_fast(self):
        owner = SimpleNamespace(
            _npu_merged_qkvgb=True,
            _npu_merged_qkvgb_fa=False,
            _qkvgb_w=None,
            _bfa_uses_block_fp8=False,
            prefix="model.layers.0.self_attn",
            fused_qkvg_proj=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.ones((4, 4), dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                ),
                weight_scale=torch.nn.Parameter(
                    torch.ones((1, 1)), requires_grad=False
                ),
            ),
            b_proj=SimpleNamespace(weight=torch.nn.Parameter(torch.ones((1, 4)))),
        )

        with self.assertRaisesRegex(RuntimeError, "Unsupported projection"):
            KimiK3DeltaAttention._merge_qkvgb_weights(owner)


def _make_owner(with_stream: bool):
    gen = torch.Generator(device="cuda").manual_seed(0)

    def _randn(*shape):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
            .mul(0.05)
            .to(torch.bfloat16)
        )

    qkvg_w = _randn(_QKVG, _H)

    def fused_qkvg_proj(x):
        return torch.nn.functional.linear(x, qkvg_w), None

    owner = SimpleNamespace(
        use_full_rank_gate=True,
        _bfa_w=_randn(_BFA_W_ROWS, _H).contiguous(),
        _bfa_f_b_w=_randn(1536, _N_FA).contiguous(),
        _bfa_fa_size=_N_FA,
        _bfa_b_size=_N_B,
        fused_qkvg_proj=fused_qkvg_proj,
        split_sizes=[3 * 1536, 1536],
        _bfa_alt_stream=torch.cuda.Stream() if with_stream else None,
        _bfa_bs_limit=128 if with_stream else 0,
    )
    return owner


def _run(owner, x):
    out = KimiK3DeltaAttention.forward_qkvbfg_fused(owner, x)
    return [t.clone() for t in out]


class TestKimiK3BfaOverlap(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_capture_replay_matches_serial(self):
        torch.manual_seed(0)
        for T in (1, 4, 12):
            with self.subTest(T=T):
                x = (
                    torch.randn(T, _H, device="cuda", dtype=torch.float32)
                    .mul(0.05)
                    .to(torch.bfloat16)
                )
                serial = _run(_make_owner(with_stream=False), x)

                owner = _make_owner(with_stream=True)
                with patch(
                    "sglang.srt.models.kimi_k3.get_is_capture_mode",
                    return_value=True,
                ):
                    # warm up allocations/JIT outside capture
                    _ = _run(owner, x)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = KimiK3DeltaAttention.forward_qkvbfg_fused(owner, x)
                    graph.replay()
                    torch.cuda.synchronize()
                # note: owners share the same seeded weights
                for got, ref, name in zip(
                    captured, serial, ("qkv", "beta", "forget_gate", "g")
                ):
                    self.assertTrue(torch.equal(got, ref), f"T={T} {name} mismatch")

    def test_eager_stream_branch_not_taken(self):
        x = torch.randn(3, _H, device="cuda", dtype=torch.bfloat16)
        serial = _run(_make_owner(with_stream=False), x)
        overlap = _run(_make_owner(with_stream=True), x)  # capture mode False
        for got, ref in zip(overlap, serial):
            self.assertTrue(torch.equal(got, ref))

    def test_block_fp8_weight_is_dequantized_for_tiny_gemm(self):
        module = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.ones((130, 129), device="cuda", dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
            weight_scale_inv=torch.nn.Parameter(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda"),
                requires_grad=False,
            ),
            quant_method=SimpleNamespace(weight_block_size=[128, 128]),
            params_dtype=torch.bfloat16,
        )

        weight = _get_k3_dense_weight(module)

        self.assertEqual(weight.dtype, torch.bfloat16)
        torch.testing.assert_close(
            weight[[0, 0, 128, 128], [0, 128, 0, 128]].float(),
            torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda"),
        )

    def test_per_tensor_fp8_weight_is_not_block_dequantized(self):
        weight = torch.nn.Parameter(
            torch.ones((2, 2), device="cuda", dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        module = SimpleNamespace(
            weight=weight, weight_scale=torch.ones(1, device="cuda")
        )

        self.assertEqual(_get_k3_dense_weight(module).data_ptr(), weight.data_ptr())


if __name__ == "__main__":
    unittest.main()
