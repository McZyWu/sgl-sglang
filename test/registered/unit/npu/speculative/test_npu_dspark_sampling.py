import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.dflash import DFlashAttention
from sglang.srt.models.dspark import DSparkDraftMixin, VanillaMarkov
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=25, suite="stage-a-unit-test-npu")


class TestNpuDsparkSampling(unittest.TestCase):
    def test_greedy_near_tie_uses_logits_argmax(self):
        device = torch.device("npu")
        logits = torch.tensor([[0.0, 1.0e-8, -1.0]], device=device)
        expected = logits.argmax(dim=-1)
        actual = SampleStepTokens.execute(
            step_logits=logits,
            temperatures=torch.ones(1, device=device),
            greedy_mask=torch.ones(1, dtype=torch.bool, device=device),
            exp_noise=torch.ones_like(logits),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_folded_proposal_graph_matches_eager(self):
        device = torch.device("npu")
        batch_size, gamma, hidden_size, vocab_size = 2, 3, 32, 64

        class _Model:
            def __init__(self):
                weight = torch.randn(
                    vocab_size, hidden_size, dtype=torch.bfloat16, device=device
                )
                self.lm_head = SimpleNamespace(
                    weight=weight, org_vocab_size=vocab_size
                )
                self.markov_head = VanillaMarkov(
                    vocab_size=vocab_size, markov_rank=8
                ).to(device=device, dtype=torch.bfloat16)

            def compute_base_logits(self, hidden_states):
                return F.linear(hidden_states, self.lm_head.weight), None

        model = _Model()
        sampler = DsparkDraftSampler(
            model=model,
            gamma=gamma,
            max_bs=batch_size,
            device=device,
            folded_sampling=True,
        )
        hidden = torch.randn(
            batch_size * gamma,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        input_ids = torch.randint(
            vocab_size, (batch_size * gamma,), dtype=torch.int64, device=device
        )
        base_logits, _ = model.compute_base_logits(hidden)
        expected_tokens, expected_logits = model.markov_head.sample_block(
            base_logits.view(batch_size, gamma, vocab_size),
            first_prev_tokens=input_ids.view(batch_size, gamma)[:, 0],
            hidden_states=hidden.view(batch_size, gamma, hidden_size),
            sampler=lambda step_logits, _step_idx: step_logits.argmax(dim=-1),
        )

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            sampler(hidden, input_ids)
        graph.replay()
        torch.npu.synchronize()

        actual_tokens = sampler.out[: batch_size * gamma].view(batch_size, gamma)
        actual_logits = sampler.corrected_out[: batch_size * gamma].view(
            batch_size, gamma, vocab_size
        )
        torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)

    def test_sampling_noise_is_staged_only_for_stochastic_batches(self):
        device = torch.device("npu")
        lm_head = SimpleNamespace(org_vocab_size=32, weight=torch.empty(0))
        model = SimpleNamespace(lm_head=lm_head, markov_head=object())
        sampler = DsparkDraftSampler(
            model=model,
            gamma=2,
            max_bs=2,
            device=device,
            folded_sampling=True,
        )
        sampler.exp_noise.fill_(7.0)

        all_greedy = SimpleNamespace(
            temperatures=torch.ones(2, device=device),
            top_ks=torch.ones(2, dtype=torch.int32, device=device),
            is_all_greedy=True,
        )
        sampler.stage_sampling_params(bs=2, sampling_info=all_greedy)
        self.assertTrue(torch.all(sampler.exp_noise == 7.0).item())

        mixed = SimpleNamespace(
            temperatures=torch.ones(2, device=device),
            top_ks=torch.tensor([1, 8], dtype=torch.int32, device=device),
            is_all_greedy=False,
        )
        sampler.stage_sampling_params(bs=2, sampling_info=mixed)
        self.assertTrue(torch.all(sampler.exp_noise > 0).item())
        self.assertFalse(torch.all(sampler.exp_noise == 7.0).item())

    def test_mixed_rows_match_exponential_race_reference(self):
        device = torch.device("npu")
        generator = torch.Generator(device=device).manual_seed(19)
        logits = torch.randn(4, 5003, device=device, generator=generator)
        temperatures = torch.tensor([0.7, 1.0, 1.3, 0.5], device=device)
        greedy_mask = torch.tensor([True, False, True, False], device=device)
        exp_noise = torch.empty_like(logits).exponential_(1, generator=generator)

        noise = torch.where(greedy_mask[:, None], 1.0, exp_noise)
        expected = (
            logits.float() - temperatures[:, None] * noise.log()
        ).argmax(dim=-1)
        actual = SampleStepTokens.execute(
            step_logits=logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestNpuDsparkOptimizedPaths(unittest.TestCase):
    def test_embedding_graph_matches_eager(self):
        device = torch.device("npu")
        owner = SimpleNamespace(
            embed_tokens=torch.nn.Embedding(
                128, 64, dtype=torch.bfloat16, device=device
            )
        )
        input_ids = torch.arange(12, dtype=torch.int64, device=device)
        expected = DSparkDraftMixin.forward_embed(owner, input_ids)

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = DSparkDraftMixin.forward_embed(owner, input_ids)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_flattened_kv_only_rope_uses_npu_3d_contract(self):
        device = torch.device("npu")
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", device="npu")
        )
        tokens, num_heads, head_dim = 5, 2, 64
        rotary = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=4096,
            base=10000.0,
            is_neox_style=True,
        ).to(device)
        positions = torch.arange(tokens, dtype=torch.int64, device=device)
        flattened = torch.randn(
            tokens,
            num_heads * head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        shaped = flattened.view(tokens, num_heads, head_dim)
        _, expected = rotary(positions, torch.empty_like(shaped), shaped)

        owner = SimpleNamespace(head_dim=head_dim, rotary_emb=rotary)
        actual = DFlashAttention.apply_k_rope(owner, positions, flattened)
        torch.testing.assert_close(
            actual, expected.reshape_as(flattened), rtol=0, atol=0
        )

    def test_stacked_ctx_kv_matches_per_layer(self):
        device = torch.device("npu")
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", device="npu")
        )
        tokens, hidden_size = 5, 128
        num_layers, num_kv_heads, head_dim = 3, 2, 64
        kv_size = num_kv_heads * head_dim
        eps = 1.0e-6
        rotary = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=4096,
            base=10000.0,
            is_neox_style=True,
        ).to(device)

        layers = []
        weights = []
        norm_weights = []
        for _ in range(num_layers):
            weight = torch.randn(
                2 * kv_size,
                hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            k_norm = RMSNorm(head_dim, eps=eps).to(
                device=device, dtype=torch.bfloat16
            )
            attn = SimpleNamespace(
                kv_size=kv_size,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                rotary_emb=rotary,
                k_norm=k_norm,
            )
            layers.append(SimpleNamespace(self_attn=attn))
            weights.append(weight)
            norm_weights.append(k_norm.weight)

        owner = SimpleNamespace(layers=layers)
        stacked = {
            "weight": torch.cat(weights, dim=0),
            "bias": None,
            "k_norm_weight": torch.stack(norm_weights).float(),
            "eps": eps,
        }
        hidden = torch.randn(
            tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
        positions = torch.arange(tokens, dtype=torch.int64, device=device)

        expected_k, expected_v = [], []
        for layer, weight in zip(layers, weights):
            kv = F.linear(hidden, weight)
            k, v = kv.split((kv_size, kv_size), dim=-1)
            k = layer.self_attn.k_norm(k.reshape(-1, head_dim)).view_as(k)
            k = k.view(tokens, num_kv_heads, head_dim)
            dummy_q = torch.empty_like(k)
            _, k = rotary(positions, dummy_q, k)
            expected_k.append(k)
            expected_v.append(v.view(tokens, num_kv_heads, head_dim))

        actual_k, actual_v = DSparkDraftMixin._project_ctx_kv_stacked(
            owner, ctx_hidden=hidden, positions=positions, stacked=stacked
        )
        for layer in range(num_layers):
            torch.testing.assert_close(
                actual_k[layer], expected_k[layer], rtol=2.0e-2, atol=2.0e-2
            )
            torch.testing.assert_close(
                actual_v[layer], expected_v[layer], rtol=2.0e-2, atol=2.0e-2
            )


if __name__ == "__main__":
    unittest.main()
