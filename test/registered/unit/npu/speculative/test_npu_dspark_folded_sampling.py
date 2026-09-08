"""Real NPU/NPUGraph coverage for PR31's enabled folded sampler."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.dspark.dspark_draft_sampling_npu import (
    sample_step_tokens_npu,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=35, suite="base-b-test-1-npu-a3")


class _IdentitySync:
    @staticmethod
    def sync(_site, values):
        return values


class TestNpuDsparkFoldedSampling(unittest.TestCase):
    @torch.inference_mode()
    def test_random_noise_refreshes_on_each_replay(self):
        bs, gamma, vocab = 4, 3, 4096
        model = SimpleNamespace(
            lm_head=SimpleNamespace(
                weight=torch.zeros(vocab, 1, device="npu"), org_vocab_size=vocab
            ),
            markov_head=VanillaMarkov(vocab_size=vocab, markov_rank=4).to("npu"),
            sample_from_anchor=True,
        )
        for parameter in model.markov_head.parameters():
            parameter.zero_()
        model.compute_base_logits = lambda hidden: (
            F.linear(hidden, model.lm_head.weight),
            None,
        )
        sampler = DsparkDraftSampler(
            model=model,
            gamma=gamma,
            max_bs=bs,
            device="npu",
            tp_sync=_IdentitySync(),
            folded_sampling=True,
        )
        hidden = torch.zeros(bs * gamma, 1, device="npu")
        ids = torch.zeros(bs * gamma, dtype=torch.long, device="npu")
        sampler(hidden, ids)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            sampler(hidden, ids)
        info = SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.ones(bs, device="npu"),
            top_ks=torch.full((bs,), TOP_K_ALL, device="npu"),
        )
        previous_noise = previous_tokens = None
        for _ in range(3):
            sampler.stage_sampling_params(bs=bs, sampling_info=info)
            noise = sampler.exp_noise.clone()
            expected = noise.argmin(-1)
            graph.replay()
            torch.npu.synchronize()
            tokens = sampler.out.view(bs, gamma).clone()
            torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
            if previous_noise is not None:
                self.assertFalse(torch.equal(noise, previous_noise))
                self.assertFalse(torch.equal(tokens, previous_tokens))
            previous_noise, previous_tokens = noise, tokens

    @torch.inference_mode()
    def test_greedy_near_tie_and_strided_sampling(self):
        device = "npu"
        logits = torch.tensor([[0.0, 1e-8, -1.0], [0.0, 1.0, 2.0]], device=device)
        noise = torch.ones((2, 3, 3), device=device)
        noise[1, 1, 0] = 1e-30
        result = sample_step_tokens_npu(
            step_logits=logits,
            temperatures=torch.ones(2, device=device),
            greedy_mask=torch.tensor([1, 0], dtype=torch.int32, device=device),
            exp_noise=noise[:, 1],
        )
        torch.testing.assert_close(result.cpu(), torch.tensor([1, 0]), rtol=0, atol=0)

        # Equal maxima across tile boundaries select the lowest token id;
        # a single valid element in the final tile must beat masked padding.
        logits = torch.zeros((2, 8193), device=device)
        logits[0, 0] = logits[0, 8192] = 2
        logits[1, 8192] = 3
        result = sample_step_tokens_npu(
            step_logits=logits,
            temperatures=torch.ones(2, device=device),
            greedy_mask=torch.ones(2, dtype=torch.bool, device=device),
            exp_noise=torch.ones_like(logits),
        )
        torch.testing.assert_close(
            result.cpu(), torch.tensor([0, 8192]), rtol=0, atol=0
        )

    @torch.inference_mode()
    def test_capture_once_replay_greedy_mixed_and_smaller_batch(self):
        self._check_capture_replay(use_variants=False)

    @torch.inference_mode()
    def test_backend_selects_greedy_and_mixed_graphs(self):
        self._check_capture_replay(use_variants=True)

    def _check_capture_replay(self, *, use_variants):
        device = "npu"
        # Production vocabulary, BS32 and gamma7 exercise multi-tile reduction,
        # independent step noise, and the strided corrected-logit destinations.
        bs, gamma, vocab, hidden_size = 32, 7, 163840, 8

        class Model:
            def __init__(self):
                self.lm_head = SimpleNamespace(
                    weight=torch.randn(
                        vocab, hidden_size, device=device, dtype=torch.bfloat16
                    ),
                    org_vocab_size=vocab,
                )
                self.markov_head = VanillaMarkov(vocab_size=vocab, markov_rank=4).to(
                    device=device, dtype=torch.bfloat16
                )
                self.sample_from_anchor = True

            def compute_base_logits(self, hidden):
                return F.linear(hidden, self.lm_head.weight), None

        model = Model()
        sampler = DsparkDraftSampler(
            model=model,
            gamma=gamma,
            max_bs=bs,
            device=device,
            tp_sync=_IdentitySync(),
            folded_sampling=True,
        )
        self.assertTrue(sampler._npu_sampling)
        hidden = torch.randn(
            bs * gamma, hidden_size, device=device, dtype=torch.bfloat16
        )
        ids = torch.zeros(bs * gamma, device=device, dtype=torch.long)
        sampler(hidden, ids)
        torch.npu.synchronize()
        if use_variants:
            runner = SimpleNamespace(
                device_module=torch.npu,
                model_runner=SimpleNamespace(
                    tp_group=SimpleNamespace(barrier=lambda: None),
                    npu_graph_variant_provider=sampler,
                ),
            )
            backend = NPUCudaGraphBackend(runner)
            self.addCleanup(backend.cleanup)
            key = ShapeKey(size=bs)
            with backend.capture_session(torch.npu.Stream()):
                backend.capture_one(key, lambda: sampler(hidden, ids))
            self.assertEqual(len(backend._graphs), 2)
            replay = lambda: backend.replay(key, None)
        else:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                sampler(hidden, ids)
            replay = graph.replay
        base, _ = model.compute_base_logits(hidden)
        base = base.view(bs, gamma, vocab)
        anchors = ids.view(bs, gamma)[:, 0]
        hidden_3d = hidden.view(bs, gamma, hidden_size)
        for live_bs, mixed in (
            (32, False),
            (32, True),
            (5, True),
            (5, False),
            (32, True),
        ):
            info = SimpleNamespace(
                is_all_greedy=not mixed,
                temperatures=torch.ones(live_bs, device=device),
                top_ks=torch.tensor(
                    [(TOP_K_ALL if mixed and row % 2 else 1) for row in range(live_bs)],
                    device=device,
                ),
            )
            sampler.stage_sampling_params(bs=live_bs, sampling_info=info)
            if mixed:
                # Distinct forced choices catch reuse of one noise matrix
                # across Markov steps, including the first mixed replay.
                sampler.exp_noise[:live_bs].fill_(1)
                for step in range(gamma):
                    sampler.exp_noise[1:live_bs:2, step, step] = 1e-30
            sampler.corrected_out.fill_(17)
            expected, corrected = model.markov_head.sample_block(
                base,
                first_prev_tokens=anchors,
                hidden_states=hidden_3d,
                sampler=lambda logits, step: (
                    logits.float()
                    - sampler.temperatures[:, None]
                    * torch.where(
                        sampler.greedy_mask.bool()[:, None],
                        1.0,
                        sampler.exp_noise[:, step],
                    ).log()
                ).argmax(-1),
            )
            if use_variants:
                self.assertEqual(
                    sampler.npu_graph_variant, "sampling" if mixed else "greedy"
                )
            replay()
            torch.npu.synchronize()
            torch.testing.assert_close(
                sampler.out.view(bs, gamma), expected, rtol=0, atol=0
            )
            if mixed:
                torch.testing.assert_close(
                    sampler.corrected_out.view(bs, gamma, vocab),
                    corrected,
                    rtol=0,
                    atol=0,
                )
            else:
                self.assertTrue((sampler.corrected_out == 17).all().item())


if __name__ == "__main__":
    unittest.main()
