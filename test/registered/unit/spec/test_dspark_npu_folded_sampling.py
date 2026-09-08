import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.dspark import dspark_draft_sampling_npu as npu_ops
from sglang.srt.environ import DsparkFoldedSampling, envs
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dspark_components import dspark_draft_sampler as sampler_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _IdentitySync:
    @staticmethod
    def sync(_site, values):
        return values


class _Model:
    def __init__(self, vocab=17, hidden=8):
        self.lm_head = SimpleNamespace(
            weight=torch.randn(vocab, hidden), org_vocab_size=vocab
        )
        self.markov_head = VanillaMarkov(vocab_size=vocab, markov_rank=4)
        self.sample_from_anchor = True

    def compute_base_logits(self, hidden_states):
        return F.linear(hidden_states, self.lm_head.weight), None


def _sampling_info(greedy):
    return SimpleNamespace(
        is_all_greedy=all(greedy),
        top_ks=torch.tensor([1 if value else TOP_K_ALL for value in greedy]),
        temperatures=torch.ones(len(greedy)),
    )


@pytest.fixture
def folded(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_is_npu", True)
    # Execute the real sampler/staging code with CPU tensors and the scalar
    # reference. A separate NPU test exercises the actual compiled kernels.
    monkeypatch.setattr(
        npu_ops, "sample_step_tokens_npu", npu_ops.sample_step_tokens_reference
    )
    model = _Model()
    sampler = sampler_mod.DsparkDraftSampler(
        model=model,
        gamma=3,
        max_bs=4,
        device="cpu",
        tp_sync=_IdentitySync(),
        folded_sampling=True,
    )
    return sampler, model


def test_greedy_stage_and_forward_do_not_generate_noise(folded):
    sampler, _ = folded
    hidden = torch.randn(12, 8)
    ids = torch.zeros(12, dtype=torch.long)
    with patch.object(torch.Tensor, "exponential_", side_effect=AssertionError("RNG")):
        sampler.stage_sampling_params(bs=4, sampling_info=None)
        sampler(hidden, ids)
        sampler.stage_sampling_params(bs=4, sampling_info=_sampling_info([True] * 4))
        sampler(hidden, ids)


def test_stochastic_noise_is_independent_across_steps_and_replays(folded):
    sampler, _ = folded
    sampler.stage_sampling_params(bs=2, sampling_info=_sampling_info([True, False]))
    first = sampler.exp_noise[:2].clone()
    assert first.shape == (2, 3, 17)
    assert not torch.equal(first[:, 0], first[:, 1])
    sampler.stage_sampling_params(bs=2, sampling_info=_sampling_info([True, False]))
    assert not torch.equal(first, sampler.exp_noise[:2])
    assert sampler.exp_noise[:2].is_contiguous()
    assert sampler.exp_noise[:2, 1].stride() == (3 * 17, 1)


def test_mixed_markov_steps_consume_their_own_noise_and_preserve_logits(folded):
    sampler, model = folded
    hidden = torch.randn(12, 8)
    ids = torch.zeros(12, dtype=torch.long)
    sampler.stage_sampling_params(bs=4, sampling_info=_sampling_info([True, False] * 2))
    sampler.exp_noise.fill_(1)
    for step in range(3):
        sampler.exp_noise[1::2, step, step] = 1e-30
    base, _ = model.compute_base_logits(hidden)
    expected_tokens, expected_logits = model.markov_head.sample_block(
        base.view(4, 3, 17),
        first_prev_tokens=ids.view(4, 3)[:, 0],
        hidden_states=hidden.view(4, 3, 8),
        sampler=lambda logits, step: npu_ops.sample_step_tokens_reference(
            step_logits=logits,
            temperatures=sampler.temperatures,
            greedy_mask=sampler.greedy_mask,
            exp_noise=sampler.exp_noise[:, step],
        ),
    )
    with patch.object(
        torch.Tensor, "exponential_", side_effect=AssertionError("graph RNG")
    ):
        sampler(hidden, ids)
    torch.testing.assert_close(sampler.out.view(4, 3), expected_tokens, rtol=0, atol=0)
    torch.testing.assert_close(
        sampler.corrected_out.view(4, 3, 17), expected_logits, rtol=0, atol=0
    )
    assert torch.equal(sampler.out.view(4, 3)[1], torch.tensor([0, 1, 2]))


def test_smaller_batch_clears_stochastic_padding_and_greedy_skips_stores(folded):
    sampler, model = folded
    sampler.stage_sampling_params(bs=4, sampling_info=_sampling_info([False] * 4))
    sampler.stage_sampling_params(bs=2, sampling_info=_sampling_info([True, False]))
    assert sampler.greedy_mask.tolist() == [True, False, True, True]
    assert sampler.greedy_mask.dtype == torch.bool
    sampler.stage_sampling_params(bs=2, sampling_info=None)
    assert sampler.greedy_mask.all()
    assert not sampler.write_corrected_logits.item()
    sampler.corrected_out.fill_(17)
    hidden = torch.randn(12, 8)
    ids = torch.zeros(12, dtype=torch.long)
    sampler(hidden, ids)
    assert (sampler.corrected_out == 17).all()
    base, _ = model.compute_base_logits(hidden)
    expected, _ = model.markov_head.sample_block(
        base.view(4, 3, 17),
        first_prev_tokens=ids.view(4, 3)[:, 0],
        hidden_states=hidden.view(4, 3, 8),
        sampler=lambda logits, _: logits.argmax(-1),
    )
    torch.testing.assert_close(sampler.out.view(4, 3), expected, rtol=0, atol=0)


def test_greedy_near_tie_keeps_direct_logit_order():
    logits = torch.tensor([[0.0, 1e-8, -1.0]])
    actual = npu_ops.sample_step_tokens_reference(
        step_logits=logits,
        temperatures=torch.ones(1),
        greedy_mask=torch.ones(1, dtype=torch.bool),
        exp_noise=torch.ones_like(logits),
    )
    assert actual.item() == 1


@pytest.mark.parametrize(
    "mode", [DsparkFoldedSampling.AUTO, DsparkFoldedSampling.FORCE]
)
def test_enabled_sampling_stays_enabled_on_npu(mode, monkeypatch):
    monkeypatch.setattr(sampler_mod, "_is_npu", True)
    with patch.object(envs.SGLANG_DSPARK_FOLDED_SAMPLING, "get", return_value=mode):
        assert sampler_mod._resolve_folded_sampling(
            model=_Model(),
            gamma=7,
            max_bs=32,
            device="npu",
            tp_rank=0,
            available_memory_gb=2.0,
        )


def test_auto_budget_includes_every_step_noise(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_is_npu", True)
    model = SimpleNamespace(
        lm_head=SimpleNamespace(
            org_vocab_size=163840, weight=torch.empty(0, dtype=torch.bfloat16)
        ),
        markov_head=SimpleNamespace(),
    )
    with patch.object(
        envs.SGLANG_DSPARK_FOLDED_SAMPLING,
        "get",
        return_value=DsparkFoldedSampling.AUTO,
    ):
        assert not sampler_mod._resolve_folded_sampling(
            model=model,
            gamma=7,
            max_bs=32,
            device="npu",
            tp_rank=0,
            available_memory_gb=1.15,
        )
        assert sampler_mod._resolve_folded_sampling(
            model=model,
            gamma=7,
            max_bs=32,
            device="npu",
            tp_rank=0,
            available_memory_gb=1.25,
        )


def test_non_npu_keeps_original_noise_layout_and_sampler(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_is_npu", False)
    sampler = sampler_mod.DsparkDraftSampler(
        model=_Model(),
        gamma=3,
        max_bs=4,
        device="cpu",
        tp_sync=_IdentitySync(),
        folded_sampling=True,
    )
    assert sampler.exp_noise.shape == (4, 17)
    assert sampler.greedy_mask.dtype == torch.bool
    assert sampler.write_corrected_logits is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
