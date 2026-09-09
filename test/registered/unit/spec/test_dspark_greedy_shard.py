import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.dspark import dspark_greedy_top1_npu as ops
from sglang.srt.models import dspark
from sglang.srt.speculative.dspark_components import dspark_draft_sampler as sampler_mod
from sglang.srt.speculative.dspark_components.dspark_greedy_shard import (
    VanillaMarkovGreedyShard,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def local_reference(base, bias, *, vocab_offset):
    values, ids = (base + bias).max(-1)
    return torch.stack((values.float(), (ids + vocab_offset).float()), -1)


def global_reference(candidates, *, vocab_size):
    values, ids = candidates.unbind(-1)
    best = values.max(-1, keepdim=True).values
    return torch.where(values == best, ids, vocab_size).min(-1).values.long()


def make_lm(vocab, width, rank, tp):
    return SimpleNamespace(
        org_vocab_size=vocab,
        tp_size=tp,
        num_embeddings_padded=width * tp,
        num_embeddings_per_partition=width,
        weight=torch.zeros(width, 8, dtype=torch.bfloat16),
        shard_indices=SimpleNamespace(
            org_vocab_start_index=min(rank * width, vocab),
            org_vocab_end_index=min((rank + 1) * width, vocab),
        ),
    )


@pytest.mark.parametrize("vocab,tp,width", [(29, 3, 10), (32, 4, 8), (17, 1, 17)])
def test_sharded_markov_chain_matches_full_bf16_and_exchanges_only_pairs(
    monkeypatch, vocab, tp, width
):
    monkeypatch.setattr(ops, "select_vanilla_local_top1_npu", local_reference)
    monkeypatch.setattr(ops, "select_vanilla_global_top1_npu", global_reference)
    torch.manual_seed(13)
    head = dspark.VanillaMarkov(vocab_size=vocab, markov_rank=4).bfloat16()
    bs, gamma = 3, 7
    base = torch.randn(bs, gamma, vocab, dtype=torch.bfloat16)
    anchor = torch.arange(bs)
    expected, full = head.sample_block(
        base,
        first_prev_tokens=anchor,
        hidden_states=None,
        sampler=lambda logits, step: logits.argmax(-1),
    )
    for rank in range(tp):

        def gather(local, dim, *, rank=rank):
            assert dim == 1 and local.shape == (bs, 2)
            step = gather.step
            gather.step += 1
            pairs = []
            for r in range(tp):
                logits = full[:, step, r * width : min((r + 1) * width, vocab)]
                values, ids = logits.max(-1)
                pairs.append(
                    torch.stack((values.float(), (ids + r * width).float()), -1)
                )
            torch.testing.assert_close(local, pairs[rank], rtol=0, atol=0)
            return torch.cat(pairs, dim=1)

        gather.step = 0
        group = SimpleNamespace(world_size=tp, rank_in_group=rank, all_gather=gather)
        shard = VanillaMarkovGreedyShard.create(
            head, make_lm(vocab, width, rank, tp), group
        )
        local = torch.full((bs, gamma, width), 10000, dtype=torch.bfloat16)
        valid = min(width, vocab - rank * width)
        local[..., :valid] = base[..., rank * width : rank * width + valid]
        sync = Mock(side_effect=lambda tokens: tokens)
        actual = shard.sample(local, first_prev_tokens=anchor, sync=sync)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert sync.call_count == gamma
        assert gather.step == (gamma if tp > 1 else 0)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_empty_vocab_ranks_disable_the_path_on_every_rank(rank):
    head = dspark.VanillaMarkov(vocab_size=8, markov_rank=4).bfloat16()
    group = SimpleNamespace(world_size=3, rank_in_group=rank)
    assert VanillaMarkovGreedyShard.create(head, make_lm(8, 8, rank, 3), group) is None


@pytest.mark.parametrize(
    "attribute", ["weight_dtype", "partition", "vocab", "added_vocab"]
)
def test_incompatible_layout_or_precision_retains_full_path(attribute):
    head = dspark.VanillaMarkov(vocab_size=16, markov_rank=4).bfloat16()
    lm = make_lm(16, 8, 0, 2)
    if attribute == "weight_dtype":
        head.float()
    elif attribute == "partition":
        lm.shard_indices.org_vocab_end_index = 7
    elif attribute == "added_vocab":
        lm.num_embeddings = 17
    else:
        lm.org_vocab_size = 15
    assert (
        VanillaMarkovGreedyShard.create(
            head, lm, SimpleNamespace(world_size=2, rank_in_group=0)
        )
        is None
    )


def test_folded_greedy_uses_shard_but_sampling_keeps_full_logits(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_is_npu", True)
    from sglang.kernels.ops.speculative.dspark import (
        dspark_draft_sampling_npu as sampling_ops,
    )

    monkeypatch.setattr(
        sampling_ops,
        "sample_step_tokens_npu",
        sampling_ops.sample_step_tokens_reference,
    )
    head = dspark.VanillaMarkov(vocab_size=17, markov_rank=4).bfloat16()
    lm = make_lm(17, 17, 0, 1)
    calls = []

    def greedy(hidden, *, first_prev_tokens, sync):
        calls.append("sharded")
        return torch.stack([sync(first_prev_tokens) for _ in range(3)], 1)

    def full(hidden):
        calls.append("full")
        return F.linear(hidden, lm.weight), None

    model = SimpleNamespace(
        markov_head=head,
        lm_head=lm,
        sample_from_anchor=True,
        compute_greedy_proposal=greedy,
        compute_base_logits=full,
    )
    sync = Mock()
    sync.sync.side_effect = lambda site, tokens: tokens
    sampler = sampler_mod.DsparkDraftSampler(
        model=model, gamma=3, max_bs=2, device="cpu", tp_sync=sync
    )
    hidden = torch.zeros(6, 8, dtype=torch.bfloat16)
    ids = torch.zeros(6, dtype=torch.long)
    for variant in ("greedy", "sampling", "greedy"):
        with sampler.npu_graph_capture_variant(variant):
            sampler(hidden, ids)
    assert calls == ["sharded", "full", "sharded"]
    assert all(
        call.args[0] == SpecTpSyncSite.DSPARK_GRAPH_SAMPLE
        for call in sync.sync.call_args_list
    )
    assert sync.sync.call_count == 9


def test_dflash_npu_reuses_the_one_qkv_projection(monkeypatch):
    from sglang.srt.models import dflash

    monkeypatch.setattr(dflash, "_is_npu", True)
    monkeypatch.setattr(
        dflash,
        "split_qkv_rmsnorm_rope",
        lambda qkv, *args, **kwargs: qkv.chunk(3, -1),
        raising=False,
    )
    projected = torch.randn(4, 6)
    rotary = SimpleNamespace(
        position_sin=None, position_cos=None, get_cos_sin_with_position=Mock()
    )
    attn = SimpleNamespace(
        qkv_proj=Mock(return_value=(projected, None)),
        rotary_emb=rotary,
        q_size=2,
        kv_size=2,
        head_dim=2,
        q_norm=SimpleNamespace(variance_epsilon=1e-6, weight=None),
        k_norm=SimpleNamespace(weight=None),
        attention_sink_bias=None,
        o_proj=Mock(side_effect=lambda x: (x, None)),
        apply_attention_output=lambda x, hidden: x,
    )
    attention = Mock(side_effect=lambda q, k, v, batch: q + k + v)
    attention.layer_id = 0
    attn.attn = attention
    attn.forward_prepare_npu = (
        lambda positions, qkv: dflash.DFlashAttention.forward_prepare_npu(
            attn, positions, qkv
        )
    )
    output = dflash.DFlashAttention.forward(
        attn, torch.arange(4), torch.randn(4, 8), None
    )
    attn.qkv_proj.assert_called_once()
    rotary.get_cos_sin_with_position.assert_called_once()
    torch.testing.assert_close(output, sum(projected.chunk(3, -1)))


@pytest.mark.parametrize("disabled", [None, "shard", "top1", "quant", "dp", "fp32"])
def test_generic_model_attach_activates_only_supported_opt_in_path(
    monkeypatch, disabled
):
    monkeypatch.setattr(dspark, "_is_npu", True)
    monkeypatch.setenv(
        "SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD", "0" if disabled == "shard" else "1"
    )
    monkeypatch.setenv(
        "SGLANG_DSPARK_FUSED_LOCAL_TOP1", "0" if disabled == "top1" else "1"
    )
    monkeypatch.setattr(
        dspark, "should_apply_lm_head_quant_method", lambda *args: disabled == "quant"
    )
    group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(
        dspark,
        "get_parallel",
        lambda: SimpleNamespace(
            tp_group=group, attn_dp_size=2 if disabled == "dp" else 1, attn_cp_size=1
        ),
    )
    model = SimpleNamespace(
        is_nemotron_35_draft=False,
        markov_head=dspark.VanillaMarkov(vocab_size=16, markov_rank=4).bfloat16(),
        logits_mup_width_multiplier=None,
    )
    lm = make_lm(16, 8, 0, 2)
    lm.quant_method = None
    if disabled == "fp32":
        model.markov_head.float()
    dspark.DSparkDraftMixin.attach_shared_modules(model, embed_tokens=None, lm_head=lm)
    assert (model._npu_greedy_shard is not None) == (disabled is None)
    if disabled is None:
        result = torch.ones(2, 3, dtype=torch.long)
        model._npu_greedy_shard.sample = Mock(return_value=result)
        hidden = torch.randn(6, 8, dtype=torch.bfloat16)
        tokens = dspark.DSparkDraftMixin.compute_greedy_proposal(
            model,
            hidden,
            first_prev_tokens=torch.zeros(2, dtype=torch.long),
            sync=lambda x: x,
        )
        assert tokens is result
        actual_local = model._npu_greedy_shard.sample.call_args.args[0]
        torch.testing.assert_close(
            actual_local, F.linear(hidden, lm.weight).view(2, 3, 8)
        )


@pytest.mark.parametrize("dp_rank", range(4))
def test_dp_local_head_uses_attention_group(monkeypatch, dp_rank):
    monkeypatch.setattr(dspark, "_is_npu", True)
    monkeypatch.setenv("SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD", "1")
    monkeypatch.setenv("SGLANG_DSPARK_FUSED_LOCAL_TOP1", "1")
    monkeypatch.setattr(dspark, "should_apply_lm_head_quant_method", lambda *a: False)
    group = SimpleNamespace(world_size=16, rank_in_group=7)
    global_group = SimpleNamespace(world_size=64, rank_in_group=dp_rank * 16 + 7)
    monkeypatch.setattr(
        dspark,
        "get_parallel",
        lambda: SimpleNamespace(
            tp_group=global_group,
            attn_tp_group=group,
            attn_dp_size=4,
            attn_cp_size=1,
        ),
    )
    model = SimpleNamespace(
        is_nemotron_35_draft=False,
        markov_head=dspark.VanillaMarkov(vocab_size=128, markov_rank=4).bfloat16(),
    )
    lm = make_lm(128, 8, 7, 16)
    lm.quant_method = None
    lm.use_attn_tp_group = True
    dspark.DSparkDraftMixin.attach_shared_modules(model, embed_tokens=None, lm_head=lm)
    assert model._npu_greedy_shard.group is group
    assert (model._npu_greedy_shard.start, model._npu_greedy_shard.end) == (56, 64)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
