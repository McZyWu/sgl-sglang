from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.arg_groups.validation_hook import (
    _validate_k3_deepep_isolation_args,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

parallel_state = pytest.importorskip("sglang.srt.distributed.parallel_state")


def _validation_cfg(**overrides):
    values = {
        "enable_deepep_topk_int32": True,
        "shared_experts_attn_tp_size": 4,
        "enable_shared_experts_attn_tp": True,
        "device": "npu",
        "moe_a2a_backend": "deepep",
        "enable_dp_attention": True,
        "tp_size": 64,
        "attn_cp_size": 1,
        "dp_size": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _init_shared_experts_group(attn_tp_group, requested_size):
    return parallel_state._init_shared_experts_tp_group(
        attn_tp_group=attn_tp_group,
        requested_size=requested_size,
        num_tensor_model_parallel_groups=1,
        tensor_model_parallel_size=64,
        attn_cp_size=1,
        attn_dp_size=4,
        attn_tp_size=16,
        local_rank=3,
        backend="hccl",
        recovered_rank=False,
        rank_offset=0,
        max_world_size=None,
    )


def test_shared_experts_tp_defaults_are_disabled():
    server_args = ServerArgs(model_path="dummy")

    assert server_args.shared_experts_attn_tp_size is None
    assert server_args.enable_deepep_topk_int32 is False


def test_shared_experts_tp_cli_is_independently_switchable():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)

    parsed = parser.parse_args(
        [
            "--model-path",
            "dummy",
            "--enable-shared-experts-attn-tp",
            "--shared-experts-attn-tp-size",
            "4",
            "--enable-deepep-topk-int32",
        ]
    )

    assert parsed.shared_experts_attn_tp_size == 4
    assert parsed.enable_deepep_topk_int32 is True


def test_k3_deepep_isolation_validation_accepts_tp4_topk_int32():
    _validate_k3_deepep_isolation_args(_validation_cfg())


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"device": "cuda"}, "--enable-deepep-topk-int32 requires --device npu"),
        (
            {"moe_a2a_backend": "none"},
            "--enable-deepep-topk-int32 requires --moe-a2a-backend deepep",
        ),
        (
            {
                "enable_deepep_topk_int32": False,
                "enable_shared_experts_attn_tp": False,
            },
            "--shared-experts-attn-tp-size requires --enable-shared-experts-attn-tp",
        ),
        (
            {"enable_deepep_topk_int32": False, "moe_a2a_backend": "none"},
            "--shared-experts-attn-tp-size requires --moe-a2a-backend deepep",
        ),
        (
            {"enable_deepep_topk_int32": False, "enable_dp_attention": False},
            "--shared-experts-attn-tp-size requires --enable-dp-attention",
        ),
        (
            {
                "enable_deepep_topk_int32": False,
                "tp_size": 16,
                "dp_size": 4,
                "shared_experts_attn_tp_size": 8,
            },
            "effective attention TP 4",
        ),
    ],
)
def test_k3_deepep_isolation_validation_rejects_noops(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate_k3_deepep_isolation_args(_validation_cfg(**overrides))


@pytest.mark.parametrize("requested", [None, 0])
def test_shared_experts_tp_default_reuses_attention_tp(requested):
    assert parallel_state.resolve_shared_experts_attn_tp_size(16, requested) == 16


@pytest.mark.parametrize("requested", [4, 8])
def test_shared_experts_tp_explicit_subgroups(requested):
    assert (
        parallel_state.resolve_shared_experts_attn_tp_size(16, requested) == requested
    )


@pytest.mark.parametrize("attn_tp_size,requested", [(16, 2), (6, 4), (4, 8)])
def test_shared_experts_tp_rejects_invalid_subgroups(attn_tp_size, requested):
    with pytest.raises(ValueError):
        parallel_state.resolve_shared_experts_attn_tp_size(attn_tp_size, requested)


@pytest.mark.parametrize("subgroup_size", [4, 8])
def test_shared_experts_tp_groups_match_tp64_dp4_layout(subgroup_size):
    groups = parallel_state.build_shared_experts_tp_group_ranks(
        num_tensor_model_parallel_groups=1,
        tensor_model_parallel_size=64,
        attn_cp_size=1,
        attn_dp_size=4,
        attn_tp_size=16,
        shared_experts_tp_size=subgroup_size,
    )

    assert groups == [
        list(range(start, start + subgroup_size))
        for start in range(0, 64, subgroup_size)
    ]


@pytest.mark.parametrize("requested", [None, 0])
def test_shared_experts_tp_default_aliases_attention_group(requested):
    attn_tp_group = object()
    with patch.object(parallel_state, "init_model_parallel_group") as init_group:
        result = _init_shared_experts_group(attn_tp_group, requested)

    assert result is attn_tp_group
    init_group.assert_not_called()


def test_shared_experts_tp4_creates_independent_group():
    attn_tp_group = object()
    independent_group = object()
    with patch.object(
        parallel_state,
        "init_model_parallel_group",
        return_value=independent_group,
    ) as init_group:
        result = _init_shared_experts_group(attn_tp_group, 4)

    assert result is independent_group
    assert result is not attn_tp_group
    group_ranks, local_rank, backend = init_group.call_args.args
    assert group_ranks == [list(range(start, start + 4)) for start in range(0, 64, 4)]
    assert (local_rank, backend) == (3, "hccl")
    assert init_group.call_args.kwargs == {
        "use_pynccl": False,
        "use_custom_allreduce": False,
        "use_torch_symm_mem_allreduce": False,
        "group_name": "shared_experts_tp",
        "recovered_rank": False,
        "rank_offset": 0,
        "max_world_size": None,
    }
