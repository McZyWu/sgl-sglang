"""Unit tests for Ascend KDA target-verify metadata and gate dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.hardware_backend.npu.attention.kda_metadata import (
    mask_dense_verify_cache_indices,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="base-a-test-1-npu-a2")


def test_dense_verify_cache_indices_mask_graph_padding():
    # A real B=1 request replayed in a captured B=4 graph. The shared graph
    # metadata uses cache slot 0 for padding, while repeated qsl offsets are the
    # source of truth for zero-length requests.
    query_start_loc = torch.tensor([0, 8, 8, 8, 8], dtype=torch.int32)
    cache_indices = torch.tensor([5, 0, 0, 0], dtype=torch.int32)

    actual = mask_dense_verify_cache_indices(cache_indices, query_start_loc)

    assert actual.dtype == torch.int64
    torch.testing.assert_close(
        actual,
        torch.tensor([5, -1, -1, -1], dtype=torch.int64),
        atol=0,
        rtol=0,
    )


def test_dense_verify_cache_indices_refreshes_replay_values():
    query_start_loc = torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32)
    cache_indices = torch.tensor([7, 11, 0, 0], dtype=torch.int32)

    first = mask_dense_verify_cache_indices(cache_indices, query_start_loc)
    torch.testing.assert_close(
        first,
        torch.tensor([7, 11, -1, -1], dtype=torch.int64),
        atol=0,
        rtol=0,
    )

    # Model a later replay of the same fixed buffers with B=1 and a different
    # live cache slot; the tensor values, rather than Python-side B, drive the
    # mask.
    query_start_loc.copy_(torch.tensor([0, 8, 8, 8, 8], dtype=torch.int32))
    cache_indices.copy_(torch.tensor([13, 0, 0, 0], dtype=torch.int32))
    second = mask_dense_verify_cache_indices(cache_indices, query_start_loc)
    torch.testing.assert_close(
        second,
        torch.tensor([13, -1, -1, -1], dtype=torch.int64),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("reuse_metadata", [False, True])
@pytest.mark.parametrize("dense_conv3d", [False, True])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_verify_preactivates_gates_and_preserves_padding(
    monkeypatch, reuse_metadata, dense_conv3d, lower_bound
):
    from sglang.srt.hardware_backend.npu.attention import ascend_kda_backend

    # A legacy launch script must not re-enable recurrent gate activation.
    monkeypatch.setenv("SGLANG_NPU_FUSED_KDA_VERIFY_GATES", "1")
    monkeypatch.setenv("SGLANG_NPU_REUSE_KDA_VERIFY_METADATA", str(int(reuse_metadata)))
    monkeypatch.setenv("SGLANG_NPU_KDA_DENSE_CONV3D", str(int(dense_conv3d)))
    backend_cls = ascend_kda_backend.AscendKDAAttnBackend
    backend = object.__new__(backend_cls)
    backend.forward_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 2], dtype=torch.int32),
        mamba_cache_indices=torch.tensor([7, 0], dtype=torch.int32),
    )
    backend._dense_cache_indices_i64 = torch.tensor([7, -1], dtype=torch.int64)
    backend._dense_verify_metadata_cache = {}
    backend.verify_intermediate_state_indices = torch.arange(2)
    cache = SimpleNamespace(
        conv=[torch.zeros(8, 3, 12)],
        temporal=torch.zeros(8, 1, 4, 4),
        intermediate_ssm=torch.zeros(2, 2, 1, 4, 4),
    )
    backend.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda _: cache)
    backend._get_conv_weights_t = lambda *_: torch.ones(4, 12)
    layer = SimpleNamespace(
        layer_id=0,
        q_dim=4,
        k_dim=4,
        v_dim=4,
        head_q_dim=4,
        head_k_dim=4,
        head_v_dim=4,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(4),
        bias=None,
        lower_bound=lower_bound,
    )
    forward_batch = SimpleNamespace(
        spec_info=SimpleNamespace(draft_token_num=2, ragged_verify_layout=None)
    )
    mixed_qkv = torch.randn(4, 12, dtype=torch.bfloat16)
    a = torch.randn(1, 4, 1, 4, dtype=torch.bfloat16)
    b = torch.randn(1, 4, 1, dtype=torch.bfloat16)
    activated_a = torch.full((1, 4, 1, 4), -0.25, dtype=torch.float32)
    expected_out = torch.randn(1, 4, 1, 4, dtype=torch.bfloat16)
    conv = Mock(side_effect=lambda x, *args, **kwargs: x)
    gate = Mock(return_value=activated_a)
    recurrent = Mock(return_value=expected_out)
    monkeypatch.setattr(torch.ops.npu, "causal_conv1d", conv, raising=False)
    monkeypatch.setattr(ascend_kda_backend, "fused_kda_gate_npu", gate)
    monkeypatch.setattr(ascend_kda_backend, "kda_target_verify_npu", recurrent)

    actual = backend._forward_target_verify(layer, forward_batch, mixed_qkv, a, b)

    assert actual is expected_out
    gate.assert_called_once()
    torch.testing.assert_close(gate.call_args.args[0], a.flatten(-2))
    assert gate.call_args.args[1] is layer.A_log
    assert gate.call_args.kwargs["gate_bias"] is layer.dt_bias
    assert gate.call_args.kwargs["lower_bound"] == lower_bound
    recurrent.assert_called_once()
    verify_args = recurrent.call_args.kwargs
    assert verify_args["gates_are_preactivated"] is True
    assert verify_args.get("lower_bound") is None
    assert verify_args["a"] is activated_a
    assert verify_args["b"].dtype == torch.float32
    torch.testing.assert_close(verify_args["b"], b.float().sigmoid())
    conv_args = conv.call_args.kwargs
    assert conv_args["cache_indices"] is backend._dense_cache_indices_i64
    assert verify_args["initial_state_indices"] is conv_args["cache_indices"]
    assert conv_args["pad_slot_id"] == -1
    assert conv.call_args.args[0].ndim == (3 if dense_conv3d else 2)
    assert (conv_args["query_start_loc"] is None) == dense_conv3d
    torch.testing.assert_close(
        conv_args["num_accepted_tokens"], torch.tensor([2, 2], dtype=torch.int32)
    )
