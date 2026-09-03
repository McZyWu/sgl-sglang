"""Unit tests for Ascend KDA target-verify metadata."""

import sys
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="base-a-test-1-npu-a2")

# Mock NPU-only packages before importing the backend. The helper under test
# uses ordinary torch operations and intentionally runs without an NPU device.
for _module in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.fla",
    "sgl_kernel_npu.fla.kda_chunk_delta_h",
    "sgl_kernel_npu.fla.kda_gate",
    "sgl_kernel_npu.fla.kda_prefill",
    "sgl_kernel_npu.fla.kda_target_verify",
    "sgl_kernel_npu.fla.solve_tril",
    "sgl_kernel_npu.fla.utils",
    "sgl_kernel_npu.mamba",
    "sgl_kernel_npu.mamba.causal_conv1d",
):
    sys.modules.setdefault(_module, MagicMock())

from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (  # noqa: E402
    _mask_dense_verify_cache_indices,
)


def test_dense_verify_cache_indices_mask_graph_padding():
    # A real B=1 request replayed in a captured B=4 graph. The shared graph
    # metadata uses cache slot 0 for padding, while repeated qsl offsets are the
    # source of truth for zero-length requests.
    query_start_loc = torch.tensor([0, 8, 8, 8, 8], dtype=torch.int32)
    cache_indices = torch.tensor([5, 0, 0, 0], dtype=torch.int32)

    actual = _mask_dense_verify_cache_indices(cache_indices, query_start_loc)

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

    first = _mask_dense_verify_cache_indices(cache_indices, query_start_loc)
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
    second = _mask_dense_verify_cache_indices(cache_indices, query_start_loc)
    torch.testing.assert_close(
        second,
        torch.tensor([13, -1, -1, -1], dtype=torch.int64),
        atol=0,
        rtol=0,
    )
