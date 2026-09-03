"""Unit tests for Ascend KDA target-verify metadata."""

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
