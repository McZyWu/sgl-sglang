"""Opt-in KDA state commits with stride-aware compatibility fallbacks."""

import torch
from sgl_kernel_npu.mamba.kda_state_commit import (
    commit_kda_extended_conv_state,
    move_kda_temporal_snapshot,
    scatter_kda_conv_snapshot,
)
from sgl_kernel_npu.mamba.mamba_state_update_triton import (
    conv_state_rollback,
    move_intermediate_cache_kda,
)
from sgl_kernel_npu.mamba.speculative_state_scatter import speculative_state_scatter_npu


def _move_temporal(dst, src, dst_indices, src_indices, steps):
    if not move_kda_temporal_snapshot(dst, src, dst_indices, src_indices, steps):
        move_intermediate_cache_kda(
            dst, src, dst_indices, src_indices, steps, h_block_size=1
        )


def _scatter_conv(dst, src, dst_indices, src_indices, steps):
    if not scatter_kda_conv_snapshot(dst, src, dst_indices, src_indices, steps):
        speculative_state_scatter_npu(dst, src, dst_indices, src_indices, steps)


def commit_kda_verify_states(
    caches,
    dst_indices,
    src_indices,
    last_steps,
    track_indices,
    track_steps,
    *,
    has_conv_snapshots,
):
    """Commit accepted persistent state and optional prefix-tracking slots.

    Tracking reads the original extended verify window, before the primary
    tail is overwritten. Only the persistent tail of that scratch window is
    live after commit; the next verify reconstructs its leading positions.
    Unsupported kernel layouts use the existing stride-aware operations.
    """
    conv = caches.conv[0]
    temporal = caches.temporal
    snapshots = caches.intermediate_ssm
    draft_tokens = snapshots.shape[2]
    _move_temporal(temporal, snapshots, dst_indices, src_indices, last_steps)

    if track_indices is not None:
        assert track_steps is not None
        track_indices = track_indices.to(torch.int32)
        track_steps = track_steps.to(torch.int32)
        _move_temporal(temporal, snapshots, track_indices, src_indices, track_steps)

    if has_conv_snapshots:
        conv_snapshots = caches.intermediate_conv_window[0]
        _scatter_conv(conv, conv_snapshots, dst_indices, src_indices, last_steps)
        if track_indices is not None:
            _scatter_conv(conv, conv_snapshots, track_indices, src_indices, track_steps)
        return

    if track_indices is not None and track_indices.numel() > 0:
        if not commit_kda_extended_conv_state(
            conv, track_indices, dst_indices, track_steps, draft_tokens
        ):
            # Keep replay-safe indexing: no nonzero() or host read of a mask.
            src_slots = torch.where(track_steps >= 0, dst_indices, track_indices)
            conv[:, track_indices] = conv[:, src_slots]
            conv_state_rollback(conv, track_indices, track_steps, draft_tokens)

    if not commit_kda_extended_conv_state(
        conv, dst_indices, dst_indices, last_steps, draft_tokens
    ):
        conv_state_rollback(conv, dst_indices, last_steps, draft_tokens)
