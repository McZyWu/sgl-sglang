"""Graph-recordable DSpark exponential-race sampling for Ascend.

Adapted from the sampling changes in sgl-project/sglang#34944. Noise is an
input staged before replay; this module never invokes an NPU random operator.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Large-vocabulary A3 sampling is dominated by the number of vector tasks at
# 1024 elements. 8192 amortizes them while fitting A3's 192 KiB UB; 16384 does not.
_BLOCK_V = 8192
_IDX_SENTINEL = tl.constexpr(2147483647)


def sample_step_tokens_reference(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
    corrected_logits_out: Optional[torch.Tensor] = None,
    write_corrected_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CPU reference; the NPU path never reads a device scalar on the host."""
    if corrected_logits_out is not None:
        if write_corrected_logits is None:
            raise ValueError("write_corrected_logits is required with corrected output")
        if bool(write_corrected_logits.item()):
            corrected_logits_out.copy_(step_logits)
    noise = torch.where(greedy_mask.bool()[:, None], 1.0, exp_noise)
    keys = step_logits.float() - temperatures[:, None] * noise.log()
    return keys.argmax(dim=-1)


@triton.jit
def _sample_partial_kernel(
    logits_ptr,
    temperatures_ptr,
    greedy_mask_ptr,
    exp_noise_ptr,
    partial_key_ptr,
    partial_idx_ptr,
    corrected_ptr,
    write_corrected_ptr,
    V,
    logits_stride_row,
    noise_stride_row,
    corrected_stride_row,
    n_tiles,
    BLOCK_V: tl.constexpr,
    STORE_CORRECTED: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    logits = tl.load(
        logits_ptr + row * logits_stride_row + offs,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if STORE_CORRECTED:
        if tl.load(write_corrected_ptr) != 0:
            tl.store(
                corrected_ptr + row * corrected_stride_row + offs,
                logits,
                mask=mask,
            )
    if tl.load(greedy_mask_ptr + row) != 0:
        # Preserve direct-logit argmax, including near ties. Greedy rows
        # perform no noise load/logarithm and require no RNG before replay.
        key = logits
    else:
        temperature = tl.load(temperatures_ptr + row)
        noise = tl.load(
            exp_noise_ptr + row * noise_stride_row + offs, mask=mask, other=1.0
        )
        key = logits - temperature * tl.log(noise)
    key = tl.where(mask, key, float("-inf"))
    tile_best = tl.max(key, axis=0)
    candidates = tl.where(mask & (key == tile_best), offs, _IDX_SENTINEL)
    tl.store(partial_key_ptr + row * n_tiles + tile, tile_best)
    tl.store(partial_idx_ptr + row * n_tiles + tile, tl.min(candidates, axis=0))


@triton.jit
def _sample_combine_kernel(
    partial_key_ptr,
    partial_idx_ptr,
    output_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < n_tiles
    keys = tl.load(
        partial_key_ptr + row * n_tiles + offs, mask=mask, other=float("-inf")
    )
    ids = tl.load(
        partial_idx_ptr + row * n_tiles + offs, mask=mask, other=_IDX_SENTINEL
    )
    best = tl.max(keys, axis=0)
    token = tl.min(tl.where(keys == best, ids, _IDX_SENTINEL), axis=0)
    token = tl.where(token == _IDX_SENTINEL, 0, token)
    tl.store(output_ptr + row, token.to(tl.int64))


def sample_step_tokens_npu(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
    corrected_logits_out: Optional[torch.Tensor] = None,
    write_corrected_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bs, vocab = step_logits.shape
    assert step_logits.stride(1) == 1
    assert exp_noise.shape == step_logits.shape and exp_noise.stride(1) == 1
    # Rows can be gamma-strided views of persistent [B, gamma, V] buffers.
    # Consuming their strides avoids a full-vocabulary copy for every step.
    temperatures = temperatures.to(torch.float32).contiguous()
    greedy_mask = greedy_mask.contiguous()
    assert greedy_mask.dtype in (torch.bool, torch.int32)
    assert exp_noise.dtype == torch.float32
    store_corrected = corrected_logits_out is not None
    if store_corrected:
        if write_corrected_logits is None:
            raise ValueError("write_corrected_logits is required with corrected output")
        assert corrected_logits_out.shape == step_logits.shape
        assert corrected_logits_out.stride(1) == 1
    else:
        corrected_logits_out = step_logits
        write_corrected_logits = greedy_mask

    n_tiles = triton.cdiv(vocab, _BLOCK_V)
    partial_key = torch.empty(
        (bs, n_tiles), dtype=torch.float32, device=step_logits.device
    )
    partial_idx = torch.empty(
        (bs, n_tiles), dtype=torch.int32, device=step_logits.device
    )
    output = torch.empty((bs,), dtype=torch.int64, device=step_logits.device)
    _sample_partial_kernel[(bs, n_tiles)](
        step_logits,
        temperatures,
        greedy_mask,
        exp_noise,
        partial_key,
        partial_idx,
        corrected_logits_out,
        write_corrected_logits,
        vocab,
        step_logits.stride(0),
        exp_noise.stride(0),
        corrected_logits_out.stride(0),
        n_tiles,
        BLOCK_V=_BLOCK_V,
        STORE_CORRECTED=store_corrected,
    )
    _sample_combine_kernel[(bs,)](
        partial_key,
        partial_idx,
        output,
        n_tiles,
        BLOCK_TILES=triton.next_power_of_2(n_tiles),
    )
    return output
