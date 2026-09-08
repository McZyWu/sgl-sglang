"""Local candidate selection for the BF16 VanillaMarkov logits contract."""

import torch
import triton
import triton.language as tl


@triton.jit
def _vanilla_local_top1_kernel(
    base_ptr,
    bias_ptr,
    output_ptr,
    V: tl.constexpr,
    OFFSET: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    best_value = tl.full((), float("-inf"), tl.float32)
    best_index = tl.full((), 2147483647, tl.int32)
    first_nan = tl.full((), 2147483647, tl.int32)
    for start in range(0, V, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        base = tl.load(base_ptr + row * BASE_STRIDE + cols, cols < V, other=0)
        bias = tl.load(bias_ptr + row * BIAS_STRIDE + cols, cols < V, other=0)
        # VanillaMarkov adds two BF16 tensors with a BF16 result. DSv4's
        # BuildStepLocal instead keeps the sum in FP32; its selector cannot
        # replace this path because rounding can change the first argmax.
        values = (base.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16)
        values = tl.where(cols < V, values.to(tl.float32), float("-inf"))
        nan_mask = (cols < V) & (values != values)
        first_nan = tl.minimum(
            first_nan, tl.min(tl.where(nan_mask, cols, 2147483647), 0)
        )
        values = tl.where(nan_mask, float("-inf"), values)
        value = tl.max(values, 0)
        index = tl.min(tl.where((cols < V) & (values == value), cols, 2147483647), 0)
        take = (value > best_value) | ((value == best_value) & (index < best_index))
        best_index = tl.where(take, index, best_index)
        best_value = tl.maximum(best_value, value)
    has_nan = first_nan != 2147483647
    tl.store(output_ptr + row * 2, tl.where(has_nan, float("nan"), best_value))
    index = tl.where(has_nan, first_nan, best_index)
    tl.store(output_ptr + row * 2 + 1, (index + OFFSET).to(tl.float32))


@triton.jit
def _vanilla_global_top1_kernel(
    candidates_ptr,
    output_ptr,
    TP: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK_TP: tl.constexpr,
):
    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK_TP)
    values = tl.load(
        candidates_ptr + row * TP * 2 + ranks * 2, ranks < TP, other=float("-inf")
    )
    ids = tl.load(
        candidates_ptr + row * TP * 2 + ranks * 2 + 1, ranks < TP, other=VOCAB
    ).to(tl.int32)
    nan_mask = (ranks < TP) & (values != values)
    first_nan = tl.min(tl.where(nan_mask, ids, VOCAB), 0)
    values = tl.where(nan_mask, float("-inf"), values)
    best = tl.max(values, 0)
    first = tl.min(tl.where((ranks < TP) & (values == best), ids, VOCAB), 0)
    tl.store(
        output_ptr + row, tl.where(first_nan < VOCAB, first_nan, first).to(tl.int64)
    )


def select_vanilla_global_top1_npu(candidates, *, vocab_size):
    if (
        candidates.ndim != 3
        or candidates.shape[2] != 2
        or not candidates.is_contiguous()
    ):
        raise ValueError("expected contiguous [batch, tp, 2] candidates")
    bs, tp, _ = candidates.shape
    output = torch.empty(bs, dtype=torch.long, device=candidates.device)
    _vanilla_global_top1_kernel[(bs,)](
        candidates, output, TP=tp, VOCAB=vocab_size, BLOCK_TP=triton.next_power_of_2(tp)
    )
    return output


def select_vanilla_local_top1_npu(base, bias, *, vocab_offset):
    """Return FP32 (rounded value, global id) pairs, including strided rows."""
    if (
        base.ndim != 2
        or base.shape != bias.shape
        or base.dtype != torch.bfloat16
        or bias.dtype != torch.bfloat16
        or base.stride(1) != 1
        or bias.stride(1) != 1
        or base.shape[1] == 0
    ):
        raise ValueError("expected nonempty row-contiguous BF16 logits and bias")
    output = torch.empty((base.shape[0], 2), dtype=torch.float32, device=base.device)
    _vanilla_local_top1_kernel[(base.shape[0],)](
        base,
        bias,
        output,
        V=base.shape[1],
        OFFSET=vocab_offset,
        BASE_STRIDE=base.stride(0),
        BIAS_STRIDE=bias.stride(0),
        BLOCK_V=min(2048, triton.next_power_of_2(base.shape[1])),
    )
    return output
