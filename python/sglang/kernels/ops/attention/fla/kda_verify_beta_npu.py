"""Read beta from a merged KDA projection without a BF16 staging copy."""

import torch
import triton
import triton.language as tl


@triton.jit
def _cast_strided_kda_beta_kernel(
    beta,
    output,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    heads = tl.arange(0, BLOCK_HEADS)
    mask = (rows[:, None] < ROWS) & (heads[None, :] < HEADS)
    values = tl.load(
        beta + rows[:, None] * ROW_STRIDE + heads[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output + rows[:, None] * HEADS + heads[None, :], values, mask=mask)


def cast_strided_kda_beta_to_fp32(beta: torch.Tensor) -> torch.Tensor:
    """Preserve the FP32 beta contract and the caller's existing sigmoid op.

    K3 QKVGBFA beta is a [1, tokens, heads] view with a padded token stride.
    Only the layout conversion and lossless BF16-to-FP32 cast are fused here;
    sigmoid and recurrent-state arithmetic are unchanged.
    """
    if (
        beta.ndim != 3
        or beta.shape[0] != 1
        or beta.stride(-1) != 1
        or beta.dtype != torch.bfloat16
    ):
        return beta.float()
    rows, heads = beta.shape[1:]
    output = torch.empty(beta.shape, dtype=torch.float32, device=beta.device)
    if rows * heads:
        _cast_strided_kda_beta_kernel[(triton.cdiv(rows, 32),)](
            beta,
            output,
            ROWS=rows,
            HEADS=heads,
            ROW_STRIDE=beta.stride(1),
            BLOCK_ROWS=32,
            BLOCK_HEADS=triton.next_power_of_2(heads),
            num_warps=1,
        )
    return output
