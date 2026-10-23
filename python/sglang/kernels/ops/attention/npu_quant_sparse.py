"""Normal-wheel A5 C8 LIM and Copy-SFA entry points (no import-time loading)."""

import torch


def load_quant_lim_operator():
    try:
        import sgl_kernel_npu  # noqa: F401

        return torch.ops.npu.fused_quant_lightning_indexer_manage.default
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "C8 LIM requires an Ascend950 sgl-kernel-npu wheel containing "
            "npu::fused_quant_lightning_indexer_manage."
        ) from error


def load_quant_copy_sfa_operator():
    try:
        import sgl_kernel_npu  # noqa: F401

        return torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention.default
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "C8 Copy-SFA requires an Ascend950 sgl-kernel-npu wheel containing "
            "npu::fused_quant_scatter_copy_sparse_flash_attention."
        ) from error


def fused_quant_lightning_indexer_manage(
    *,
    weights,
    query_scale,
    query,
    key_scale,
    key,
    block_table,
    query_ends,
    seq_lens,
    prefix_lens,
    cache_sizes,
    request_state,
    request_slots,
    slot_pool,
    source_ids,
    device_slots,
    topk_misses,
    miss_sources,
    miss_slots,
    miss_counts,
    operator=None,
):
    """Run FP8 index retrieval; tail hits are caller-owned slots C + source % 256."""
    if query.dtype != torch.float8_e4m3fn or key.dtype != query.dtype:
        raise ValueError("A5 C8 LIM requires FP8 E4M3FN query and index keys.")
    if weights.dtype != torch.bfloat16:
        raise ValueError("A5 C8 LIM requires BF16 index weights.")
    if query_scale.dtype != torch.float32 or key_scale.dtype != torch.float32:
        raise ValueError("A5 C8 LIM requires real FP32 query/key dequant scales.")
    if operator is None:
        operator = load_quant_lim_operator()
    operator(
        weights,
        query_scale,
        query,
        key_scale,
        key,
        block_table,
        query_ends,
        seq_lens,
        prefix_lens,
        cache_sizes,
        request_state,
        request_slots,
        slot_pool,
        source_ids,
        device_slots,
        topk_misses,
        miss_sources,
        miss_slots,
        miss_counts,
    )


def fused_quant_scatter_copy_sparse_flash_attention(
    query,
    hbm_kv_bytes,
    dram_kv_bytes,
    dram_kv_dev_ptr,
    hbm_block_table,
    dram_block_table,
    topk_dst_slots,
    topk_src_ids,
    topk_miss_counts,
    query_ends,
    seq_lens,
    hbm_lengths,
    miss_src_ids,
    miss_dst_slots,
    miss_counts,
    scale_value,
    attention_out,
    *,
    operator=None,
):
    """Attend packed C8 rows, retaining the owning registered host Tensor.

    hbm_lengths includes the caller's tail address space for sparse requests;
    zero selects dense addressing. DRAM remains registered host memory.
    """
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("A5 C8 attention query/output require BF16 or FP16.")
    if query.shape[-1] != 576 or attention_out.shape != (*query.shape[:-1], 512):
        raise ValueError("C8 attention requires query [...,576] and output [...,512].")
    if any(
        t.dtype != torch.int8 or t.shape[-1] != 656
        for t in (hbm_kv_bytes, dram_kv_bytes)
    ):
        raise ValueError("C8 caches must be opaque INT8 byte views of 656-byte rows.")
    if operator is None:
        operator = load_quant_copy_sfa_operator()
    operator(
        query,
        hbm_kv_bytes,
        dram_kv_bytes,
        dram_kv_dev_ptr,
        hbm_block_table,
        dram_block_table,
        topk_dst_slots,
        topk_src_ids,
        topk_miss_counts,
        query_ends,
        seq_lens,
        hbm_lengths,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
        float(scale_value),
        attention_out,
    )
