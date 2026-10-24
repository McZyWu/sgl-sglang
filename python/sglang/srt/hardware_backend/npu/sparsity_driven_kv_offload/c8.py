"""A5 C8 storage and attention integration; request/graph ownership stays in SRT."""

import torch

from sglang.kernels.ops.attention.npu_quant_sparse import (
    fused_quant_scatter_copy_sparse_flash_attention,
    load_quant_copy_sfa_operator,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.block_tables import (
    fill_request_block_table,
)

PACKED_ROW_BYTES = 656
TAIL_SLOTS = 256


def c8_write_slots(requests, positions, lengths, capacity, request_rows):
    """Publish the page tail in the native LIM's prefix-relative slots.

    Q1's prefix is the page boundary before its newest token. Selected tail
    sources use C + (source - prefix), including when that boundary is odd.
    A sparse cache is populated by LIM, never overwritten with dense identity KV.
    """
    valid = (
        (requests > 0)
        & (requests < request_rows)
        & (positions >= 0)
        & (positions < lengths)
    )
    dense = valid & (lengths <= capacity) & (positions < capacity)
    prefix = ((lengths - 1).clamp(min=0) // 128) * 128
    tail = valid & (positions >= prefix)
    return positions, dense, capacity + positions - prefix, tail


class C8Cache:
    def __init__(self, manager, query_heads, compute_dtype):
        self.operator = load_quant_copy_sfa_operator()
        self.query = torch.empty(
            (manager.batch_capacity, query_heads, 576),
            dtype=compute_dtype,
            device=manager.device,
        )
        self.request_lengths = torch.zeros(
            manager.size + 1, dtype=torch.int32, device=manager.device
        )

    def store_rows(
        self,
        manager,
        layer_index,
        batch,
        source,
        source_ids,
        destination_ids,
        mask,
        copy,
    ):
        requests = destination_ids // manager.max_context_len
        positions = destination_ids.remainder(manager.max_context_len)
        safe_requests = requests.clamp(min=0, max=manager.size)
        batch_requests = batch.req_pool_indices.long()
        safe_batch = batch_requests.clamp(min=0, max=manager.size)
        valid_batch = (batch_requests > 0) & (batch_requests < manager.size)
        self.request_lengths.zero_()
        self.request_lengths.index_copy_(
            0, safe_batch, torch.where(valid_batch, batch.seq_lens, 0).to(torch.int32)
        )
        lengths = self.request_lengths[safe_requests]
        dense_slots, dense, tail_slots, tail = c8_write_slots(
            requests, positions, lengths, manager.cache_capacity, manager.size
        )
        for slots, valid in ((dense_slots, dense), (tail_slots, tail)):
            destination = (
                requests * manager.hbm_tokens_per_request + slots
            ).contiguous()
            copy(
                source,
                manager.device_kv_buffer[layer_index],
                source_ids,
                destination,
                (mask & valid).contiguous(),
                1,
                2,
                block_dim=48,
            )
        if batch.forward_mode.is_extend_without_speculative():
            # A chunked extension changes payload without maintaining sparse residency.
            manager.lim_adapter.reset_requests(batch_requests[valid_batch])

    def decode(self, manager, layer, batch, q, q_rope, plan, query_heads, scale):
        if plan is None:
            raise RuntimeError("C8 Copy-SFA requires the native quant LIM plan.")
        count = batch.batch_size
        valid = plan.valid_rows
        plan.device_slots.masked_fill_(~valid[:, None], -1)
        plan.topk_misses.masked_fill_(~valid, 0)
        plan.miss_counts.masked_fill_(~valid, 0)
        manager.planner_actual_seq_lengths_kv[:count].copy_(plan.seq_lens)
        manager.planner_num_cache_tokens[:count].copy_(
            torch.where(valid & plan.sparse_rows, manager.hbm_tokens_per_request, 0)
        )
        for pages, offsets, output in (
            (
                manager.hbm_tokens_per_request // 128,
                manager.hbm_page_offsets,
                manager.planner_hbm_block_table,
            ),
            (
                manager.max_context_len // 128,
                manager.dram_page_offsets,
                manager.planner_dram_block_table,
            ),
        ):
            fill_request_block_table(
                plan.request_slots, valid, offsets, pages, output[:count]
            )
        q = q.reshape(count, -1, 512)
        q_rope = q_rope.reshape(count, -1, 64)
        merged = self.query[:count]
        if merged.shape[1] < q.shape[1] or merged.dtype != q.dtype:
            raise ValueError("C8 query exceeds the graph-stable query workspace.")
        if query_heads < merged.shape[1]:
            merged.zero_()
        merged[:, :query_heads, :512].copy_(q[:, :query_heads])
        merged[:, :query_heads, 512:].copy_(q_rope[:, :query_heads])
        output = manager.planner_attn_out[:count]
        layer_index = layer.layer_id - manager.start_layer
        hbm = manager.device_kv_buffer[layer_index].view(-1, 128, 1, PACKED_ROW_BYTES)
        dram = manager.host_kv_buffer[layer_index].view(-1, 128, 1, PACKED_ROW_BYTES)
        fused_quant_scatter_copy_sparse_flash_attention(
            merged,
            hbm,
            dram,
            manager.dev_ptr_list[layer_index],
            manager.planner_hbm_block_table[:count],
            manager.planner_dram_block_table[:count],
            plan.device_slots.unsqueeze(1),
            plan.source_ids.unsqueeze(1),
            plan.topk_misses,
            manager.planner_actual_seq_lengths_query[:count],
            manager.planner_actual_seq_lengths_kv[:count],
            manager.planner_num_cache_tokens[:count],
            plan.miss_sources.unsqueeze(1),
            plan.miss_slots.unsqueeze(1),
            plan.miss_counts,
            scale,
            output,
            operator=self.operator,
        )
        output.masked_fill_(~valid[:, None, None], 0)
        manager.last_decode_path = batch.sparse_kv_decode_path = "copy_sfa_c8"
        return output[:, :query_heads].reshape(count, -1).contiguous()


def forward_c8_offload(
    backend, q, k, layer, batch, save_kv_cache, q_rope, k_rope, topk_indices
):
    """Q1 uses fused C8; prefill gathers page-aligned request KV for vendor QSFA."""
    import torch_npu

    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
        unidex_copy_inplace,
    )

    manager = backend.sparse_kv_manager
    stream = torch.npu.current_stream(backend.device)
    if save_kv_cache:
        manager.offload_v2(
            k.reshape(-1, 1, 512), k_rope.reshape(-1, 1, 64), layer, batch, stream
        )
    if batch.forward_mode.is_decode():
        return manager.c8_cache.decode(
            manager,
            layer,
            batch,
            q,
            q_rope,
            backend.forward_metadata.sparse_kv_lim_plan,
            layer.tp_q_head_num,
            layer.scaling,
        )
    if not batch.forward_mode.is_extend_without_speculative():
        raise ValueError("A5 C8 offload supports prefill and Q1 decode.")
    reqs_cpu = batch.req_pool_indices_cpu.tolist()
    lengths_cpu = batch.seq_lens_cpu.tolist()
    pages_cpu = []
    for request, length in zip(reqs_cpu, lengths_cpu):
        if (
            length < 0
            or length > manager.max_context_len
            or (length > 0 and not 0 < request < manager.size)
        ):
            raise ValueError(
                "C8 prefill request/length exceeds registered host storage."
            )
        pages_cpu.append((int(length) + 127) // 128)
    total_pages = max(1, sum(pages_cpu))
    max_pages = max(1, max(pages_cpu, default=0))
    pages = torch.tensor(pages_cpu, dtype=torch.int64, device=q.device)
    starts = pages.cumsum(0) - pages
    rows = torch.repeat_interleave(
        batch.req_pool_indices.long(), pages * 128, output_size=sum(pages_cpu) * 128
    )
    offsets = torch.repeat_interleave(
        starts * 128, pages * 128, output_size=rows.numel()
    )
    positions = torch.arange(rows.numel(), device=q.device) - offsets
    lengths = torch.repeat_interleave(
        batch.seq_lens, pages * 128, output_size=rows.numel()
    )
    packed = torch.zeros(
        (total_pages, 128, 1, PACKED_ROW_BYTES), dtype=torch.int8, device=q.device
    )
    layer_index = layer.layer_id - manager.start_layer
    unidex_copy_inplace(
        manager.host_kv_buffer[layer_index],
        packed.view(-1, 1, PACKED_ROW_BYTES),
        (rows * manager.max_context_len + positions).contiguous(),
        torch.arange(rows.numel(), dtype=torch.int64, device=q.device),
        (positions < lengths).contiguous(),
        2,
        1,
        block_dim=48,
        src_ptr=manager.dev_ptr_list[layer_index],
    )
    page_offsets = torch.arange(max_pages, device=q.device)
    table = torch.where(
        page_offsets[None] < pages[:, None], starts[:, None] + page_offsets[None], 0
    ).to(torch.int32)
    ends = backend.forward_metadata.actual_seq_lengths_q
    if ends is None:
        ends = batch.extend_seq_lens.cumsum(0)
    merged = torch.cat(
        (q.reshape(-1, q.shape[-2], 512), q_rope.reshape(-1, q_rope.shape[-2], 64)),
        dim=-1,
    ).contiguous()
    padded_heads = manager.c8_cache.query.shape[1]
    if merged.shape[1] != padded_heads:
        padded = torch.zeros(
            (merged.shape[0], padded_heads, 576),
            dtype=merged.dtype,
            device=merged.device,
        )
        padded[:, : merged.shape[1]].copy_(merged)
        merged = padded
    result = torch_npu.npu_kv_quant_sparse_flash_attention(
        query=merged,
        key=packed.view(torch.float8_e4m3fn),
        value=packed.view(torch.float8_e4m3fn),
        sparse_indices=topk_indices.reshape(-1, 1, manager.sparse_context_len),
        scale_value=layer.scaling,
        key_quant_mode=2,
        value_quant_mode=2,
        key_dequant_scale=None,
        value_dequant_scale=None,
        actual_seq_lengths_query=ends.to(device=q.device, dtype=torch.int32),
        actual_seq_lengths_kv=batch.seq_lens.to(torch.int32),
        block_table=table,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    result = result[0] if isinstance(result, tuple) else result
    return result[:, : layer.tp_q_head_num].contiguous()
