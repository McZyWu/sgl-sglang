"""CPU contracts for LIM request state and persistent HBM reservation."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.npu_lim import (
    fused_lightning_indexer_manage,
    load_fused_lim_operator,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.block_tables import (
    prepare_prefill_gather_rows,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.budget import (
    resolve_sparse_kv_budget,
    sparse_batch_capacity,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8 import (
    C8Cache,
    c8_write_slots,
    forward_c8_offload,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.lim import (
    LIMDecodeAdapter,
    prepare_lim_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a")


class TestLIMState(CustomTestCase):
    def test_c8_ragged_prefill_keeps_request_pages_and_query_heads_separate(self):
        # Different partial-page lengths would alias if packed TND rows were
        # passed as page-aligned PA storage. The vendor boundary consumes the
        # actual gathered bytes and page table, producing observable values.
        host = torch.zeros(3, 256, 1, 656, dtype=torch.int8)
        for request, positions, values in (
            (1, [0, 1, 2], [1.0, 2.0, 3.0]),
            (2, [0, 128, 129], [4.0, 5.0, 6.0]),
        ):
            for position, value in zip(positions, values):
                host[request, position, 0, :512] = (
                    torch.full((512,), value).to(torch.float8_e4m3fn).view(torch.int8)
                )
                host[request, position, 0, 640:] = torch.ones(4).view(torch.int8)
        manager = SimpleNamespace(
            size=3,
            max_context_len=256,
            start_layer=0,
            host_kv_buffer=[host],
            dev_ptr_list=[123],
            sparse_context_len=2048,
            c8_cache=SimpleNamespace(query=torch.empty(2, 8, 576)),
        )
        backend = SimpleNamespace(
            device="cpu",
            sparse_kv_manager=manager,
            forward_metadata=SimpleNamespace(actual_seq_lengths_q=torch.tensor([1, 2])),
        )
        forward = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: False, is_extend_without_speculative=lambda: True
            ),
            req_pool_indices=torch.tensor([1, 2]),
            req_pool_indices_cpu=torch.tensor([1, 2]),
            seq_lens=torch.tensor([3, 130]),
            seq_lens_cpu=torch.tensor([3, 130]),
            extend_seq_lens=torch.ones(2, dtype=torch.int32),
        )

        def byte_copy(
            src, dst, source_ids, destination_ids, valid, src_ndims, dst_ndims, **kwargs
        ):
            dst.view(-1, 656).index_copy_(
                0, destination_ids[valid], src.view(-1, 656)[source_ids[valid]]
            )

        def vendor_attention(**args):
            packed = args["key"].view(torch.int8).view(-1, 656)
            output = []
            for row in range(2):
                ids = args["sparse_indices"][row, 0]
                ids = ids[ids >= 0].long()
                physical = args["block_table"][row, ids // 128].long() * 128 + ids % 128
                payload = packed[physical].contiguous()
                values = payload[:, :512].contiguous().view(torch.float8_e4m3fn).float()
                scales = (
                    payload[:, 640:]
                    .contiguous()
                    .view(torch.float32)
                    .repeat_interleave(128, -1)
                )
                output.append(
                    (values * scales).mean(0).expand(args["query"].shape[1], -1)
                )
            return torch.stack(output).to(torch.bfloat16)

        external = ModuleType("torch_npu")
        external.npu_kv_quant_sparse_flash_attention = vendor_attention
        manager_module = ModuleType(
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager"
        )
        manager_module.unidex_copy_inplace = byte_copy
        ids = torch.full((2, 2048), -1, dtype=torch.int32)
        ids[0, :3] = torch.tensor([0, 1, 2])
        ids[1, :3] = torch.tensor([0, 128, 129])
        with (
            patch.dict(
                "sys.modules",
                {external.__name__: external, manager_module.__name__: manager_module},
            ),
            patch.object(
                torch,
                "npu",
                SimpleNamespace(current_stream=lambda device: None),
                create=True,
            ),
        ):
            result = forward_c8_offload(
                backend,
                torch.zeros(2, 4, 512, dtype=torch.bfloat16),
                None,
                SimpleNamespace(layer_id=0, tp_q_head_num=4, scaling=1.0),
                forward,
                False,
                torch.zeros(2, 4, 64, dtype=torch.bfloat16),
                None,
                ids,
            )
        self.assertEqual(result.shape, (2, 4, 512))
        torch.testing.assert_close(result[0], torch.full_like(result[0], 2))
        torch.testing.assert_close(result[1], torch.full_like(result[1], 5))

    def test_c8_decode_exposes_tail_address_space_and_zeros_padding(self):
        # The external native boundary reads resident slots. Check actual output
        # values so a wrong C, block table, layer, or padding mask changes evidence.
        batch, capacity, rows, heads = 3, 2048, 3, 8
        physical = capacity + 256

        def ints(*shape):
            return torch.empty(*shape, dtype=torch.int32)

        manager = SimpleNamespace(
            hbm_tokens_per_request=physical,
            cache_capacity=capacity,
            size=rows,
            max_context_len=4224,
            start_layer=7,
            planner_actual_seq_lengths_kv=ints(batch),
            planner_num_cache_tokens=ints(batch),
            planner_actual_seq_lengths_query=torch.arange(
                1, batch + 1, dtype=torch.int32
            ),
            hbm_page_offsets=torch.arange(physical // 128, dtype=torch.int32),
            dram_page_offsets=torch.arange(33, dtype=torch.int32),
            planner_hbm_block_table=ints(batch, physical // 128),
            planner_dram_block_table=ints(batch, 33),
            planner_attn_out=torch.full((batch, heads, 512), 123, dtype=torch.bfloat16),
            device_kv_buffer=[torch.zeros(rows, physical, 1, 656, dtype=torch.int8)],
            host_kv_buffer=[torch.zeros(rows, 4224, 1, 656, dtype=torch.int8)],
            dev_ptr_list=[1234],
        )
        plan = SimpleNamespace(
            valid_rows=torch.tensor([True, False, True]),
            sparse_rows=torch.tensor([True, False, False]),
            seq_lens=torch.tensor([2050, 0, 130], dtype=torch.int32),
            request_slots=torch.tensor([1, 0, 2], dtype=torch.int32),
            topk_misses=torch.zeros(batch, dtype=torch.int32),
            miss_counts=torch.zeros(batch, dtype=torch.int32),
            source_ids=torch.full((batch, capacity), -1, dtype=torch.int32),
            device_slots=torch.full((batch, capacity), -1, dtype=torch.int32),
            miss_sources=ints(batch, 32768),
            miss_slots=ints(batch, 32768),
        )
        plan.source_ids[0, :2] = torch.tensor([0, 2049])
        plan.device_slots[0, :2] = torch.tensor([0, capacity + 1])
        plan.source_ids[2, :2] = plan.device_slots[2, :2] = torch.tensor([0, 1])
        plan.device_slots[1].fill_(123)
        for request, slots, values in (
            (1, [0, capacity + 1], [1.0, 2.0]),
            (2, [0, 1], [3.0, 4.0]),
        ):
            for slot, value in zip(slots, values):
                manager.device_kv_buffer[0][request, slot, 0, :512] = (
                    torch.full((512,), value).to(torch.float8_e4m3fn).view(torch.int8)
                )

        def external_attention(
            query,
            hbm,
            dram,
            pointer,
            hbm_table,
            dram_table,
            dst,
            src,
            topk_misses,
            ends,
            lengths,
            resident_lengths,
            misses,
            miss_slots,
            miss_counts,
            scale,
            output,
        ):
            self.assertEqual(pointer, 1234)
            self.assertEqual(resident_lengths.tolist(), [2304, 0, 0])
            self.assertEqual(lengths.tolist(), [2050, 0, 130])
            self.assertTrue(torch.all(dst[1] == -1))
            self.assertEqual(misses.stride(), (32768, 32768, 1))
            for row in (0, 2):
                logical = dst[row, 0, :2].long()
                physical_rows = (
                    hbm_table[row, logical // 128].long() * 128 + logical % 128
                )
                values = (
                    hbm.view(-1, 656)[physical_rows, :512]
                    .contiguous()
                    .view(torch.float8_e4m3fn)
                    .float()
                )
                output[row].copy_(values.mean(0).expand(heads, -1))

        cache = C8Cache.__new__(C8Cache)
        cache.query = torch.empty(batch, heads, 576, dtype=torch.bfloat16)
        cache.operator = external_attention
        forward = SimpleNamespace(batch_size=batch)
        output = cache.decode(
            manager,
            SimpleNamespace(layer_id=7),
            forward,
            torch.zeros(batch, heads, 512, dtype=torch.bfloat16),
            torch.zeros(batch, heads, 64, dtype=torch.bfloat16),
            plan,
            heads,
            1.0,
        )
        torch.testing.assert_close(output[0], torch.full_like(output[0], 1.5))
        torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
        torch.testing.assert_close(output[2], torch.full_like(output[2], 3.5))
        short_heads = cache.decode(
            manager,
            SimpleNamespace(layer_id=7),
            forward,
            torch.zeros(batch, 4, 512, dtype=torch.bfloat16),
            torch.zeros(batch, 4, 64, dtype=torch.bfloat16),
            plan,
            4,
            1.0,
        )
        self.assertEqual(short_heads.shape, (batch, 4 * 512))
        torch.testing.assert_close(short_heads[0], torch.full_like(short_heads[0], 1.5))

    def test_c8_tail_writes_are_unique_and_do_not_overwrite_sparse_slots(self):
        # Native LIM addresses a tail relative to P, not modulo 256. An odd
        # page boundary (P=2176) must still put the first tail record at C.
        # Check the KV values a native plan would read after a prefill write.
        for length, prefix in ((2050, 2048), (2176, 2048), (2177, 2176)):
            with self.subTest(length=length):
                positions = torch.arange(length)
                _, dense, tail_slots, tail = c8_write_slots(
                    torch.ones_like(positions),
                    positions,
                    torch.full_like(positions, length),
                    2048,
                    3,
                )
                self.assertFalse(dense.any())
                self.assertEqual(tail.sum().item(), length - prefix)
                self.assertEqual(tail_slots[tail].unique().numel(), length - prefix)
                hbm = torch.full((2304,), -1, dtype=torch.int64)
                hbm[tail_slots[tail]] = positions[tail]
                torch.testing.assert_close(
                    hbm[2048 : 2048 + length - prefix],
                    torch.arange(prefix, length),
                )
                self.assertTrue(torch.all(hbm[:2048] == -1))
        _, dense, _, tail = c8_write_slots(
            torch.tensor([1, 1, 0, 3]),
            torch.tensor([0, 129, 0, 0]),
            torch.tensor([130, 130, 130, 130]),
            2048,
            3,
        )
        self.assertEqual(dense.tolist(), [True, True, False, False])
        self.assertEqual(tail.tolist(), [False, True, False, False])

    def test_c8_budget_charges_tail_and_bf16_compute_workspaces(self):
        budget = resolve_sparse_kv_budget(
            request_rows=3,
            context_len=4097,
            num_layers=2,
            top_k=2048,
            kv_bytes_per_token=656,
            copy_sfa=True,
            copy_sfa_query_heads=8,
            graph_batch_capacity=16,
            indexer_impl="lim",
            packed_c8=True,
        )
        self.assertEqual(budget.device_kv_bytes, 3 * 2 * 2304 * 656)
        self.assertEqual(budget.host_kv_bytes, 3 * 2 * 4224 * 656)
        self.assertGreaterEqual(budget.copy_sfa_bytes, 16 * 8 * (512 + 576) * 2)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            resolve_sparse_kv_budget(
                request_rows=3,
                context_len=4097,
                num_layers=2,
                top_k=2048,
                kv_bytes_per_token=656,
                copy_sfa=True,
                copy_sfa_query_heads=8,
                graph_batch_capacity=16,
                indexer_impl="lim",
                packed_c8=True,
                hbm_budget_bytes=budget.managed_hbm_bytes - 1,
            )
        with self.assertRaisesRegex(ValueError, "source pages"):
            resolve_sparse_kv_budget(
                request_rows=3,
                context_len=262145,
                num_layers=2,
                top_k=2048,
                kv_bytes_per_token=656,
                copy_sfa=True,
                indexer_impl="lim",
                packed_c8=True,
            )

        for rows, context, capacity in ((64, 131072, 2048), (202, 4096, 32384)):
            with self.subTest(rows=rows, context=context, capacity=capacity):
                with self.assertRaisesRegex(ValueError, "uint32 byte addressing"):
                    resolve_sparse_kv_budget(
                        request_rows=rows,
                        context_len=context,
                        num_layers=1,
                        top_k=2048,
                        kv_bytes_per_token=656,
                        cache_tokens=capacity,
                        copy_sfa=True,
                        indexer_impl="lim",
                        packed_c8=True,
                    )

    def test_prefill_gather_shape_ignores_padding_and_checks_host_bounds(self):
        # Host row order and total must match the ragged KV payload, including
        # interior padding. Device-side nonzero would require a stream sync.
        keep, total = prepare_prefill_gather_rows(
            torch.tensor([3, -1, 1, 0]), torch.tensor([4, 2, 3, 0]), 4, 8
        )
        self.assertEqual(keep.tolist(), [0, 2])
        self.assertEqual(total, 7)
        keep, total = prepare_prefill_gather_rows(
            torch.tensor([0, -1]), torch.tensor([0, 3]), 4, 8
        )
        self.assertEqual((keep.numel(), total), (0, 0))
        for requests, lengths, error in (
            ([4], [1], "outside sparse pool"),
            ([1], [9], "exceeds sparse context"),
        ):
            with self.subTest(requests=requests, lengths=lengths):
                with self.assertRaisesRegex(RuntimeError, error):
                    prepare_prefill_gather_rows(
                        torch.tensor(requests), torch.tensor(lengths), 4, 8
                    )

    def test_copy_sfa_shares_native_miss_stride_and_masks_interior_padding(self):
        batch, top_k = 3, 2048
        ints = lambda *shape: torch.empty(*shape, dtype=torch.int32)
        manager = SimpleNamespace(
            max_context_len=4224,
            planner_topk_src_ids=ints(batch, 1, top_k),
            planner_topk_dst_slots=ints(batch, 1, top_k),
            planner_topk_miss_counts=ints(batch),
            planner_miss_src_ids=ints(batch, top_k),
            planner_miss_dst_slots=ints(batch, top_k),
            planner_miss_counts=ints(batch),
            planner_actual_seq_lengths_query=torch.arange(
                1, batch + 1, dtype=torch.int32
            ),
            planner_actual_seq_lengths_kv=ints(batch),
            planner_num_cache_tokens=ints(batch),
            hbm_page_offsets=torch.arange(16, dtype=torch.int32),
            dram_page_offsets=torch.arange(33, dtype=torch.int32),
            planner_hbm_block_table=ints(batch, 16),
            planner_dram_block_table=ints(batch, 33),
        )
        adapter = LIMDecodeAdapter.__new__(LIMDecodeAdapter)
        adapter.top_k = adapter.capacity = top_k
        sources = torch.arange(batch * 32768, dtype=torch.int32).view(batch, 32768)
        plan = SimpleNamespace(
            source_ids=torch.arange(top_k, dtype=torch.int32).expand(batch, -1).clone(),
            device_slots=torch.arange(top_k, dtype=torch.int32)
            .expand(batch, -1)
            .clone(),
            topk_misses=torch.tensor([3, 99, 7], dtype=torch.int32),
            miss_sources=sources,
            miss_slots=sources + 1,
            miss_counts=torch.tensor([3, 99, 7], dtype=torch.int32),
            valid_rows=torch.tensor([True, False, True]),
            sparse_rows=torch.tensor([True, False, True]),
            request_slots=torch.tensor([3, 4, 1], dtype=torch.int32),
        )
        plan.source_ids[1].fill_(-1)
        metadata = adapter.copy_sfa_metadata(manager, plan)
        self.assertTrue(all(t.is_contiguous() for t in metadata))
        torch.testing.assert_close(metadata[3][2], sources[2])
        for actual, native in zip(
            metadata[:6],
            (
                plan.source_ids,
                plan.device_slots,
                plan.topk_misses,
                plan.miss_sources,
                plan.miss_slots,
                plan.miss_counts,
            ),
        ):
            self.assertEqual(actual.data_ptr(), native.data_ptr())
        self.assertEqual(metadata[5].tolist(), [3, 0, 7])
        self.assertEqual(metadata[7].tolist(), [2048, 0, 2048])
        self.assertEqual(metadata[2][1].item(), 0)
        self.assertEqual(metadata[9][:, 0].tolist(), [48, 0, 16])
        self.assertEqual(metadata[10][:, 0].tolist(), [99, 0, 33])
        plan.source_ids[0, 0] = 19
        self.assertEqual(metadata[0][0, 0, 0].item(), 19)

    def test_padded_graph_scratch_keeps_request_bounds_and_avoids_tail_workspace(self):
        rows, batch, top_k = 3, 16, 2048
        key = torch.empty((32, 128, 1, 128), dtype=torch.bfloat16)
        manager = SimpleNamespace(
            indexer_impl="li_reference",
            size=rows,
            batch_capacity=batch,
            device_cache_size=top_k,
            sparse_context_len=top_k,
            copy_sfa_enabled=True,
            max_context_len=4096,
            layer_num=2,
            head_num=1,
            head_dim=576,
            store_dtype=torch.bfloat16,
            device="cpu",
            start_layer=0,
            paged_kv_cache=SimpleNamespace(get_index_k_buffer=lambda _: key),
        )
        adapter = LIMDecodeAdapter(manager)
        self.assertIsNone(adapter.combined_kv)
        self.assertIsNone(adapter.combined_ids)
        self.assertEqual(adapter.last_prefix.shape, (2, rows + 1))
        self.assertEqual(adapter.source_ids.shape[0], batch)
        req_ids = torch.tensor([1, 2, 99] + [0] * (batch - 3))
        lengths = torch.tensor([2049, 2177, 4097] + [0] * (batch - 3))
        forward_batch = SimpleNamespace(
            req_pool_indices=req_ids,
            seq_lens=lengths,
            seq_lens_cpu=lengths,
        )

        def reference_indexer(**_kwargs):
            return torch.arange(top_k, dtype=torch.int32).expand(batch, 1, -1)

        plan = adapter.select(
            manager,
            query=torch.empty((batch, 32, 128), dtype=torch.bfloat16),
            weights=torch.empty((batch, 32), dtype=torch.bfloat16),
            key=key,
            block_table=torch.zeros((batch, 32), dtype=torch.int32),
            forward_batch=forward_batch,
            layer_id=0,
            reference_indexer=reference_indexer,
        )
        self.assertEqual(plan.request_slots.tolist(), [1, 2] + [rows] * (batch - 2))
        self.assertTrue(torch.all(plan.source_ids[2:] == -1))
        budget = resolve_sparse_kv_budget(
            request_rows=rows,
            graph_batch_capacity=batch,
            context_len=4096,
            num_layers=2,
            top_k=top_k,
            kv_bytes_per_token=1152,
            indexer_impl="li_reference",
            copy_sfa=True,
        )
        allocated = sum(
            tensor.numel() * tensor.element_size()
            for tensor in vars(adapter).values()
            if isinstance(tensor, torch.Tensor)
        )
        self.assertEqual(budget.adapter_bytes, allocated)

    def test_graph_budget_grows_scratch_without_growing_residency(self):
        options = dict(
            request_rows=9,
            context_len=32768,
            num_layers=4,
            top_k=2048,
            kv_bytes_per_token=1152,
            indexer_impl="lim",
            copy_sfa=True,
        )
        small = resolve_sparse_kv_budget(**options)
        padded = resolve_sparse_kv_budget(**options, graph_batch_capacity=16)
        self.assertEqual(sparse_batch_capacity(9, 16), 16)
        self.assertEqual(small.device_kv_bytes, padded.device_kv_bytes)
        self.assertEqual(small.slot_map_bytes, padded.slot_map_bytes)
        self.assertEqual(small.host_kv_bytes, padded.host_kv_bytes)
        self.assertEqual(padded.selected_kv_bytes, 16 * 2048 * 1152)
        self.assertGreater(padded.adapter_bytes, small.adapter_bytes)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            resolve_sparse_kv_budget(
                **options,
                graph_batch_capacity=16,
                hbm_budget_bytes=small.managed_hbm_bytes,
            )

    def test_full_visible_q1_transition_padding_and_reuse(self):
        previous = torch.tensor([-1, -1, 2048, 3000, -1], dtype=torch.int32)
        result = prepare_lim_rows(
            torch.tensor([2048, 2049, 2177, 0]),
            torch.tensor([1, 2, 3, 0]),
            previous,
            2048,
            4,
            full_visible=True,
        )
        self.assertEqual(result[2].tolist(), [2048, 2049, 2177, 0])
        self.assertEqual(result[5].tolist(), [-2, -1, -2, -3])
        self.assertEqual(result[0].tolist(), [1, 2, 3, 4])

    def test_page_boundary_and_cold_to_warm(self):
        previous = torch.full((5,), -1, dtype=torch.int32)
        slots = torch.tensor([1, 2, 3])
        lengths = torch.tensor([2048, 2049, 2176])
        result = prepare_lim_rows(lengths, slots, previous, 2048, 4)
        self.assertEqual(result[2].tolist(), [1920, 2048, 2048])
        self.assertEqual(result[5].tolist(), [-3, -2, -2])
        previous[result[0].long()] = result[6]
        result = prepare_lim_rows(lengths + 1, slots, previous, 2048, 4)
        self.assertEqual(result[2].tolist(), [2048, 2048, 2176])
        self.assertEqual(result[5].tolist(), [-2, -1, -1])

    def test_padding_reset_and_rollback(self):
        previous = torch.tensor([-1, 8192, -1, -1, -1], dtype=torch.int32)
        slots = torch.tensor([1, 0, -1, 99])
        result = prepare_lim_rows(
            torch.tensor([4097, 100, 0, 9000]), slots, previous, 4096, 4
        )
        self.assertEqual(result[0].tolist(), [1, 4, 4, 4])
        self.assertEqual(result[1].tolist(), [4097, 0, 0, 0])
        self.assertEqual(result[5].tolist(), [-2, -3, -3, -3])
        previous[1] = -1  # request pool slot was freed/reused
        result = prepare_lim_rows(
            torch.tensor([8193]), torch.tensor([1]), previous, 4096, 4
        )
        self.assertEqual(result[5].tolist(), [-2])

    def test_copy_sfa_budget_includes_planner_and_page_aligned_host_rows(self):
        budget = resolve_sparse_kv_budget(
            request_rows=3,
            context_len=4097,
            num_layers=4,
            top_k=2048,
            kv_bytes_per_token=1152,
            indexer_impl="lim",
            copy_sfa=True,
            copy_sfa_query_heads=8,
        )
        self.assertEqual(budget.cache_tokens, 2048)
        self.assertEqual(budget.host_kv_bytes, 3 * 4 * 4224 * 1152)
        self.assertGreater(budget.copy_sfa_bytes, 4 * 4 * 2048 * 4)
        with self.assertRaisesRegex(ValueError, "K=C=2048"):
            resolve_sparse_kv_budget(
                request_rows=3,
                context_len=4097,
                num_layers=4,
                top_k=2048,
                kv_bytes_per_token=1152,
                cache_tokens=4096,
                copy_sfa=True,
            )

    def test_native_budget_accounts_actual_persistent_buffers(self):
        kwargs = dict(
            request_rows=3,
            context_len=32896,
            num_layers=4,
            top_k=2048,
            kv_bytes_per_token=1152,
            indexer_impl="lim",
        )
        budget = resolve_sparse_kv_budget(**kwargs, cache_tokens=4096)
        # C>K is managed by the native pool; no duplicate Python LRU arrays.
        self.assertEqual(budget.policy_bytes, 2048 * 8)
        self.assertEqual(budget.slot_map_bytes, 4 * 4 * 32896 * 4)
        self.assertGreater(budget.adapter_bytes, 3 * (2048 + 128) * 1152)
        exact = resolve_sparse_kv_budget(
            **kwargs, hbm_budget_bytes=budget.managed_hbm_bytes
        )
        self.assertEqual(exact.cache_tokens, 4096)
        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            resolve_sparse_kv_budget(**kwargs, cache_tokens=4097)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            resolve_sparse_kv_budget(
                **kwargs,
                cache_tokens=4096,
                hbm_budget_bytes=budget.managed_hbm_bytes - 1,
            )


class TestLIMKernelLoad(CustomTestCase):
    """Package registration must replace the optional external binary path."""

    def test_packaged_loader_preserves_in_place_outputs(self):
        package = ModuleType("sgl_kernel_npu")
        package.__path__ = []
        api = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload")

        def native(*tensors):
            # Emulate the native boundary, not the framework's tensor ordering.
            tensors[13].copy_(torch.arange(2048, dtype=torch.int32).view(1, 1, -1))
            tensors[18].copy_(tensors[9])

        api.fused_lightning_indexer_manage = native
        modules = {"sgl_kernel_npu": package, api.__name__: api}
        registration = SimpleNamespace(
            fused_lightning_indexer_manage=SimpleNamespace(default=native)
        )
        with (
            patch.dict("sys.modules", modules),
            patch.object(torch.ops, "npu", registration),
        ):
            operator = load_fused_lim_operator()
        ints = lambda *shape: torch.zeros(*shape, dtype=torch.int32)
        ids, counts = ints(1, 1, 2048), ints(1)
        fused_lightning_indexer_manage(
            weights=torch.ones(1, 32, dtype=torch.bfloat16),
            query_scale=torch.ones(1, 32),
            query=torch.ones(1, 32, 128, dtype=torch.bfloat16),
            key_scale=torch.ones(33, 128, 1),
            key=torch.ones(33, 128, 1, 128, dtype=torch.bfloat16),
            block_table=ints(1, 33),
            query_ends=ints(1),
            seq_lens=ints(1),
            prefix_lens=ints(1),
            cache_sizes=torch.tensor([4096], dtype=torch.int32),
            request_state=ints(1),
            request_slots=ints(1),
            slot_pool=ints(1, 4224),
            source_ids=ids,
            device_slots=ints(1, 1, 2048),
            topk_misses=ints(1),
            miss_sources=ints(1, 32768),
            miss_slots=ints(1, 32768),
            miss_counts=counts,
            operator=operator,
        )
        self.assertEqual(counts.item(), 4096)
        torch.testing.assert_close(ids.view(-1), torch.arange(2048, dtype=torch.int32))

    def test_missing_native_registration_fails_before_graph(self):
        package = ModuleType("sgl_kernel_npu")
        package.__path__ = []
        api = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload")
        api.fused_lightning_indexer_manage = lambda *args: None
        with (
            patch.dict("sys.modules", {"sgl_kernel_npu": package, api.__name__: api}),
            patch.object(torch.ops, "npu", SimpleNamespace()),
        ):
            with self.assertRaisesRegex(RuntimeError, "matching kernel wheel"):
                load_fused_lim_operator()


if __name__ == "__main__":
    unittest.main()
