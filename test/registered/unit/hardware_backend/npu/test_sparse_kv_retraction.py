"""Retraction snapshots for MF-backed split MLA KV."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestSparseKVRetraction(CustomTestCase):
    def test_native_kv_without_index_and_empty_snapshot(self):
        pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
        pool.k_buffer = torch.arange(24, dtype=torch.float32).view(1, 2, 4, 1, 3)
        pool.v_buffer = torch.arange(16, dtype=torch.float32).view(1, 2, 4, 1, 2)
        pool.indexer_layer_id_to_slot = {}
        pool.index_k_scale_buffer = None
        pool.start_layer, pool.layer_num = 0, 1
        pool.dsa_kv_cache_store_fp8 = False
        pool.is_draft_worker, pool.dcp_size, pool.dcp_rank = True, 1, 0
        pool.page_size, pool.cpu_offloading_chunk_size = 4, 2
        old, new = torch.tensor([1, 5]), torch.tensor([6, 3])
        want = [
            x.flatten(0, 1)[old].clone() for x in (pool.k_buffer[0], pool.v_buffer[0])
        ]
        with patch.object(
            torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True
        ):
            empty = pool.get_cpu_copy(old[:0])
            pool.load_cpu_copy(empty, new[:0])
            snapshot = pool.get_cpu_copy(old)
            pool.k_buffer.zero_()
            pool.v_buffer.zero_()
            pool.load_cpu_copy(snapshot, new)
        self.assertEqual(empty, [[]])
        for buf, expected in zip((pool.k_buffer[0], pool.v_buffer[0]), want):
            torch.testing.assert_close(buf.flatten(0, 1)[new], expected, rtol=0, atol=0)

    def test_shared_split_snapshot_and_new_slots(self):
        k = [torch.arange(24, dtype=torch.bfloat16).view(2, 4, 1, 3)]
        v = [torch.arange(16, dtype=torch.bfloat16).view(2, 4, 1, 2)]
        index = torch.arange(16, dtype=torch.uint8).view(1, 2, 4, 1, 2)
        scale = torch.arange(8, dtype=torch.float32).view(1, 2, 4, 1)
        pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
        published, resets = [], []

        def copy_rows(sources, destinations, source_rows, destination_rows, valid):
            for src, dst in zip(sources, destinations):
                dst[destination_rows[valid]] = src[source_rows[valid]]

        host_pool = SimpleNamespace(
            rank=0, publish=lambda: published.append(True), copy_rows=copy_rows
        )
        pool.sparse_kv_manager = SimpleNamespace(
            host_pool=host_pool,
            host_k_buffer=k,
            host_v_buffer=v,
            reset_requests=lambda ids: resets.append(ids),
        )
        pool.k_buffer = pool.v_buffer = None
        pool.index_k_buffer, pool.index_k_scale_buffer = index, scale
        pool.indexer_layer_id_to_slot = {10: 0}
        pool.start_layer, pool.layer_num = 10, 1
        pool.dsa_kv_cache_store_fp8 = True
        pool.is_draft_worker, pool.dcp_size, pool.dcp_rank = False, 2, 0
        pool.page_size, pool.cpu_offloading_chunk_size = 4, 2
        pool.device = "cpu"
        old = torch.tensor([5, 1, 6])
        new = torch.tensor([0, 7, 2])
        expected = [
            x.flatten(0, 1)[old].clone() for x in (k[0], v[0], index[0], scale[0])
        ]
        with patch.object(
            torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True
        ):
            snapshot = pool.get_cpu_copy(old)
            for buf in (k[0], v[0], index[0], scale[0]):
                buf.zero_()
            pool.load_cpu_copy(snapshot, new, req_pool_index=3)
        for buf, want in zip((k[0], v[0], index[0], scale[0]), expected):
            torch.testing.assert_close(buf.flatten(0, 1)[new], want, rtol=0, atol=0)
        self.assertEqual(published, [True])
        self.assertEqual(resets, [[3]])

        # A non-owner restores its own index state, but cannot overwrite the
        # shared payload restored by rank zero.
        host_pool.rank = 1
        k[0].fill_(42)
        v[0].fill_(43)
        index.zero_()
        scale.zero_()
        with patch.object(
            torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True
        ):
            pool.load_cpu_copy(snapshot, new, req_pool_index=4)
        self.assertTrue(torch.all(k[0] == 42))
        self.assertTrue(torch.all(v[0] == 43))
        torch.testing.assert_close(
            index[0].flatten(0, 1)[new], expected[2], rtol=0, atol=0
        )
        torch.testing.assert_close(
            scale[0].flatten(0, 1)[new], expected[3], rtol=0, atol=0
        )
        self.assertEqual(resets, [[3], [4]])


if __name__ == "__main__":
    unittest.main()
