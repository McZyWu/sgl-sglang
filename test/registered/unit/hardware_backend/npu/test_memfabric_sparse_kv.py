"""CPU checks of MF shared allocation and byte-addressed split-KV transfers."""

import ctypes
import sys
import types
import unittest
from unittest.mock import patch

import torch
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import memfabric_pool
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.memfabric_pool import (
    MemFabricKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class _Group:
    world_size = 2

    def __init__(self, rank, messages):
        self.rank_in_group = rank
        self.messages = messages
        self.cursor = 0

    def broadcast_object(self, value):
        if self.rank_in_group == 0:
            self.messages.append(value)
        else:
            value = self.messages[self.cursor]
        self.cursor += 1
        return value

    def barrier(self):
        pass


class _Offload:
    OffloadConfig = types.SimpleNamespace
    Scene = types.SimpleNamespace(SHARED="shared")

    def __init__(self):
        self.configs = []
        self.allocations = []

    def initialize(self, config):
        self.configs.append(config)
        return 0

    def empty(self, shape, dtype, pin_memory):
        tensor = torch.empty(shape, dtype=dtype)
        self.allocations.append(tensor)
        return tensor

    def sparse_copy(self, sources, destinations, lengths, count, device):
        # Emulate the MF boundary with actual memory transfers, so malformed
        # K/V byte strides, masking or ordering produce incorrect payloads.
        for src, dst, length in zip(
            sources[: count.item()].tolist(),
            destinations[: count.item()].tolist(),
            lengths[: count.item()].tolist(),
        ):
            if length:
                ctypes.memmove(dst, src, length)
        return 0


class TestMemFabricSparseKV(CustomTestCase):
    def make_pool(self):
        pool = MemFabricKVPool.__new__(MemFabricKVPool)
        pool.offload = _Offload()
        return pool

    def test_tp_ranks_alias_one_aligned_split_pool(self):
        offload = _Offload()
        messages = []
        module = types.ModuleType("memfabric_hybrid")
        module.offload = offload
        with (
            patch.dict(sys.modules, {"memfabric_hybrid": module}),
            patch.object(memfabric_pool.socket, "socket") as socket_factory,
            patch.object(
                torch,
                "npu",
                types.SimpleNamespace(current_device=lambda: 0),
                create=True,
            ),
        ):
            socket_factory.return_value.__enter__.return_value.getsockname.return_value = (
                "127.0.0.1",
                8500,
            )
            owner = MemFabricKVPool(
                tokens=256,
                layers=2,
                widths=(512, 64),
                dtype=torch.bfloat16,
                device="cpu",
                group=_Group(0, messages),
            )
            allocation_count = len(offload.allocations)
            peer = MemFabricKVPool(
                tokens=256,
                layers=2,
                widths=(512, 64),
                dtype=torch.bfloat16,
                device="cpu",
                group=_Group(1, messages),
            )
        self.assertEqual(len(offload.allocations), allocation_count)
        self.assertGreater(offload.configs[0].alloc_size, 0)
        self.assertEqual(offload.configs[1].alloc_size, 0)
        self.assertEqual(offload.configs[0].store_url, offload.configs[1].store_url)
        self.assertEqual(
            offload.configs[0].reserve_size, offload.configs[1].reserve_size
        )
        for owner_component, peer_component in zip(owner.buffers, peer.buffers):
            for host, shared in zip(owner_component, peer_component):
                self.assertEqual(host.data_ptr() % owner.alignment, 0)
                self.assertEqual(host.data_ptr(), shared.data_ptr())
                host.fill_(1.25)
                torch.testing.assert_close(host, shared)
        self.assertNotEqual(
            owner.buffers[0][0].data_ptr(), owner.buffers[1][0].data_ptr()
        )

    def test_split_copy_masks_padding_and_preserves_unselected_rows(self):
        pool = self.make_pool()
        sources = [
            torch.arange(5 * width).view(5, 1, width).to(torch.float16)
            for width in (512, 64)
        ]
        destinations = [
            torch.full((2, 4, 1, width), -7, dtype=torch.float16) for width in (512, 64)
        ]
        src = torch.tensor([4, -1, 1])
        dst = torch.tensor([6, -1, 2])
        valid = torch.tensor([True, False, True])
        pool.copy_rows(sources, destinations, src, dst, valid)
        for source, destination in zip(sources, destinations):
            flat = destination.view(8, 1, -1)
            torch.testing.assert_close(flat[[6, 2]], source[[4, 1]])
            self.assertTrue(torch.all(flat[[0, 1, 3, 4, 5, 7]] == -7))

    def test_overlapping_move_stages_both_planes_before_writing(self):
        pool = self.make_pool()
        host = [
            torch.arange(8 * width).view(1, 8, 1, width).to(torch.float16)
            for width in (512, 64)
        ]
        original = [x.clone() for x in host]
        scratch = [torch.empty(3, 1, width, dtype=torch.float16) for width in (512, 64)]
        src, dst, rows = (
            torch.tensor([2, 3, 4]),
            torch.tensor([3, 4, 2]),
            torch.arange(3),
        )
        valid = torch.ones(3, dtype=torch.bool)
        pool.copy_rows(host, scratch, src, rows, valid)
        pool.copy_rows(scratch, host, rows, dst, valid)
        for before, after in zip(original, host):
            torch.testing.assert_close(after[0, dst], before[0, src])

    def test_pd_page_layout_transfers_prefill_planes_into_decode_dram(self):
        pool = self.make_pool()
        pool.buffers = [
            [torch.full((4, 128, 1, width), -1, dtype=torch.bfloat16)]
            for width in (512, 64)
        ]
        pointers, sizes, page_bytes = pool.get_transfer_buf_infos()
        self.assertEqual(page_bytes, [128 * 512 * 2, 128 * 64 * 2])
        self.assertEqual(sizes, [4 * size for size in page_bytes])
        sources = [
            torch.arange(4, dtype=torch.bfloat16)[:, None, None, None]
            .expand(4, 128, 1, width)
            .contiguous()
            for width in (512, 64)
        ]
        for source, dst_base, length in zip(sources, pointers, page_bytes):
            for src_page, dst_page in ((3, 1), (0, 2)):
                ctypes.memmove(
                    dst_base + dst_page * length,
                    source.data_ptr() + src_page * length,
                    length,
                )
        for component in pool.buffers:
            self.assertTrue(torch.all(component[0][1] == 3))
            self.assertTrue(torch.all(component[0][2] == 0))
            self.assertTrue(torch.all(component[0][[0, 3]] == -1))

    def test_rejects_noncontiguous_or_incompatible_components(self):
        pool = self.make_pool()
        rows, valid = torch.arange(2), torch.ones(2, dtype=torch.bool)
        source = torch.zeros(2, 1, 576)[..., :512]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            pool.copy_rows([source], [source.clone()], rows, rows, valid)
        with self.assertRaisesRegex(ValueError, "dtypes"):
            pool.copy_rows(
                [source.clone()], [source.to(torch.float16)], rows, rows, valid
            )
        with self.assertRaisesRegex(ValueError, "component"):
            pool.copy_rows([source], [], rows, rows, valid)

    def test_sparse_copy_failure_is_reported(self):
        pool = self.make_pool()
        pool.offload.sparse_copy = lambda *args: -5
        tensor = torch.zeros(2, 1, 64)
        with self.assertRaisesRegex(RuntimeError, "-5"):
            pool.copy_rows(
                [tensor],
                [tensor.clone()],
                torch.arange(2),
                torch.arange(2),
                torch.ones(2, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
