"""MF-backed overlapping accepted-token relocation, eager and graph replay."""

import gc
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    SparseKVCacheManager,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.memfabric_pool import (
    MemFabricKVPool,
)


@pytest.fixture(autouse=True)
def _cleanup_mf_entity():
    yield
    from memfabric_hybrid import offload

    torch.npu.synchronize()
    gc.collect()
    offload.uninitialize()


class _SingleRankGroup:
    rank_in_group = 0
    world_size = 1

    def broadcast_object(self, value):
        return value

    def broadcast(self, value, src=0):
        return value

    def barrier(self):
        pass


def test_paged_host_overlap_move_real_npu():
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    device_id = torch.npu.current_device()
    manager = SparseKVCacheManager.__new__(SparseKVCacheManager)
    manager.device = torch.device("npu", device_id)
    manager.head_num, manager.head_dim = 1, 576
    manager.kv_lora_rank, manager.qk_rope_head_dim = 512, 64
    manager.store_dtype = torch.bfloat16
    manager.host_pool = MemFabricKVPool(
        tokens=512,
        layers=1,
        widths=(512, 64),
        dtype=torch.bfloat16,
        device=manager.device,
        group=_SingleRankGroup(),
    )
    manager.host_k_buffer, manager.host_v_buffer = manager.host_pool.buffers
    pool = SimpleNamespace(size=512, page_size=0, sparse_kv_manager=manager)
    original = [
        torch.arange(512, dtype=torch.float32)
        .remainder(127)
        .to(torch.bfloat16)[:, None, None]
        .expand(512, 1, width)
        .clone()
        for width in (512, 64)
    ]
    hosts = [manager.host_k_buffer[0], manager.host_v_buffer[0]]
    source = torch.tensor([40, 41, 42], device=manager.device)
    target = torch.tensor([41, 42, 40], device=manager.device)

    def reset():
        for host, values in zip(hosts, original):
            host.view_as(values).copy_(values)

    def check():
        torch.npu.synchronize()
        for host, values in zip(hosts, original):
            torch.testing.assert_close(
                host.view_as(values)[[41, 42, 40]], values[[40, 41, 42]], rtol=0, atol=0
            )

    try:
        reset()
        NPUMLATokenToKVPool.move_kv_cache(pool, target, source)
        check()
        reset()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            NPUMLATokenToKVPool.move_kv_cache(pool, target, source)
        torch.npu.synchronize()
        reset()
        graph.replay()
        check()
    finally:
        torch.npu.synchronize()
        # Run this hardware test in its own worker/process: MF owns a
        # process-wide offload entity, as it does in the serving worker.
