"""Small real MF/NPU/TP retraction probe; launch with torchrun on one host."""

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.memfabric_pool import (
    MemFabricKVPool,
)


class Group:
    def __init__(self, cpu_group):
        self.cpu_group = cpu_group
        self.rank_in_group = dist.get_rank()
        self.world_size = dist.get_world_size()

    def broadcast_object(self, value):
        values = [value]
        dist.broadcast_object_list(values, src=0, group=self.cpu_group)
        return values[0]

    def barrier(self):
        dist.barrier(group=self.cpu_group)

    def broadcast(self, value, src=0):
        dist.broadcast(value, src=src)


def bytes_equal(actual, expected):
    return torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    dist.init_process_group(backend="hccl")
    cpu_group = dist.new_group(backend="gloo")
    group = Group(cpu_group)
    rank = group.rank_in_group
    host = MemFabricKVPool(
        tokens=384,
        layers=2,
        widths=(512, 64),
        dtype=torch.bfloat16,
        device=device,
        group=group,
    )
    resets = []
    manager = SimpleNamespace(
        host_pool=host,
        host_k_buffer=host.buffers[0],
        host_v_buffer=host.buffers[1],
        reset_requests=lambda ids: resets.append(ids),
    )
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.sparse_kv_manager = manager
    pool.k_buffer = pool.v_buffer = None
    pool.index_k_buffer = torch.zeros(
        (2, 3, 128, 1, 128), dtype=torch.uint8, device=device
    )
    pool.index_k_scale_buffer = torch.zeros(
        (2, 3, 128, 1), dtype=torch.float32, device=device
    )
    pool.indexer_layer_id_to_slot = {10: 0, 11: 1}
    pool.start_layer, pool.layer_num = 10, 2
    pool.dsa_kv_cache_store_fp8 = True
    pool.is_draft_worker, pool.dcp_size, pool.dcp_rank = False, 1, 0
    pool.page_size, pool.cpu_offloading_chunk_size = 128, 2
    pool.device = device
    old = torch.tensor([1, 127, 128, 255, 256, 383], dtype=torch.int64, device=device)
    new = torch.tensor([2, 129, 126, 257, 254, 380], dtype=torch.int64, device=device)
    rows = torch.arange(old.numel(), dtype=torch.int64, device=device)
    valid = torch.ones_like(rows, dtype=torch.bool)
    expected_kv = []
    expected_index = []
    for layer in range(2):
        values = [
            (
                (
                    torch.arange(old.numel() * width, dtype=torch.float32).reshape(
                        -1, 1, width
                    )
                    + layer * 11
                    + component * 29
                )
                % 127
            ).to(torch.bfloat16)
            for component, width in enumerate((512, 64))
        ]
        expected_kv.append(values)
        if rank == 0:
            host.copy_rows(
                [x.to(device) for x in values],
                [host.buffers[c][layer] for c in range(2)],
                rows,
                old,
                valid,
            )
        index = (
            (
                torch.arange(old.numel() * 128).reshape(-1, 1, 128)
                + rank * 13
                + layer * 17
            )
            % 256
        ).to(torch.uint8)
        scale = (
            torch.arange(old.numel(), dtype=torch.float32).reshape(-1, 1)
            + rank
            + layer / 4
        )
        pool.index_k_buffer[layer].flatten(0, 1)[old] = index.to(device)
        pool.index_k_scale_buffer[layer].flatten(0, 1)[old] = scale.to(device)
        expected_index.append((index, scale))
    host.publish()
    torch.npu.synchronize()
    snapshot = pool.get_cpu_copy(old)
    checks = {}
    for layer in range(2):
        expected = [*expected_kv[layer], *expected_index[layer]]
        for chunk_index, chunk in enumerate(snapshot[layer]):
            start = chunk_index * pool.cpu_offloading_chunk_size
            stop = min(start + pool.cpu_offloading_chunk_size, len(old))
            checks[f"layer{layer}_snapshot_chunk{chunk_index}_components"] = len(
                chunk
            ) == len(expected)
            for component, (actual, want) in enumerate(zip(chunk, expected)):
                checks[
                    f"layer{layer}_snapshot_chunk{chunk_index}_component{component}"
                ] = bytes_equal(actual, want[start:stop])
    # No rank may overwrite shared old slots until all snapshots have been read.
    group.barrier()
    for layer in range(2):
        if rank == 0:
            zeros = [torch.zeros_like(x, device=device) for x in expected_kv[layer]]
            host.copy_rows(
                zeros, [host.buffers[c][layer] for c in range(2)], rows, old, valid
            )
        pool.index_k_buffer[layer].flatten(0, 1)[old] = 0
        pool.index_k_scale_buffer[layer].flatten(0, 1)[old] = 0
        if rank != 0:
            for chunk in snapshot[layer]:
                chunk[0].zero_()
                chunk[1].zero_()
    host.publish()
    torch.npu.synchronize()
    pool.load_cpu_copy(snapshot, new, req_pool_index=3)
    for layer in range(2):
        scratch = [torch.empty_like(x, device=device) for x in expected_kv[layer]]
        host.copy_rows(
            [host.buffers[c][layer] for c in range(2)], scratch, new, rows, valid
        )
        torch.npu.synchronize()
        for component, value in enumerate(scratch):
            checks[f"layer{layer}_kv{component}"] = bytes_equal(
                value.cpu(), expected_kv[layer][component]
            )
        checks[f"layer{layer}_index"] = bytes_equal(
            pool.index_k_buffer[layer].flatten(0, 1)[new].cpu(),
            expected_index[layer][0],
        )
        checks[f"layer{layer}_scale"] = bytes_equal(
            pool.index_k_scale_buffer[layer].flatten(0, 1)[new].cpu(),
            expected_index[layer][1],
        )
    checks["reset_spy"] = resets == [[3]]
    result = {
        "rank": rank,
        "world_size": group.world_size,
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    results = [None] * group.world_size
    dist.all_gather_object(results, result, group=cpu_group)
    if rank == 0:
        print(
            json.dumps(
                {
                    "summary": results,
                    "passed": all(x["passed"] for x in results),
                    "limitations": "reset spy only; no full manager, attention or PD validation",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    passed = all(x["passed"] for x in results)
    torch.npu.synchronize()
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
