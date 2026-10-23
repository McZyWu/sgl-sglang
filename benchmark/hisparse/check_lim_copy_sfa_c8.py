"""A5 C8: real manager/registered host KV, IndexShare, padding and graph replay."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.ascend_backend import (  # noqa: F401
    ForwardMetadata,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.attention import (
    forward_sparsity_driven_kv_offload,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
    SparseKVCacheManager,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class IndexBoundary(MLATokenToKVPool):
    """Fixed projection/index-cache fixture; production manager and attention run."""

    def __init__(self, keys, scales):
        self.key, self.scale = keys, scales
        self.dtype = self.store_dtype = torch.float8_e4m3fn
        self.dsa_kv_cache_store_fp8 = True
        self.kv_lora_rank, self.qk_rope_head_dim = 512, 64
        self.start_layer, self.layer_num = 0, 2

    def get_index_k_buffer(self, layer_id):
        return self.key

    def get_index_k_scale_buffer(self, layer_id):
        return self.scale

    _pack_dsa_fp8_kv_cache = NPUMLATokenToKVPool._pack_dsa_fp8_kv_cache


def unpack(rows):
    raw = rows.contiguous().view(torch.uint8)
    latent = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    rope = raw[:, 512:640].contiguous().view(torch.bfloat16).float()
    scales = raw[:, 640:].contiguous().view(torch.float32)
    latent = (latent * scales.repeat_interleave(128, dim=-1)).to(torch.bfloat16).float()
    return latent, rope


def run(output):
    if not is_npu_arch35():
        raise RuntimeError("C8 framework validation requires Ascend950 (A5).")
    from sgl_kernel_npu.sparsity_driven_kv_offload import free_shm

    torch.npu.set_device(0)
    batch_size, context, heads = 8, 4352, 8
    generator = torch.Generator().manual_seed(20261023)
    keys = torch.zeros(3, context, 128, dtype=torch.bfloat16)
    for request, length in ((1, 2050), (2, 4098)):
        keys[request, :, 0] = -8
        keys[request, length - 2048 : length, 0] = 8
    keys = keys.to(device="npu", dtype=torch.float8_e4m3fn).view(-1, 128, 1, 128)
    scales = torch.ones(keys.shape[:3], device="npu")
    index = IndexBoundary(keys, scales)
    pool = ReqToTokenPool(2, context, "npu", False)
    with (
        envs.SGLANG_NPU_SPARSE_KV_INDEXER.override("lim"),
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
        envs.SGLANG_NPU_ENABLE_COPY_SFA.override(True),
        envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.override(2048),
        envs.SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB.override(0),
    ):
        manager = SparseKVCacheManager(
            pool,
            SimpleNamespace(get_kvcache=lambda: index),
            2048,
            copy_sfa_padded_query_heads=heads,
            graph_batch_capacity=batch_size,
        )
        try:
            for layer in range(2):
                layer_desc = SimpleNamespace(layer_id=layer)
                lengths = torch.tensor([2050, 4098], dtype=torch.int32)
                prefill = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=2,
                    req_pool_indices=torch.tensor(
                        [1, 2], dtype=torch.int64, device="npu"
                    ),
                    req_pool_indices_cpu=torch.tensor([1, 2]),
                    seq_lens=lengths.to("npu"),
                    seq_lens_cpu=lengths,
                    extend_prefix_lens=torch.zeros(2, dtype=torch.int64, device="npu"),
                    extend_seq_lens=lengths.to("npu"),
                    extend_seq_lens_cpu=lengths.tolist(),
                    out_cache_loc=torch.arange(6148, device="npu"),
                )
                payload = torch.randn(6148, 1, 576, generator=generator).to(
                    device="npu", dtype=torch.bfloat16
                )
                manager.offload_v2(
                    payload[..., :512].contiguous(),
                    payload[..., 512:].contiguous(),
                    layer_desc,
                    prefill,
                    torch.npu.current_stream(),
                )
            requests = torch.tensor(
                [1, 0, 2, 0, 0, 0, 0, 0], dtype=torch.int32, device="npu"
            )
            lengths = torch.tensor(
                [2050, 0, 4098, 0, 0, 0, 0, 0], dtype=torch.int32, device="npu"
            )
            table = requests[:, None] * (context // 128) + torch.arange(
                context // 128, dtype=torch.int32, device="npu"
            )
            forward = SimpleNamespace(
                batch_size=batch_size,
                forward_mode=ForwardMode.DECODE,
                req_pool_indices=requests,
                seq_lens=lengths,
                out_cache_loc=torch.arange(batch_size, device="npu"),
            )
            q_index = torch.zeros(batch_size, 32, 128)
            q_index[..., 0] = 1
            q_index = q_index.to(device="npu", dtype=torch.float8_e4m3fn)
            weights = torch.full(
                (batch_size, 32), 1 / 32, dtype=torch.bfloat16, device="npu"
            )
            q_scale = torch.ones(batch_size, 32, device="npu")
            q = torch.randn(batch_size, heads, 512, generator=generator).to(
                device="npu", dtype=torch.bfloat16
            )
            rope = torch.randn(batch_size, heads, 64, generator=generator).to(
                device="npu", dtype=torch.bfloat16
            )
            backend = SimpleNamespace(
                sparse_kv_manager=manager,
                device="npu",
                forward_metadata=SimpleNamespace(sparse_kv_lim_plan=None),
            )
            layer = SimpleNamespace(layer_id=0, tp_q_head_num=heads, scaling=576**-0.5)
            result = [None]

            def step():
                plan = manager.lim_adapter.select(
                    manager,
                    query=q_index,
                    query_scale=q_scale,
                    key_scale=scales,
                    weights=weights,
                    key=keys,
                    block_table=table,
                    forward_batch=forward,
                    layer_id=0,
                )
                backend.forward_metadata.sparse_kv_lim_plan = plan
                for i in range(2):
                    layer.layer_id = i
                    result[0] = forward_sparsity_driven_kv_offload(
                        backend,
                        q,
                        q[:, :1],
                        None,
                        layer,
                        forward,
                        save_kv_cache=False,
                        q_rope=rope,
                        k_rope=rope[:, :1],
                        topk_indices=plan.source_ids,
                    )

            def verify():
                plan = backend.forward_metadata.sparse_kv_lim_plan
                for row, request, length in ((0, 1, 2050), (2, 2, 4098)):
                    ids = plan.source_ids[row].cpu().long()
                    slots = plan.device_slots[row].cpu().long()
                    assert set(ids.tolist()) == set(range(length - 2048, length))
                    for i in range(2):
                        raw = manager.host_kv_buffer[i][request, ids, 0]
                        torch.testing.assert_close(
                            manager.device_kv_buffer[i][request, slots, 0].cpu(),
                            raw,
                            rtol=0,
                            atol=0,
                        )
                    latent, key_rope = unpack(
                        manager.host_kv_buffer[-1][request, ids, 0]
                    )
                    score = (
                        q[row].cpu().float() @ latent.T
                        + rope[row].cpu().float() @ key_rope.T
                    ) * layer.scaling
                    expected = score.softmax(-1) @ latent
                    torch.testing.assert_close(
                        result[0][row].cpu().float().view(heads, 512),
                        expected,
                        rtol=0.05,
                        atol=0.008,
                    )
                assert torch.count_nonzero(result[0][requests == 0]) == 0

            step()
            torch.npu.synchronize()
            verify()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                step()
            for reset in (False, True, False):
                if reset:
                    manager.reset_requests([1, 2])
                graph.replay()
                torch.npu.synchronize()
                verify()
            record = {
                "status": "pass",
                "scope": "A5 C8 fixed projection framework; no model accuracy/timing claim",
                "request_rows": manager.size,
                "batch_capacity": manager.batch_capacity,
                "decode_path": manager.last_decode_path,
                "payload_bitwise": True,
                "attention_cpu_oracle": True,
                "graph_reset_and_padding": True,
                "index_share_layers": 2,
            }
            output.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record, indent=2))
        finally:
            torch.npu.synchronize()
            free_shm()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
