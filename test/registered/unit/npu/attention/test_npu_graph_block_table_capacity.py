"""Exercise Ascend graph metadata with a request pool larger than the draft."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def load_backend():
    # Run the production metadata methods without importing the NPU runtime.
    source = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttnBackend"
    )
    names = {
        "init_cuda_graph_state",
        "_init_dsv4_graph_buffers",
        "_init_cuda_graph_metadata",
        "_apply_cuda_graph_metadata",
    }
    methods = [node for node in owner.body if getattr(node, "name", None) in names]
    assert {node.name for node in methods} == names
    wrapper = ast.parse(
        "from __future__ import annotations\nclass Backend:\n    pass\n"
    )
    wrapper.body[1].body = methods
    scope = {
        "torch": torch,
        "ForwardMetadata": SimpleNamespace,
        "_is_dflash_verify": lambda info: False,
    }
    exec(compile(ast.fix_missing_locations(wrapper), str(source), "exec"), scope)
    return scope["Backend"]


class Mode:
    def __init__(self, target=False):
        self.target = target

    def is_target_verify(self):
        return self.target

    def is_decode_or_idle(self):
        return not self.target

    def is_draft_extend_v2(self):
        return False

    def is_dllm_extend(self):
        return False


class TestGraphBlockTableCapacity(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.backend_class = load_backend()

    def make_backend(self, context, pool_width, page=16, draft=8, swa=False):
        owner = self.backend_class()
        owner.device = self.device
        owner.max_context_len = context
        owner.speculative_num_draft_tokens = draft
        owner.page_size = page
        owner.is_hybrid_swa = swa
        owner.use_sliding_window_kv_pool = False
        owner.q_head_num_padding = None
        owner.speculative_step_id = 0
        owner.speculative_step_offset_npu = torch.tensor(1, device=self.device)
        owner.req_to_token = torch.arange(
            33 * pool_width, dtype=torch.int32, device=self.device
        ).reshape(33, pool_width)
        if swa:
            owner.full_to_swa_index_mapping = torch.arange(
                33 * pool_width, dtype=torch.int32, device=self.device
            )
            owner.sliding_window_size = 32
        owner.init_cuda_graph_state(max_bs=32, max_num_tokens=256)
        return owner

    def replay(self, owner, length, bs=32, target=False):
        mode = Mode(target)
        seq_lens = torch.full((bs,), length, dtype=torch.int32, device=self.device)
        if bs not in owner.graph_metadata:
            owner._init_cuda_graph_metadata(bs, mode, torch.zeros_like(seq_lens))
        # Reorder request rows so a stale/incorrect mapping cannot pass.
        req_ids = torch.arange(bs, 0, -1, dtype=torch.int64, device=self.device)
        owner._apply_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=req_ids,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            forward_mode=mode,
            spec_info=SimpleNamespace(),
        )
        max_len = length + (owner.speculative_num_draft_tokens if target else 1)
        expected = (
            owner.req_to_token[req_ids, : max_len : owner.page_size] // owner.page_size
        )
        table = owner.forward_metadata.block_tables
        torch.testing.assert_close(table[:, : expected.shape[1]], expected)
        self.assertEqual(torch.count_nonzero(table[:, expected.shape[1] :]).item(), 0)
        torch.testing.assert_close(
            owner.forward_metadata.seq_lens, torch.full_like(seq_lens, max_len)
        )
        if owner.is_hybrid_swa:
            torch.testing.assert_close(owner.forward_metadata.block_tables_swa, table)
            indices = torch.arange(owner.graph_metadata["swa_mask"].shape[-1])
            expected_mask = (indices < max(length - 32, 0)) | (indices >= length)
            torch.testing.assert_close(
                owner.forward_metadata.swa_mask[:, 0].cpu(),
                expected_mask.expand(bs, -1),
            )

    def test_bs32_draft_crosses_8193_columns_and_keeps_storage(self):
        owner = self.make_backend(131072, 133132)
        table = owner.graph_metadata["block_tables"]
        pointer = table.data_ptr()
        # 131088 + one speculative position needs 8194 pages. Continue well
        # beyond that first overflow, then shrink to check stale-tail clearing.
        for length in (131071, 131087, 131088, 132096, 133130, 127999):
            with self.subTest(length=length):
                self.replay(owner, length)
                self.assertEqual(
                    owner.forward_metadata.block_tables.data_ptr(), pointer
                )

    def test_target_and_multiple_batch_views(self):
        owner = self.make_backend(128, 268)
        pointer = owner.graph_metadata["block_tables"].data_ptr()
        for bs, length in ((32, 128), (8, 256), (32, 136)):
            with self.subTest(bs=bs, length=length):
                self.replay(owner, length, bs=bs, target=True)
                self.assertEqual(
                    owner.forward_metadata.block_tables.data_ptr(), pointer
                )

    def test_page_aligned_speculative_reserve(self):
        owner = self.make_backend(131072, 131095)
        for length in (131071, 131088, 131093):
            with self.subTest(length=length):
                self.replay(owner, length)

    def test_npu_graph_observes_live_table(self):
        if self.device != "npu":
            self.skipTest("Run the standalone NPU launcher for graph replay")
        owner = self.make_backend(131072, 133132)
        self.replay(owner, 131071)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(2):
                owner.forward_metadata.block_tables.clone()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            observed = owner.forward_metadata.block_tables.clone()
        for length in (131088, 132096, 133130, 127999):
            with self.subTest(length=length):
                self.replay(owner, length)
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(
                    observed.cpu(), owner.forward_metadata.block_tables.cpu()
                )

    def test_pool_padding_and_non_aligned_widths(self):
        for page in (1, 16, 128):
            for draft in (None, 1, 8):
                with self.subTest(page=page, draft=draft):
                    owner = self.make_backend(63, 257, page=page, draft=draft)
                    self.assertGreaterEqual(
                        owner.graph_metadata["block_tables"].shape[1],
                        (257 + page - 1) // page,
                    )
                    self.replay(owner, 256, bs=8)

    def test_existing_capacity_is_not_reduced(self):
        for draft in (None, 1, 8):
            with self.subTest(draft=draft):
                owner = self.make_backend(256, 128, draft=draft)
                old_width = (256 + (draft or 0) + 15) // 16
                self.assertEqual(
                    owner.graph_metadata["block_tables"].shape[1], old_width
                )

    def test_swa_tables_masks_and_indices_share_capacity(self):
        owner = self.make_backend(64, 269, swa=True)
        for length in (63, 144, 267, 31):
            with self.subTest(length=length):
                self.replay(owner, length, bs=8)


if __name__ == "__main__":
    unittest.main()
