import unittest
import asyncio
import json

from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
    ALIGN,
    LAYOUT,
    ROW_BYTES,
    BridgeBudget,
    BridgeEntry,
    BridgeGeometry,
    BridgeLeases,
    BridgeMetadata,
    checked_token_pairs,
    expand_physical_pages,
    pair_bridge_entries,
    plan_native_page_transfers,
    validate_bridge_entries,
    validate_bridge_manifest,
)
from sglang.srt.disaggregation.ascend.mla_bridge_ops import pack_mla_rows
from sglang.srt.disaggregation.common.decode_kv_export import prepare_decode_kv_buffers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestAscendMLABridgeContract(CustomTestCase):
    def test_real_sender_page_slices_with_cached_prefix_and_64k_suffix(self):
        import numpy as np
        from types import SimpleNamespace
        from sglang.srt.disaggregation.common.conn import CommonKVSender
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
            bridge_segment_token_bounds,
        )

        def sender(page_count):
            return SimpleNamespace(
                curr_idx=0,
                num_kv_indices=page_count,
                kv_mgr=SimpleNamespace(
                    enable_all_cp_ranks_for_transfer=False, is_dummy_cp_rank=False
                ),
            )

        cached = sender(1)
        _, page_slice, last, skip = CommonKVSender._prepare_send_indices(
            cached, np.array([3], dtype=np.int32)
        )
        # finalize_bootstrap sets start_send_idx=63872; _send_kv_chunk sends
        # [63872, 64000), but the destination room contains only 128 suffix tokens.
        self.assertEqual(
            (page_slice.start, page_slice.stop, last, skip), (0, 1, True, False)
        )
        self.assertEqual(bridge_segment_token_bounds(page_slice, 128, 128), (0, 128))
        full = sender(512)
        boundaries = []
        for first in (0, 256):
            _, page_slice, last, skip = CommonKVSender._prepare_send_indices(
                full, np.arange(first, first + 256, dtype=np.int32)
            )
            boundaries.append((page_slice.start, page_slice.stop, last, skip))
            begin, end = bridge_segment_token_bounds(page_slice, 32768, 65536)
            self.assertEqual((begin, end), (first * 128, (first + 256) * 128))
            chunks = [min(1820, end - i) for i in range(begin, end, 1820)]
            self.assertEqual(sum(chunks), 32768)
            self.assertEqual(chunks[-1], 8)
        self.assertEqual(boundaries, [(0, 256, False, False), (256, 512, True, False)])
        with self.assertRaises(ValueError):
            bridge_segment_token_bounds(slice(0, 2), 128, 128)
        with self.assertRaises(ValueError):
            bridge_segment_token_bounds(slice(0, 1), 127, 256)

    def test_cpu_mocked_manager_chain_two_layers_and_partial_tail(self):
        import ctypes
        import sys
        from threading import Lock
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import torch
        from sglang.srt.disaggregation.ascend.conn import (
            AscendKVManager,
            AscendKVReceiver,
        )
        from sglang.srt.disaggregation.ascend.mla_bridge_collect import (
            collect_native_prefill_planes,
        )
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
            BridgeRoomDescriptor,
        )
        from sglang.srt.disaggregation.ascend.mla_bridge_resources import (
            SparseDecodeBridgeResources,
        )
        from sglang.srt.disaggregation.ascend.mla_bridge_workspace import (
            PackedBridgeWorkspace,
        )
        from sglang.srt.disaggregation.base.conn import KVPoll

        torch.manual_seed(91)

        def native_pool(layers, draft):
            return SimpleNamespace(
                enable_sparsity_driven_kv_offload=False,
                is_draft_worker=draft,
                dsa_kv_cache_store_fp8=False,
                page_size=128,
                index_page_size=128,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                model_layer_ids=list(range(layers)),
                indexer_layer_ids=list(range(layers)),
                k_buffer=torch.randn((layers, 15, 128, 1, 512), dtype=torch.bfloat16),
                v_buffer=torch.randn((layers, 15, 128, 1, 64), dtype=torch.bfloat16),
                index_k_buffer=torch.randn(
                    (layers, 15, 128, 1, 8), dtype=torch.bfloat16
                ),
                index_k_scale_buffer=None,
            )

        main, draft, dst_draft = (
            native_pool(2, False),
            native_pool(1, True),
            native_pool(1, True),
        )
        for tensor in (
            dst_draft.k_buffer,
            dst_draft.v_buffer,
            dst_draft.index_k_buffer,
        ):
            tensor.zero_()
        manager_class = type(
            "SparseKVCacheManager",
            (),
            {
                "__module__": "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager"
            },
        )
        owner = manager_class()
        owner.host_kv_buffer = [
            torch.zeros((15, 128, 1, 576), dtype=torch.bfloat16) for _ in range(2)
        ]
        owner.dev_ptr_list = [111, 222]
        dpool = SimpleNamespace(
            enable_sparsity_driven_kv_offload=True,
            is_draft_worker=False,
            k_buffer=None,
            v_buffer=None,
            dsa_kv_cache_store_fp8=False,
            page_size=128,
            index_page_size=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            store_dtype=torch.bfloat16,
            sparse_kv_manager=owner,
            model_layer_ids=[0, 1],
            indexer_layer_ids=[0, 1],
            index_k_buffer=torch.zeros_like(main.index_k_buffer),
            index_k_scale_buffer=None,
        )
        geometry = BridgeGeometry(LAYOUT, 16, 1, 1, 128, "bfloat16", 512, 64)
        resources = SparseDecodeBridgeResources(dpool, geometry, device="cpu")
        draft_planes = collect_native_prefill_planes(None, dst_draft)
        ptrs = list(resources.entries.native_ptrs) + [
            p.tensor.data_ptr() for p in draft_planes
        ]
        sizes = list(resources.entries.native_sizes) + [
            p.tensor.nbytes for p in draft_planes
        ]
        strides = list(resources.entries.native_strides) + [
            p.entry.item_stride for p in draft_planes
        ]
        args = SimpleNamespace(
            kv_data_ptrs=ptrs, kv_data_lens=sizes, kv_item_lens=strides
        )
        resources.bind_native_manifest(dst_draft, args)

        class Engine:
            def batch_register_memory(self, ptrs, lengths):
                return 0

            def unregister_memory(self, ptr):
                return 0

            def batch_transfer_sync_write(self, session, srcs, dsts, lengths):
                for src, dst, length in zip(srcs, dsts, lengths):
                    ctypes.memmove(dst, src, length)
                return 0

        engine = Engine()
        resources.register(engine)
        metadata = resources.candidate_metadata()
        coverage = resources.register_room(
            room=4,
            session="s",
            pages=list(range(15)),
            num_tokens=1833,
            prefix_tokens=0,
            draft_pool=dst_draft,
            peers=[("127.0.0.1", 5678)],
        )
        dmanager = object.__new__(AscendKVManager)
        dmanager.kv_args = SimpleNamespace(gpu_id=0)
        dmanager._sparse_decode_bridge_resources = resources
        dmanager.npu_mla_bridge_metadata = metadata
        original_submit = resources.ring.submit_scatter

        def cpu_scatter(packed, host, src, dst, valid, **kwargs):
            host[dst[valid]] = packed[src[valid]]

        def submit(*a, **kw):
            kw["scatter"] = cpu_scatter
            return original_submit(*a, **kw)

        resources.ring.submit_scatter = submit

        class Event:
            def record(self):
                pass

            def synchronize(self):
                pass

            def query(self):
                return True

        fake_torch = SimpleNamespace(
            npu=SimpleNamespace(set_device=lambda x: None, Event=Event),
            device=lambda value: "cpu",
            int64=torch.int64,
            bool=torch.bool,
            bfloat16=torch.bfloat16,
            tensor=torch.tensor,
            arange=torch.arange,
            ones=torch.ones,
            index_select=torch.index_select,
        )
        planes = collect_native_prefill_planes(main, draft)
        sender = object.__new__(AscendKVManager)
        sender.kv_args = SimpleNamespace(
            _npu_mla_prefill_planes=planes,
            kv_data_ptrs=[p.tensor.data_ptr() for p in planes],
            gpu_id=0,
        )
        sender.engine = SimpleNamespace(engine=engine)
        sender._npu_mla_sender_workspace = PackedBridgeWorkspace("cpu")
        sender._npu_mla_sender_workspace.transfer.register(engine)
        sender._npu_mla_bridge_ready = True
        sender._npu_mla_rpc_lock = Lock()
        sender._npu_mla_rpc_pending = {}
        sender._npu_mla_ack_pending = {}
        sender._npu_mla_unsettled_rooms = set()
        sender._npu_mla_active_rooms = set()
        sender._npu_mla_shutdown = False
        sender.local_ip, sender.rank_port = "127.0.0.1", 5678
        sender._poison_unsettled_send = Mock()

        def native_copy(session, blocks):
            for src, dst, length in blocks:
                ctypes.memmove(dst, src, length)
            return 0

        sender._transfer_data = native_copy

        def to_decode(endpoint, parts, **kwargs):
            if parts[0] == b"NPU_MLA_LEASE_REQ":
                dmanager.handle_npu_mla_bridge_lease_request(parts)
            elif parts[0] == b"NPU_MLA_CHUNK_READY":
                dmanager.handle_npu_mla_bridge_chunk(parts)
            elif parts[0] == b"NPU_MLA_NATIVE_READY":
                dmanager.handle_npu_mla_bridge_native_ready(parts)
            else:
                raise AssertionError(parts[0])

        def to_prefill(endpoint, parts, **kwargs):
            if parts[0] == b"NPU_MLA_LEASE_RSP":
                sender.handle_npu_mla_bridge_lease_response(parts)
            elif parts[0] == b"NPU_MLA_ACK":
                sender.handle_npu_mla_bridge_ack(parts)
            else:
                raise AssertionError(parts[0])

        sender._send_multipart_locked = to_decode
        dmanager._send_multipart_locked = to_prefill
        registration = SimpleNamespace(
            npu_mla_bridge_metadata=metadata, dst_kv_ptrs=ptrs, dst_kv_item_lens=strides
        )
        req = SimpleNamespace(
            room=4,
            mooncake_session_id="s",
            decode_prefix_len=0,
            npu_mla_room=BridgeRoomDescriptor(coverage.generation, 1833, 0).encode(),
            dst_device_kv_indices=None,
            endpoint="127.0.0.1",
            dst_port=1234,
        )
        chunk = SimpleNamespace(
            prefill_kv_indices=list(reversed(range(15))),
            index_slice=slice(0, 15),
            num_kv_tokens=1833,
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(
                sender._do_npu_mla_bridge_transfer(
                    chunk, req, registration, list(range(15)), None, None, 0
                ),
                (0, False),
            )
        source_rows = torch.tensor(
            [
                page * 128 + offset
                for page in reversed(range(15))
                for offset in range(128)
            ][:1833],
            dtype=torch.long,
        )
        for layer in range(2):
            expected = torch.cat(
                (
                    main.k_buffer[layer].view(-1, 512)[source_rows],
                    main.v_buffer[layer].view(-1, 64)[source_rows],
                ),
                dim=1,
            )
            self.assertTrue(
                torch.equal(
                    owner.host_kv_buffer[layer].view(-1, 576)[:1833].view(torch.uint16),
                    expected.view(torch.uint16),
                )
            )
            self.assertTrue(
                torch.equal(
                    dpool.index_k_buffer[layer, 0], main.index_k_buffer[layer, 14]
                )
            )
            self.assertTrue(
                torch.equal(
                    dpool.index_k_buffer[layer, 14], main.index_k_buffer[layer, 0]
                )
            )
        self.assertTrue(torch.equal(dst_draft.k_buffer[0, 0], draft.k_buffer[0, 14]))
        self.assertTrue(torch.equal(dst_draft.v_buffer[0, 14], draft.v_buffer[0, 0]))
        self.assertTrue(
            torch.equal(dst_draft.index_k_buffer[0, 0], draft.index_k_buffer[0, 14])
        )
        receiver = object.__new__(AscendKVReceiver)
        receiver.kv_mgr = dmanager
        receiver._npu_mla_bridge_coverage = coverage
        receiver._npu_mla_bridge_request = object()
        self.assertEqual(
            receiver.filter_transfer_status(KVPoll.Success), KVPoll.Success
        )
        receiver.finish_npu_mla_transfer()
        self.assertFalse(resources._rooms)
        sender._npu_mla_sender_workspace.close(
            engine, transport_drained=True, synchronize=lambda: None
        )
        resources.close(engine, transport_drained=True, synchronize=lambda: None)

    def test_native_page_ranges_and_terminal_components_gate_success(self):
        from sglang.srt.disaggregation.ascend.mla_bridge_coverage import (
            BridgeRoomCoverage,
        )

        coverage = BridgeRoomCoverage(
            room=4,
            generation=2,
            session="s",
            pages=[1, 2],
            num_tokens=130,
            prefix_tokens=0,
            physical_capacity=512,
            layers=[3],
            required_native=[
                ("main", 3, "index"),
                ("draft", 0, "ckv"),
                ("aux",),
                ("state",),
            ],
        )
        rows = coverage.reserve(
            room=4, generation=2, session="s", layer=3, start=0, count=130, lease_id=1
        )
        self.assertEqual(len(rows), 130)
        coverage.complete_scatter(
            room=4, generation=2, session="s", layer=3, lease_id=1
        )
        coverage.complete_native_range(
            room=4, generation=2, session="s", start=0, count=128
        )
        coverage.complete_terminal(room=4, generation=2, session="s")
        self.assertFalse(coverage.success())
        with self.assertRaises(ValueError):
            coverage.complete_native_range(
                room=4, generation=2, session="s", start=0, count=2
            )
        coverage.complete_native_range(
            room=4, generation=2, session="s", start=128, count=2
        )
        self.assertTrue(coverage.success())

    def test_bridge_receiver_holds_success_and_failure_allocations(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from sglang.srt.disaggregation.ascend.conn import AscendKVReceiver
        from sglang.srt.disaggregation.base.conn import KVPoll

        receiver = object.__new__(AscendKVReceiver)
        resources = Mock()
        receiver.kv_mgr = SimpleNamespace(
            npu_mla_bridge_metadata=None, _sparse_decode_bridge_resources=resources
        )
        self.assertFalse(receiver.requires_strict_transfer_drain())
        self.assertEqual(
            receiver.filter_transfer_status(KVPoll.Success), KVPoll.Success
        )
        receiver.kv_mgr.npu_mla_bridge_metadata = b"negotiated"
        receiver._npu_mla_bridge_coverage = Mock(room=4, generation=2, session="s")
        receiver._npu_mla_bridge_coverage.success.return_value = False
        self.assertTrue(receiver.requires_strict_transfer_drain())
        self.assertEqual(
            receiver.filter_transfer_status(KVPoll.Success), KVPoll.Transferring
        )
        self.assertEqual(receiver.filter_transfer_status(KVPoll.Failed), KVPoll.Failed)
        receiver.begin_strict_transfer_drain()
        resources.quarantine_room.assert_called_once_with(4, 2, "s")
        self.assertFalse(receiver.is_strict_transfer_drain_safe())

    def test_control_rpc_event_ack_roundtrip_without_device(self):
        import sys
        from threading import Lock
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from sglang.srt.disaggregation.ascend.conn import AscendKVManager

        sender = object.__new__(AscendKVManager)
        sender._npu_mla_rpc_lock = Lock()
        sender._npu_mla_rpc_pending = {}
        sender._npu_mla_ack_pending = {}
        sender._npu_mla_unsettled_rooms = set()
        sender.local_ip, sender.rank_port = "127.0.0.1", 5678
        receiver = object.__new__(AscendKVManager)
        receiver.npu_mla_bridge_metadata = b"negotiated"
        receiver.kv_args = SimpleNamespace(gpu_id=0)
        resources = Mock()
        resources.reserve_chunk.return_value = (1, 9, (384, 385))
        resources.reserved_rows.return_value = (384, 385)
        resources.host_for_layer.return_value = (object(), 123)
        resources.finish_chunk.side_effect = [False, True]
        receiver._sparse_decode_bridge_resources = resources

        def from_sender(endpoint, parts, **kwargs):
            if parts[0] == b"NPU_MLA_LEASE_REQ":
                receiver.handle_npu_mla_bridge_lease_request(parts)
            else:
                receiver.handle_npu_mla_bridge_chunk(parts)

        def from_receiver(endpoint, parts, **kwargs):
            if parts[0] == b"NPU_MLA_LEASE_RSP":
                sender.handle_npu_mla_bridge_lease_response(parts)
            else:
                sender.handle_npu_mla_bridge_ack(parts)

        sender._send_multipart_locked = from_sender
        receiver._send_multipart_locked = from_receiver
        fake_torch = SimpleNamespace(
            npu=SimpleNamespace(set_device=Mock(), Event=Mock()),
            device=lambda value: value,
            int64="i64",
            bool="bool",
            arange=Mock(return_value=object()),
            tensor=Mock(return_value=object()),
            ones=Mock(return_value=object()),
        )
        req = SimpleNamespace(
            room=4, mooncake_session_id="session", endpoint="127.0.0.1", dst_port=1234
        )
        descriptor = SimpleNamespace(generation=2)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            slot, lease = sender.request_npu_mla_lease(
                req, descriptor, 3, 0, 2, timeout=1
            )
            sender.notify_npu_mla_chunk_and_wait(
                req, descriptor, 3, 0, 2, slot, lease, timeout=1
            )
        self.assertEqual(resources.finish_chunk.call_count, 2)
        resources.ring.write_drained.assert_called_once_with(1, 4, 2, "session", 9)
        resources.ring.submit_scatter.assert_called_once()
        resources.quarantine_room.assert_not_called()
        self.assertFalse(sender._npu_mla_unsettled_rooms)

    def test_prefill_lease_rpc_matches_identity_and_quarantines_timeout(self):
        from threading import Lock
        from types import SimpleNamespace
        from sglang.srt.disaggregation.ascend.conn import AscendKVManager

        manager = object.__new__(AscendKVManager)
        manager._npu_mla_rpc_lock = Lock()
        manager._npu_mla_rpc_pending = {}
        manager._npu_mla_unsettled_rooms = set()
        manager.local_ip, manager.rank_port = "127.0.0.1", 5678
        req = SimpleNamespace(
            room=4, mooncake_session_id="session", endpoint="127.0.0.1", dst_port=1234
        )
        descriptor = SimpleNamespace(generation=2)

        def reply(endpoint, parts, **kwargs):
            manager.handle_npu_mla_bridge_lease_response(
                [b"NPU_MLA_LEASE_RSP", *parts[1:7], b"OK", b"1", b"9"]
            )

        manager._send_multipart_locked = reply
        self.assertEqual(
            manager.request_npu_mla_lease(req, descriptor, 3, 0, 2, timeout=1), (1, 9)
        )
        self.assertFalse(manager._npu_mla_rpc_pending)
        manager._send_multipart_locked = lambda *args, **kwargs: None
        with self.assertRaises(TimeoutError):
            manager.request_npu_mla_lease(req, descriptor, 3, 0, 2, timeout=0.001)
        self.assertIn(4, manager._npu_mla_unsettled_rooms)
        with self.assertRaises(RuntimeError):
            manager.request_npu_mla_lease(req, descriptor, 3, 0, 2, timeout=1)

    def test_decode_lease_rpc_requires_room_peer_and_quarantines_send_failure(self):
        from unittest.mock import Mock
        from sglang.srt.disaggregation.ascend.conn import AscendKVManager

        manager = object.__new__(AscendKVManager)
        resources = Mock()
        resources.reserve_chunk.return_value = (1, 9, (384, 385))
        manager._sparse_decode_bridge_resources = resources
        manager.npu_mla_bridge_metadata = b"negotiated"
        manager._send_multipart_locked = Mock()
        msg = [
            b"NPU_MLA_LEASE_REQ",
            b"4",
            b"2",
            b"session",
            b"3",
            b"0",
            b"2",
            b"127.0.0.1",
            b"1234",
        ]
        manager.handle_npu_mla_bridge_lease_request(msg)
        resources.require_peer.assert_called_once_with(
            4, 2, "session", ("127.0.0.1", 1234)
        )
        self.assertEqual(
            manager._send_multipart_locked.call_args.args[1],
            [b"NPU_MLA_LEASE_RSP", *msg[1:7], b"OK", b"1", b"9"],
        )
        manager._send_multipart_locked.side_effect = RuntimeError("network failed")
        manager.handle_npu_mla_bridge_lease_request(msg)
        resources.quarantine_room.assert_called_once_with(4, 2, "session")
        resources.reset_mock()
        resources.require_peer.side_effect = ValueError("stale")
        manager.handle_npu_mla_bridge_lease_request(msg)
        resources.reserve_chunk.assert_not_called()
        resources.quarantine_room.assert_not_called()

    def test_page_major_source_flatten_and_partial_tail_bytes(self):
        import torch
        from sglang.srt.disaggregation.ascend.mla_bridge_collect import (
            NativePlane,
            flatten_main_mla_plane,
        )

        torch.manual_seed(1234)
        k = (
            torch.randint(0, 65536, (2, 128, 1, 512), dtype=torch.int32)
            .to(torch.uint16)
            .view(torch.bfloat16)
        )
        v = (
            torch.randint(0, 65536, (2, 128, 1, 64), dtype=torch.int32)
            .to(torch.uint16)
            .view(torch.bfloat16)
        )
        kp = NativePlane(
            BridgeEntry("native", "main", 0, "ckv", "bfloat16", 131072, 2, "page", 128),
            k,
        )
        vp = NativePlane(
            BridgeEntry("native", "main", 0, "kpe", "bfloat16", 16384, 2, "page", 128),
            v,
        )
        krows, vrows = flatten_main_mla_plane(kp), flatten_main_mla_plane(vp)
        self.assertEqual(krows.data_ptr(), k.data_ptr())
        indices = torch.tensor([127, 128, 130], dtype=torch.int64)
        from types import SimpleNamespace
        from unittest.mock import Mock
        from sglang.srt.disaggregation.ascend.mla_bridge_workspace import (
            PackedBridgeWorkspace,
        )

        fake = SimpleNamespace(pack=Mock(return_value="packed"))
        self.assertEqual(
            PackedBridgeWorkspace.pack_native_planes(fake, 7, kp, vp, indices), "packed"
        )
        packed_call = fake.pack.call_args.args
        self.assertEqual(packed_call[1].data_ptr(), k.data_ptr())
        self.assertEqual(packed_call[2].data_ptr(), v.data_ptr())
        packed = torch.empty((3, 576), dtype=torch.bfloat16)
        pack_mla_rows(
            krows,
            vrows,
            indices,
            packed,
            torch.empty((3, 512), dtype=torch.bfloat16),
            torch.empty((3, 64), dtype=torch.bfloat16),
        )
        self.assertTrue(
            torch.equal(
                packed[:, :512].contiguous().view(torch.uint16),
                krows[indices].view(torch.uint16),
            )
        )
        self.assertTrue(
            torch.equal(
                packed[:, 512:].contiguous().view(torch.uint16),
                vrows[indices].view(torch.uint16),
            )
        )
        with self.assertRaises(ValueError):
            flatten_main_mla_plane(NativePlane(kp.entry, k.transpose(0, 1)))

    def test_resource_ack_waits_for_event_and_owned_page_coverage(self):
        from threading import Lock
        from sglang.srt.disaggregation.ascend.mla_bridge_coverage import (
            BridgeRoomCoverage,
        )
        from sglang.srt.disaggregation.ascend.mla_bridge_resources import (
            SparseDecodeBridgeResources,
        )

        class Ring:
            def __init__(self):
                self.done = False
                self.aborted = False

            def acquire(self, *args):
                return (0, 1)

            def finish_scatter(self, *args):
                return self.done

            def abort_room(self, *args):
                self.aborted = True

        resources = object.__new__(SparseDecodeBridgeResources)
        resources._lock = Lock()
        resources._lease_rows = {}
        resources.ring = Ring()
        coverage = BridgeRoomCoverage(
            room=4,
            generation=1,
            session="session",
            pages=[3],
            num_tokens=2,
            prefix_tokens=0,
            physical_capacity=1024,
            layers=[8],
            required_native=[("aux",)],
        )
        resources._rooms = {4: coverage}
        slot, lease, rows = resources.reserve_chunk(
            room=4, generation=1, session="session", layer=8, start=0, count=2
        )
        self.assertEqual((slot, lease, rows), (0, 1, (384, 385)))
        self.assertEqual(
            resources.reserved_rows(
                room=4,
                generation=1,
                session="session",
                layer=8,
                start=0,
                count=2,
                slot=slot,
                lease_id=lease,
            ),
            rows,
        )
        args = dict(
            room=4, generation=1, session="session", layer=8, slot=slot, lease_id=lease
        )
        self.assertFalse(resources.finish_chunk(**args))
        self.assertFalse(coverage.success())
        resources.ring.done = True
        self.assertTrue(resources.finish_chunk(**args))
        coverage.complete_native(
            room=4, generation=1, session="session", component=("aux",)
        )
        self.assertTrue(coverage.success())
        with self.assertRaises(ValueError):
            resources.reserve_chunk(
                room=4, generation=1, session="session", layer=8, start=0, count=2
            )
        self.assertTrue(resources.ring.aborted)
        self.assertFalse(coverage.success())

    def test_room_descriptor_binds_generation_and_preserves_legacy_wire(self):
        import numpy as np
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
            BridgeRoomDescriptor,
        )
        from sglang.srt.disaggregation.mooncake.conn import TransferInfo

        from types import SimpleNamespace
        from sglang.srt.disaggregation.ascend.conn import AscendKVReceiver

        descriptor = BridgeRoomDescriptor(7, 141, 128)
        receiver = object.__new__(AscendKVReceiver)
        receiver.kv_mgr = SimpleNamespace(npu_mla_bridge_metadata=b"enabled")
        with self.assertRaises(RuntimeError):
            receiver._npu_mla_room_frame()
        receiver._npu_mla_bridge_coverage = SimpleNamespace(
            generation=7, num_tokens=141, prefix_tokens=128
        )
        encoded = receiver._npu_mla_room_frame()
        self.assertEqual(encoded, descriptor.encode())
        receiver.kv_mgr.npu_mla_bridge_metadata = None
        self.assertIsNone(receiver._npu_mla_room_frame())
        self.assertEqual(BridgeRoomDescriptor.decode(encoded), descriptor)
        for invalid in (
            b"",
            b"{}",
            b"x" * 257,
            b'{"version":true,"generation":1,"num_tokens":1,"prefix_tokens":0}',
        ):
            with self.assertRaises(ValueError):
                BridgeRoomDescriptor.decode(invalid)
        frames = [
            b"3",
            b"127.0.0.1",
            b"1234",
            b"session",
            np.array([1, 3], dtype=np.int32).tobytes(),
            b"0",
            b"",
            b"1",
            b"128",
            b"",
        ]
        legacy = TransferInfo.from_zmq(frames)
        self.assertEqual(legacy.npu_mla_room, b"")
        modern = TransferInfo.from_zmq(frames + [encoded])
        self.assertEqual(BridgeRoomDescriptor.decode(modern.npu_mla_room), descriptor)
        np.testing.assert_array_equal(modern.dst_kv_indices, legacy.dst_kv_indices)

    def test_room_coverage_derives_rows_and_gates_all_components(self):
        from sglang.srt.disaggregation.ascend.mla_bridge_coverage import (
            BridgeRoomCoverage,
        )

        room = BridgeRoomCoverage(
            room=5,
            generation=2,
            session="peer",
            pages=[4, 1],
            num_tokens=130,
            prefix_tokens=128,
            physical_capacity=640,
            layers=[0, 1],
            required_native={"index", "draft", "aux"},
        )
        rows = room.reserve(
            room=5,
            generation=2,
            session="peer",
            layer=0,
            start=0,
            count=130,
            lease_id=10,
        )
        self.assertEqual(rows, tuple(range(512, 640)) + (128, 129))
        with self.assertRaises(ValueError):
            room.reserve(
                room=5,
                generation=1,
                session="peer",
                layer=1,
                start=0,
                count=130,
                lease_id=11,
            )
        with self.assertRaises(ValueError):
            room.complete_scatter(
                room=5, generation=2, session="peer", layer=0, lease_id=11
            )
        self.assertFalse(room.success())
        room.complete_scatter(
            room=5, generation=2, session="peer", layer=0, lease_id=10
        )
        room.reserve(
            room=5,
            generation=2,
            session="peer",
            layer=1,
            start=0,
            count=130,
            lease_id=11,
        )
        room.complete_scatter(
            room=5, generation=2, session="peer", layer=1, lease_id=11
        )
        for component in ("index", "draft"):
            room.complete_native(
                room=5, generation=2, session="peer", component=component
            )
        self.assertFalse(room.success())
        room.complete_native(room=5, generation=2, session="peer", component="aux")
        self.assertTrue(room.success())
        with self.assertRaises(ValueError):
            room.complete_native(room=5, generation=2, session="peer", component="aux")

    def test_prefill_collection_preserves_target_index_and_draft_order(self):
        import torch
        from types import SimpleNamespace

        from sglang.srt.disaggregation.ascend.mla_bridge_collect import (
            collect_native_prefill_planes,
        )

        def pool(draft):
            return SimpleNamespace(
                enable_sparsity_driven_kv_offload=False,
                is_draft_worker=draft,
                dsa_kv_cache_store_fp8=False,
                page_size=128,
                index_page_size=128,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                model_layer_ids=[0],
                indexer_layer_ids=[0],
                k_buffer=torch.empty((1, 2, 128, 1, 512), dtype=torch.bfloat16),
                v_buffer=torch.empty((1, 2, 128, 1, 64), dtype=torch.bfloat16),
                index_k_buffer=torch.empty((1, 2, 128, 1, 128), dtype=torch.uint8),
                index_k_scale_buffer=None,
            )

        main, draft = pool(False), pool(True)
        planes = collect_native_prefill_planes(main, draft)
        self.assertEqual(
            [(x.entry.model, x.entry.component) for x in planes],
            [
                ("main", "ckv"),
                ("main", "kpe"),
                ("main", "index"),
                ("draft", "ckv"),
                ("draft", "kpe"),
                ("draft", "index"),
            ],
        )
        self.assertEqual(
            [x.tensor.data_ptr() for x in planes],
            [
                main.k_buffer[0].data_ptr(),
                main.v_buffer[0].data_ptr(),
                main.index_k_buffer[0].data_ptr(),
                draft.k_buffer[0].data_ptr(),
                draft.v_buffer[0].data_ptr(),
                draft.index_k_buffer[0].data_ptr(),
            ],
        )
        from sglang.srt.disaggregation.ascend.mla_bridge_collect import (
            plan_native_component_blocks,
        )

        staged = BridgeEntry(
            "staged", "main", 0, "combined", "bfloat16", 1152, 256, "token", 128
        )
        native = [
            p.entry
            for p in planes
            if p.entry.component not in ("ckv", "kpe") or p.entry.model == "draft"
        ]
        from sglang.srt.disaggregation.ascend.conn import AscendKVManager

        sender = object.__new__(AscendKVManager)
        sender.kv_args = SimpleNamespace(
            _npu_mla_prefill_planes=planes,
            kv_data_ptrs=[p.tensor.data_ptr() for p in planes],
        )
        metadata = BridgeMetadata(
            BridgeGeometry(LAYOUT, 16, 1, 1, 128, "bfloat16", 512, 64),
            tuple([staged, *native]),
            ALIGN,
            2 * ALIGN,
        ).encode()
        self.assertIn(
            "not ready", sender.validate_npu_mla_bridge_registration(metadata)
        )
        sender._npu_mla_bridge_ready = True
        registration = SimpleNamespace(
            dst_kv_ptrs=[0x10000000 + i * 0x100000 for i in range(len(native))],
            dst_kv_item_lens=[entry.item_stride for entry in native],
        )
        self.assertIsNone(
            sender.validate_npu_mla_bridge_registration(metadata, registration)
        )
        registration.dst_kv_item_lens[0] += 1
        self.assertIn(
            "manifest mismatch",
            sender.validate_npu_mla_bridge_registration(metadata, registration),
        )
        registration.dst_kv_item_lens[0] -= 1
        sender.kv_args.kv_data_ptrs = [123]
        self.assertIn(
            "pointers changed",
            sender.validate_npu_mla_bridge_registration(metadata, registration),
        )
        sender.kv_args.kv_data_ptrs = [p.tensor.data_ptr() for p in planes]
        from threading import Lock
        from unittest.mock import Mock, patch
        import sys
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
            BridgeRoomDescriptor,
        )

        workspace = Mock()
        workspace.capacity = 1820
        workspace.transfer.size = ALIGN
        workspace.try_acquire.return_value = 1
        sender._npu_mla_sender_workspace = workspace
        sender._npu_mla_rpc_lock = Lock()
        sender._npu_mla_unsettled_rooms = set()
        sender._npu_mla_active_rooms = set()
        sender._npu_mla_shutdown = False
        sender.kv_args.gpu_id = 0
        sender.engine = SimpleNamespace(engine=object())
        sender.request_npu_mla_lease = Mock(return_value=(0, 9))
        sender.notify_npu_mla_chunk_and_wait = Mock()
        sender._transfer_data = Mock(return_value=0)
        sender._poison_unsettled_send = Mock()
        sender._send_multipart_locked = Mock()
        sender.local_ip, sender.rank_port = "127.0.0.1", 5678
        registration.npu_mla_bridge_metadata = metadata
        request = SimpleNamespace(
            room=4,
            mooncake_session_id="s",
            endpoint="127.0.0.1",
            dst_port=1234,
            decode_prefix_len=0,
            npu_mla_room=BridgeRoomDescriptor(2, 2, 0).encode(),
            dst_device_kv_indices=None,
        )
        chunk = SimpleNamespace(
            prefill_kv_indices=[1], index_slice=slice(0, 1), num_kv_tokens=2
        )
        fake_torch = SimpleNamespace(
            npu=SimpleNamespace(set_device=Mock(), Event=Mock()),
            device=lambda value: value,
            int64="i64",
            tensor=Mock(return_value=object()),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(
                sender._do_npu_mla_bridge_transfer(
                    chunk, request, registration, [0], None, None, 0
                ),
                (0, False),
            )
        workspace.pack_native_planes.assert_called_once()
        workspace.send.assert_called_once()
        sender.notify_npu_mla_chunk_and_wait.assert_called_once()
        workspace.release.assert_called_once()
        self.assertEqual(sender._transfer_data.call_count, len(native))
        self.assertEqual(
            sender._send_multipart_locked.call_args.args[1][0], b"NPU_MLA_NATIVE_READY"
        )
        sender._send_multipart_locked.reset_mock()
        sender._transfer_data.return_value = -1
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(
                sender._do_npu_mla_bridge_transfer(
                    chunk, request, registration, [0], None, None, 0
                ),
                (-1, False),
            )
        self.assertIn(4, sender._npu_mla_unsettled_rooms)
        sender._poison_unsettled_send.assert_called_once_with(4)
        sender._send_multipart_locked.assert_not_called()
        remote_ptrs = [0x10000000 + i * 0x100000 for i in range(len(native))]
        plans = plan_native_component_blocks(
            planes, [staged, *native], remote_ptrs, [1], [0]
        )
        self.assertEqual(
            [key for key, _ in plans], [(e.model, e.layer, e.component) for e in native]
        )
        for (key, blocks), entry, dst in zip(plans, native, remote_ptrs):
            src = next(
                p
                for p in planes
                if (p.entry.model, p.entry.layer, p.entry.component) == key
            )
            self.assertEqual(
                blocks,
                ((src.tensor.data_ptr() + entry.item_stride, dst, entry.item_stride),),
            )
        with self.assertRaises(ValueError):
            plan_native_component_blocks(
                planes, [staged, *native], remote_ptrs, [2], [0]
            )
        with self.assertRaises(ValueError):
            plan_native_component_blocks(
                planes, [staged, *native], remote_ptrs[:-1], [1], [0]
            )
        main.index_page_size = 64
        with self.assertRaises(ValueError):
            collect_native_prefill_planes(main, draft)
        main.index_page_size = 128
        main.index_k_buffer = None
        with self.assertRaises(ValueError):
            collect_native_prefill_planes(main, draft)

    def test_ascend_factory_binds_sparse_resources_without_native_kv(self):
        import sys
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.disaggregation.ascend.conn import AscendKVManager
        from sglang.srt.environ import envs

        fake_module = ModuleType("sglang.srt.hardware_backend.npu.memory_pool_npu")

        class Pool:
            enable_sparsity_driven_kv_offload = True
            page_size = 128
            store_dtype = "torch.bfloat16"
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            index_k_buffer = SimpleNamespace(device="cpu")

        fake_module.NPUMLATokenToKVPool = Pool
        args = SimpleNamespace()
        entries = SimpleNamespace(
            native_ptrs=(7,), native_sizes=(8,), native_strides=(4,)
        )
        owner = SimpleNamespace(entries=entries)
        parallel = SimpleNamespace(attn_tp_size=16, pp_size=1, attn_dcp_size=1)
        with (
            envs.SGLANG_NPU_ENABLE_PD_MLA_BRIDGE.override(True),
            patch.dict(sys.modules, {fake_module.__name__: fake_module}),
            patch(
                "sglang.srt.disaggregation.ascend.conn.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.disaggregation.ascend.mla_bridge_resources.SparseDecodeBridgeResources",
                return_value=owner,
            ),
        ):
            self.assertEqual(
                AscendKVManager.prepare_decode_kv_buffers(args, Pool()), ([7], [8], [4])
            )
        self.assertIs(args._npu_mla_bridge_resources, owner)

        Pool.enable_sparsity_driven_kv_offload = False
        prefill_args = SimpleNamespace()
        with (
            patch.dict(sys.modules, {fake_module.__name__: fake_module}),
            patch(
                "sglang.srt.disaggregation.ascend.conn.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.disaggregation.ascend.mla_bridge_collect.collect_native_prefill_planes",
                return_value=("plane",),
            ),
        ):
            AscendKVManager.prepare_prefill_mla_bridge_inputs(
                prefill_args, Pool(), None
            )
        self.assertEqual(prefill_args._npu_mla_prefill_planes, ("plane",))

    def test_sparse_collection_keeps_native_index_and_combined_host_separate(self):
        import torch
        from types import SimpleNamespace

        from sglang.srt.disaggregation.ascend.mla_bridge_collect import (
            collect_sparse_decode_entries,
        )

        manager_class = type(
            "SparseKVCacheManager",
            (),
            {
                "__module__": "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager"
            },
        )
        manager = manager_class()
        manager.host_kv_buffer = [torch.empty((1, 4, 1, 576), dtype=torch.bfloat16)]
        manager.dev_ptr_list = [123]
        index = torch.empty((1, 3, 128), dtype=torch.uint8)
        scale = torch.empty((1, 3, 8), dtype=torch.float32)
        pool = SimpleNamespace(
            enable_sparsity_driven_kv_offload=True,
            is_draft_worker=False,
            k_buffer=None,
            v_buffer=None,
            dsa_kv_cache_store_fp8=False,
            page_size=128,
            index_page_size=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            store_dtype=torch.bfloat16,
            sparse_kv_manager=manager,
            model_layer_ids=[3],
            indexer_layer_ids=[3],
            index_k_buffer=index,
            index_k_scale_buffer=scale,
        )
        geometry = BridgeGeometry(LAYOUT, 16, 1, 1, 128, "bfloat16", 512, 64)
        pool.index_page_size = 64
        with self.assertRaises(ValueError):
            collect_sparse_decode_entries(pool, geometry)
        pool.index_page_size = 128
        result = collect_sparse_decode_entries(pool, geometry)
        self.assertEqual(
            [x.component for x in result.entries], ["combined", "index", "index_scale"]
        )
        self.assertEqual(result.native_ptrs, (index[0].data_ptr(), scale[0].data_ptr()))
        self.assertNotIn(123, result.native_ptrs)
        self.assertEqual(result.host_device_ptrs, (123,))
        from unittest.mock import Mock

        from sglang.srt.disaggregation.ascend.conn import AscendKVManager
        from sglang.srt.disaggregation.ascend.mla_bridge_resources import (
            SparseDecodeBridgeResources,
        )
        from sglang.srt.disaggregation.utils import DisaggregationMode

        resources = SparseDecodeBridgeResources(pool, geometry, device="cpu")
        host_rows, mapped_ptr = resources.host_for_layer(3)
        self.assertEqual(host_rows.shape, (4, 576))
        self.assertEqual(host_rows.data_ptr(), manager.host_kv_buffer[0].data_ptr())
        self.assertEqual(mapped_ptr, 123)
        with self.assertRaises(ValueError):
            resources.host_for_layer(4)
        args = SimpleNamespace(
            kv_data_ptrs=list(result.native_ptrs),
            kv_data_lens=list(result.native_sizes),
            kv_item_lens=list(result.native_strides),
        )
        resources.bind_native_manifest(None, args)
        with self.assertRaises(RuntimeError):
            resources.candidate_metadata()
        ready_resources = SparseDecodeBridgeResources(pool, geometry, device="cpu")
        ready_resources.bind_native_manifest(None, args)
        good_engine = Mock()
        good_engine.batch_register_memory.return_value = 0
        good_engine.unregister_memory.return_value = 0
        ready_manager = AscendKVManager.__new__(AscendKVManager)
        ready_manager.disaggregation_mode = DisaggregationMode.DECODE
        ready_manager.engine = SimpleNamespace(engine=good_engine)
        ready_manager.bind_sparse_decode_bridge_resources(ready_resources)
        candidate = BridgeMetadata.decode(ready_manager._npu_mla_candidate_metadata)
        self.assertEqual(candidate.ring_pointer, ready_resources.ring.transfer.pointer)
        self.assertEqual(candidate.ring_bytes, 2 * ALIGN)
        ready_resources.close(
            good_engine, transport_drained=True, synchronize=lambda: None
        )
        bad_args = SimpleNamespace(kv_data_ptrs=[], kv_data_lens=[], kv_item_lens=[])
        another = SparseDecodeBridgeResources(pool, geometry, device="cpu")
        with self.assertRaises(ValueError):
            another.bind_native_manifest(None, bad_args)
        another.close(good_engine, transport_drained=True, synchronize=lambda: None)
        raw_engine = Mock()
        raw_engine.batch_register_memory.return_value = -1
        raw_engine.unregister_memory.return_value = 0
        kv_manager = AscendKVManager.__new__(AscendKVManager)
        kv_manager.disaggregation_mode = DisaggregationMode.DECODE
        kv_manager.engine = SimpleNamespace(engine=raw_engine)
        with self.assertRaises(RuntimeError):
            kv_manager.bind_sparse_decode_bridge_resources(resources)
        self.assertIs(kv_manager._sparse_decode_bridge_resources, resources)
        kv_manager.npu_mla_bridge_metadata = b"advertised"
        with self.assertRaises(RuntimeError):
            kv_manager.teardown()
        self.assertIs(kv_manager._sparse_decode_bridge_resources, resources)
        coverage = resources.register_room(
            room=5,
            session="s",
            pages=[0],
            num_tokens=2,
            prefix_tokens=0,
            draft_pool=None,
        )
        self.assertEqual(
            coverage.reserve(
                room=5,
                generation=coverage.generation,
                session="s",
                layer=3,
                start=0,
                count=2,
                lease_id=1,
            ),
            (0, 1),
        )
        with self.assertRaises(RuntimeError):
            resources.close_room(5, globally_drained=False)
        resources.close_room(5, globally_drained=True)
        resources.close(raw_engine, transport_drained=True, synchronize=lambda: None)
        pool.sparse_kv_manager = None
        with self.assertRaises(ValueError):
            collect_sparse_decode_entries(pool, geometry)

    def test_partial_scatter_failure_cannot_retry_or_reuse_slot(self):
        import torch
        from unittest.mock import Mock

        from sglang.srt.disaggregation.ascend.mla_bridge_receiver import (
            BridgeReceiverRing,
        )

        ring = BridgeReceiverRing("cpu")
        ring.open_room(1, 1, "s", {"kv"})
        slot, lease = ring.acquire(1, 1, "s", "kv")
        ring.write_drained(slot, 1, 1, "s", lease)
        idx = torch.tensor([0], dtype=torch.long)
        kw = dict(
            rows=1,
            host=torch.empty((1, 576), dtype=torch.bfloat16),
            src_indices=idx,
            dst_indices=idx,
            valid=torch.tensor([True]),
            host_dev_ptr=1,
            event_factory=Mock(),
            scatter=Mock(side_effect=RuntimeError("partial enqueue")),
        )
        with self.assertRaises(RuntimeError):
            ring.submit_scatter(slot, 1, 1, "s", lease, **kw)
        self.assertIs(ring._submitted_inputs[(slot, 1, 1, "s", lease)][1], idx)
        ring.abort_room(1, 1, "s")
        with self.assertRaises(ValueError):
            ring.submit_scatter(slot, 1, 1, "s", lease, **kw)
        with self.assertRaises(RuntimeError):
            ring.cancel_aborted_unsubmitted(slot, 1, 1, "s", lease)
        with self.assertRaises(RuntimeError):
            ring.close(Mock(), transport_drained=True, synchronize=lambda: None)

    def test_receiver_slot_is_not_reused_before_scatter_event(self):
        import torch
        from unittest.mock import Mock

        from sglang.srt.disaggregation.ascend.mla_bridge_receiver import (
            BridgeReceiverRing,
        )

        ring = BridgeReceiverRing("cpu")
        ring.open_room(1, 1, "session", {"kv"})
        slot, lease = ring.acquire(1, 1, "session", "kv")
        ring.write_drained(slot, 1, 1, "session", lease)
        event = Mock()
        event.query.side_effect = [False, True]
        index = torch.tensor([0], dtype=torch.long)
        ring.submit_scatter(
            slot,
            1,
            1,
            "session",
            lease,
            rows=1,
            host=torch.empty((1, 576), dtype=torch.bfloat16),
            src_indices=index,
            dst_indices=index,
            valid=torch.tensor([True]),
            host_dev_ptr=1,
            event_factory=lambda: event,
            scatter=Mock(),
        )
        self.assertFalse(ring.finish_scatter(slot, 1, 1, "session", lease))
        with self.assertRaises(RuntimeError):
            ring.close(Mock(), transport_drained=True, synchronize=lambda: None)
        self.assertTrue(ring.finish_scatter(slot, 1, 1, "session", lease))
        self.assertFalse(ring._submitted_inputs)
        self.assertTrue(ring.room_success(1, 1, "session"))
        with self.assertRaises(RuntimeError):
            ring.close(Mock(), transport_drained=True, synchronize=lambda: None)
        ring.close_room(1, 1, "session")
        ring.close(Mock(), transport_drained=True, synchronize=lambda: None)
        with self.assertRaises(RuntimeError):
            ring.open_room(2, 2, "session", {"kv"})

    def test_shared_pack_workspace_keeps_lease_until_all_work_drains(self):
        import torch
        from unittest.mock import Mock

        from sglang.srt.disaggregation.ascend.mla_bridge_workspace import (
            PackedBridgeWorkspace,
        )

        workspace = PackedBridgeWorkspace("cpu")
        engine = Mock()
        engine.batch_register_memory.return_value = 0
        engine.unregister_memory.return_value = 0
        engine.batch_transfer_sync_write.return_value = 0
        workspace.transfer.register(engine)
        lease = workspace.try_acquire()
        self.assertIsNone(workspace.try_acquire())
        k = torch.ones((2, 512), dtype=torch.bfloat16)
        v = torch.full((2, 64), 2, dtype=torch.bfloat16)
        packed = workspace.pack(lease, k, v, torch.tensor([1, 0]))
        self.assertTrue(torch.equal(packed, torch.cat((k[[1, 0]], v[[1, 0]]), dim=1)))
        event = Mock()
        workspace.send(
            lease,
            engine,
            "peer",
            rows=2,
            dst_base=ALIGN,
            dst_size=2 * ALIGN,
            dst_offset=ALIGN,
            pack_event=event,
        )
        engine.batch_transfer_sync_write.assert_called_once_with(
            "peer", [workspace.transfer.pointer], [2 * ALIGN], [2 * ROW_BYTES]
        )
        event.synchronize.assert_called_once()
        with self.assertRaises(RuntimeError):
            workspace.send(
                lease,
                engine,
                "peer",
                rows=2,
                dst_base=ALIGN,
                dst_size=2 * ALIGN,
                dst_offset=ALIGN,
                pack_event=event,
            )
        with self.assertRaises(RuntimeError):
            workspace.release(lease, transport_drained=True, device_work_drained=False)
        with self.assertRaises(RuntimeError):
            workspace.close(engine, transport_drained=True, synchronize=lambda: None)
        workspace.release(lease, transport_drained=True, device_work_drained=True)
        with self.assertRaises(ValueError):
            workspace.release(lease, transport_drained=True, device_work_drained=True)
        workspace.close(engine, transport_drained=True, synchronize=lambda: None)

    def test_buffer_retains_owned_memory_until_drain_and_unregister(self):
        from unittest.mock import Mock

        from sglang.srt.disaggregation.ascend.mla_bridge_buffer import (
            RegisteredBridgeBuffer,
        )

        engine = Mock()
        engine.batch_register_memory.return_value = -1
        engine.unregister_memory.return_value = -1
        buffer = RegisteredBridgeBuffer(usable_bytes=2 * ALIGN, device="cpu")
        self.assertEqual(buffer.pointer % ALIGN, 0)
        self.assertLessEqual(
            buffer.pointer + buffer.size, buffer.raw.data_ptr() + buffer.raw.numel()
        )
        with self.assertRaises(RuntimeError):
            buffer.register(engine)
        with self.assertRaises(RuntimeError):
            buffer.close(engine, transport_drained=False, synchronize=lambda: None)
        with self.assertRaises(RuntimeError):
            buffer.close(engine, transport_drained=True, synchronize=lambda: None)
        self.assertIsNotNone(buffer.raw)
        engine.unregister_memory.return_value = 0
        buffer.close(engine, transport_drained=True, synchronize=lambda: None)
        self.assertIsNone(buffer.raw)

    def test_terminal_poll_can_be_held_before_conclusion(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver

        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.conclude_state = None
        receiver.bootstrap_room = 1
        receiver.kv_mgr = SimpleNamespace(check_status=Mock(return_value=KVPoll.Failed))
        receiver.filter_transfer_status = Mock(return_value=KVPoll.Transferring)
        self.assertEqual(receiver._poll(), KVPoll.Transferring)
        self.assertIsNone(receiver.conclude_state)
        receiver.filter_transfer_status = Mock(return_value=KVPoll.Failed)
        self.assertEqual(receiver._poll(), KVPoll.Failed)
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)

    def test_wire_keeps_legacy_frame_count_and_rejects_unconfirmed_peer(self):
        from threading import Lock
        from types import SimpleNamespace
        from unittest.mock import Mock

        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver

        args = SimpleNamespace(
            kv_data_ptrs=[1],
            aux_data_ptrs=[2],
            state_data_ptrs=[],
            state_item_lens=[],
            state_layer_ids=[],
            kv_layer_ids=[],
            engine_rank=0,
            kv_item_lens=[128],
        )
        manager = SimpleNamespace(
            kv_args=args,
            attn_tp_size=1,
            dcp_size=1,
            dcp_rank=0,
            enable_staging=False,
            local_ip="127.0.0.1",
            rank_port=5,
            record_failure=Mock(),
            update_status=Mock(),
        )
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_infos = [{}]
        receiver.bootstrap_room = 4
        receiver.session_id = "session"
        receiver.prefill_info = SimpleNamespace(npu_mla_bridge_capability=None)
        sock = Mock()
        receiver._connect_to_bootstrap_server = Mock(return_value=(sock, Lock()))
        self.assertTrue(receiver._register_kv_args())
        import struct

        self.assertEqual(
            sock.send_multipart.call_args.args[0],
            [
                b"None",
                b"127.0.0.1",
                b"5",
                b"session",
                struct.pack("Q", 1),
                struct.pack("Q", 2),
                b"",
                b"0",
                b"1",
                b"128",
                b"",
                b"",
                b"",
                b"",
                b"",
                b"",
                b"1",
                b"0",
                b"",
                struct.pack("Q", 128),
            ],
        )
        manager.npu_mla_bridge_metadata = b"not-even-decoded"
        sock.reset_mock()
        self.assertFalse(receiver._register_kv_args())
        sock.send_multipart.assert_not_called()
        del manager.npu_mla_bridge_metadata
        manager._sparse_decode_bridge_resources = object()
        self.assertFalse(receiver._register_kv_args())
        sock.send_multipart.assert_not_called()

    def test_bootstrap_capability_is_optional_and_consistent(self):
        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        class Request:
            def __init__(self, data=None):
                self.data = data
                self.query = dict(
                    prefill_dp_rank="-1",
                    prefill_cp_rank="-1",
                    target_tp_rank="-1",
                    target_pp_rank="-1",
                )

            async def json(self):
                return self.data

        def server():
            s = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
            for name in (
                "attn_tp_size",
                "attn_cp_size",
                "dp_size",
                "pp_size",
                "page_size",
                "kv_cache_dtype",
                "kv_layout",
                "npu_mla_bridge_capability",
                "dsv41_spec_layout",
                "prefill_http_port",
                "follow_bootstrap_room",
                "enable_dsa_cache_layer_split",
            ):
                setattr(s, name, None)
            s._registered_count = 0
            s.decode_allocation_policy = "early"
            s.prefill_port_table = {}
            s.lock = asyncio.Lock()
            return s

        async def exercise():
            data = dict(
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                pp_size=1,
                pp_rank=0,
                system_dp_size=1,
                system_dp_rank=0,
                rank_ip="127.0.0.1",
                rank_port=1,
                page_size=128,
                kv_cache_dtype="bfloat16",
            )
            old = server()
            self.assertEqual((await old._handle_route_put(Request(data))).status, 200)
            old_payload = json.loads((await old._handle_route_get(Request())).text)
            self.assertNotIn("npu_mla_bridge_capability", old_payload)
            capable = server()
            with_capability = dict(data, npu_mla_bridge_capability=LAYOUT)
            self.assertEqual(
                (await capable._handle_route_put(Request(with_capability))).status, 200
            )
            new_payload = json.loads((await capable._handle_route_get(Request())).text)
            self.assertEqual(new_payload.pop("npu_mla_bridge_capability"), LAYOUT)
            self.assertEqual(new_payload, old_payload)
            self.assertEqual(
                (await capable._handle_route_put(Request(data))).status, 400
            )
            pending = server()
            base = dict(data, attn_tp_size=16, npu_mla_bridge_pending=True)
            ranks = [dict(base, attn_tp_rank=i, rank_port=i + 1) for i in range(16)]
            for invalid in (
                dict(base, attn_tp_rank=16),
                dict(base, attn_tp_rank=-1),
                dict(base, attn_tp_size=2),
                dict(base, attn_cp_size=2),
                dict(base, attn_tp_rank=True),
            ):
                self.assertEqual(
                    (await pending._handle_route_put(Request(invalid))).status, 400
                )
            for rank in ranks:
                self.assertEqual(
                    (await pending._handle_route_put(Request(rank))).status, 200
                )
            for rank in ranks[:-1]:
                self.assertEqual(
                    (
                        await pending._handle_route_put(
                            Request(dict(rank, npu_mla_bridge_capability=LAYOUT))
                        )
                    ).status,
                    200,
                )
            self.assertFalse(pending._is_ready())
            self.assertEqual(
                (
                    await pending._handle_route_put(
                        Request(dict(ranks[-1], npu_mla_bridge_capability=LAYOUT))
                    )
                ).status,
                200,
            )
            self.assertTrue(pending._is_ready())
            self.assertEqual(pending._registered_count, 16)
            self.assertEqual(
                (await pending._handle_route_put(Request(ranks[0]))).status, 200
            )
            self.assertEqual(
                (
                    await pending._handle_route_put(
                        Request(dict(ranks[0], npu_mla_bridge_capability=LAYOUT))
                    )
                ).status,
                200,
            )
            self.assertEqual(pending._registered_count, 16)
            self.assertTrue(pending._is_ready())
            ready_payload = json.loads(
                (await pending._handle_route_get(Request())).text
            )
            self.assertEqual(ready_payload["npu_mla_bridge_capability"], LAYOUT)
            self.assertEqual(
                (
                    await pending._handle_route_put(
                        Request(
                            dict(
                                ranks[0],
                                rank_port=999,
                                npu_mla_bridge_capability=LAYOUT,
                            )
                        )
                    )
                ).status,
                400,
            )

        asyncio.run(exercise())

    def test_decode_export_hook_preserves_default_path(self):
        pool, args = object(), object()
        expected = ([1], [2], [3])

        def native(actual_pool):
            self.assertIs(actual_pool, pool)
            return expected

        def forbidden():
            raise AssertionError("non-Ascend path resolved a manager early")

        self.assertIs(
            prepare_decode_kv_buffers(
                is_ascend=False,
                manager_factory=forbidden,
                native_export=native,
                kv_args=args,
                pool=pool,
            ),
            expected,
        )
        self.assertIs(
            prepare_decode_kv_buffers(
                is_ascend=True,
                manager_factory=lambda: object,
                native_export=native,
                kv_args=args,
                pool=pool,
            ),
            expected,
        )

        class Backend:
            @staticmethod
            def prepare_decode_kv_buffers(actual_args, actual_pool):
                if actual_args is not args or actual_pool is not pool:
                    raise AssertionError("factory did not receive original ownership")
                return expected

        self.assertIs(
            prepare_decode_kv_buffers(
                is_ascend=True,
                manager_factory=lambda: Backend,
                native_export=lambda _: forbidden(),
                kv_args=args,
                pool=pool,
            ),
            expected,
        )

    def test_geometry_and_bounded_allocation(self):
        geometry = BridgeGeometry(LAYOUT, 16, 1, 1, 128, "bfloat16", 512, 64)
        geometry.validate_peer(geometry)
        with self.assertRaises(ValueError):
            geometry.validate_peer(
                BridgeGeometry(LAYOUT, 16, 1, 2, 128, "bfloat16", 512, 64)
            )
        budget = BridgeBudget()
        budget.validate()
        self.assertEqual(budget.ring_allocation_bytes, 3 * ALIGN)
        self.assertEqual(budget.scratch_allocation_bytes, 2 * ALIGN)
        self.assertLessEqual(budget.tokens_per_chunk * ROW_BYTES, ALIGN)
        self.assertGreater((budget.tokens_per_chunk + 1) * ROW_BYTES, ALIGN)
        self.assertEqual(budget.pack_transient_reserve_bytes, ALIGN)

    def test_entry_identity_separates_main_index_and_draft(self):
        staged = BridgeEntry(
            "staged", "main", 0, "combined", "bfloat16", 1152, 128, "token", 128
        )
        main_index = BridgeEntry(
            "native", "main", 0, "index", "uint8", 128, 128, "page", 128
        )
        draft_ckv = BridgeEntry(
            "native", "draft", 0, "ckv", "bfloat16", 1024, 128, "page", 128
        )
        geometry = BridgeGeometry(LAYOUT, 16, 1, 1, 128, "bfloat16", 512, 64)
        metadata = BridgeMetadata(
            geometry, (staged, main_index, draft_ckv), ALIGN, 2 * ALIGN
        )
        required = [(x.kind, x.model, x.layer, x.component) for x in metadata.entries]
        self.assertEqual(
            validate_bridge_manifest(metadata.entries, required), metadata.entries
        )
        with self.assertRaises(ValueError):
            validate_bridge_manifest((staged, main_index), required)
        self.assertEqual(BridgeMetadata.decode(metadata.encode()), metadata)
        source = (
            BridgeEntry(
                "native", "main", 0, "ckv", "bfloat16", 131072, 128, "page", 128
            ),
            BridgeEntry(
                "native", "main", 0, "kpe", "bfloat16", 16384, 128, "page", 128
            ),
            main_index,
            draft_ckv,
        )
        self.assertEqual(len(pair_bridge_entries(source, metadata.entries)), 4)
        with self.assertRaises(ValueError):
            pair_bridge_entries(source[:-1], metadata.entries)
        with self.assertRaises(ValueError):
            pair_bridge_entries(source, (staged, draft_ckv))
        with self.assertRaises(ValueError):
            BridgeMetadata.decode(metadata.encode().replace(b'"dcp":1', b'"dcp":true'))
        with self.assertRaises(ValueError):
            BridgeMetadata.decode(b"{}")
        self.assertEqual(
            len(validate_bridge_entries([staged, main_index, draft_ckv])), 3
        )
        with self.assertRaises(ValueError):
            validate_bridge_entries([staged, main_index, main_index])
        with self.assertRaises(ValueError):
            validate_bridge_entries(
                [
                    BridgeEntry(
                        "staged",
                        "draft",
                        0,
                        "combined",
                        "bfloat16",
                        1152,
                        128,
                        "token",
                        128,
                    )
                ]
            )

    def test_physical_ids_validate_before_transfer(self):
        self.assertEqual(
            checked_token_pairs([1, 1, 0], [4, 2, 3], 2, 5), ((1, 1, 0), (4, 2, 3))
        )
        for src, dst in [([0, 1], [3, 3]), ([2], [0]), ([0], [5]), ([True], [0])]:
            with self.assertRaises(ValueError):
                checked_token_pairs(src, dst, 2, 5)

    def test_page_expansion_preserves_noncontiguous_partial_tail(self):
        import numpy as np

        source, destination = expand_physical_pages(
            np.array([2, 0], dtype=np.int32),
            np.array([4, 1], dtype=np.int32),
            segment_start=128,
            num_tokens=130,
            src_capacity=384,
            dst_capacity=640,
        )
        self.assertEqual(source, tuple(range(256, 384)) + (0, 1))
        self.assertEqual(destination, tuple(range(512, 640)) + (128, 129))
        with self.assertRaises(ValueError):
            expand_physical_pages(
                [0, 1],
                [0, 0],
                segment_start=0,
                num_tokens=130,
                src_capacity=256,
                dst_capacity=256,
            )
        with self.assertRaises(ValueError):
            expand_physical_pages(
                [0],
                [0],
                segment_start=0,
                num_tokens=129,
                src_capacity=256,
                dst_capacity=256,
            )
        with self.assertRaises(ValueError):
            expand_physical_pages(
                [0],
                [0],
                segment_start=1,
                num_tokens=1,
                src_capacity=256,
                dst_capacity=256,
            )

    def test_native_plane_respects_independent_strides_and_bounds(self):
        source = bytes(range(128))
        target = bytearray([255] * 192)
        blocks = plan_native_page_transfers(
            [3, 0, 3],
            [1, 2, 0],
            src_base=4096,
            dst_base=8192,
            src_bytes=128,
            dst_bytes=192,
            src_stride=32,
            dst_stride=48,
            item_bytes=16,
        )
        for src, dst, size in blocks:
            target[dst - 8192 : dst - 8192 + size] = source[
                src - 4096 : src - 4096 + size
            ]
        self.assertEqual(target[48:64], source[96:112])
        self.assertEqual(target[96:112], source[:16])
        self.assertEqual(target[:16], source[96:112])
        self.assertEqual(target[16:48], bytes([255] * 32))
        with self.assertRaises(ValueError):
            plan_native_page_transfers(
                [0],
                [4],
                src_base=1,
                dst_base=1,
                src_bytes=128,
                dst_bytes=192,
                src_stride=32,
                dst_stride=48,
                item_bytes=16,
            )

    def test_packed_rows_preserve_split_plane_and_order(self):
        import torch

        k = torch.arange(4 * 512, dtype=torch.float32).view(4, 512).to(torch.bfloat16)
        v = (torch.arange(4 * 64, dtype=torch.float32).view(4, 64) + 5000).to(
            torch.bfloat16
        )
        indices = torch.tensor([3, 0, 3], dtype=torch.int64)
        packed = torch.empty((3, 576), dtype=torch.bfloat16)
        pack_mla_rows(
            k,
            v,
            indices,
            packed,
            torch.empty((3, 512), dtype=torch.bfloat16),
            torch.empty((3, 64), dtype=torch.bfloat16),
        )
        self.assertTrue(torch.equal(packed, torch.cat((k[indices], v[indices]), dim=1)))
        with self.assertRaises(ValueError):
            pack_mla_rows(
                k, v, indices, packed, k[:3], torch.empty((3, 64), dtype=torch.bfloat16)
            )

    def test_abort_requires_write_and_scatter_drain(self):
        leases = BridgeLeases(1)
        leases.open(7, 1, "session", {"kv", "index", "draft"})
        slot, lease = leases.acquire(7, 1, "session", "kv")
        leases.abort(7, 1, "session")
        with self.assertRaises(ValueError):
            leases.write_drained(slot, 7, 1, "other-session", lease)
        with self.assertRaises(ValueError):
            leases.release(slot, 7, 1, "session", lease)
        with self.assertRaises(ValueError):
            leases.open(7, 2, "new", {"kv"})
        with self.assertRaises(ValueError):
            leases.close(7, 1, "session")
        leases.write_drained(slot, 7, 1, "session", lease)
        leases.release(slot, 7, 1, "session", lease)
        self.assertFalse(leases.success(7, 1, "session"))
        leases.close(7, 1, "session")
        leases.open(7, 2, "new", {"kv"})
        with self.assertRaises(ValueError):
            leases.write_drained(slot, 7, 1, "session", lease)

    def test_success_waits_for_all_components_and_scatter(self):
        leases = BridgeLeases(1)
        leases.open(1, 4, "peer", {"kv", "index", "scale", "draft"})
        for item in ("kv", "index", "scale", "draft"):
            slot, lease = leases.acquire(1, 4, "peer", item)
            with self.assertRaises(ValueError):
                leases.acquire(1, 4, "peer", item)
            leases.write_drained(slot, 1, 4, "peer", lease)
            with self.assertRaises(ValueError):
                leases.release(slot, 1, 4, "peer", lease)
            leases.scatter_drained(slot, 1, 4, "peer", lease)
            leases.release(slot, 1, 4, "peer", lease)
            if item != "draft":
                self.assertFalse(leases.success(1, 4, "peer"))
        self.assertTrue(leases.success(1, 4, "peer"))
        with self.assertRaises(ValueError):
            leases.release(slot, 1, 4, "peer", lease)
        leases.close(1, 4, "peer")
        with self.assertRaises(ValueError):
            leases.open(1, 4, "peer", {"kv"})


if __name__ == "__main__":
    unittest.main()
