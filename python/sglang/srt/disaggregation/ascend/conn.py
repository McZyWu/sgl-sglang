import concurrent.futures
import enum
import logging
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)


def _wait_for_npu_mla_bridge_chunk(
    resources, *, room, generation, session, layer, slot, lease_id, deadline
):
    """Wait for one scatter completion with a bounded scheduling yield."""
    false_queries = 0
    while not resources.finish_chunk(
        room=room,
        generation=generation,
        session=session,
        layer=layer,
        slot=slot,
        lease_id=lease_id,
    ):
        if time.monotonic() >= deadline:
            raise TimeoutError("NPU MLA scatter event did not finish")
        false_queries += 1
        time.sleep(0 if false_queries <= 4 else 0.001)


class AscendStateType(str, enum.Enum):
    """DSV4-on-NPU PD components without a cross-hardware equivalent."""

    DSV4_C128 = "dsv4_c128"
    # C4 compress-state rows (attention + indexer) addressed within each
    # req_pool_idx bank on A5 (CYCLE cache_mode).  Separate from StateType.SWA
    # because each peer maps logical positions into its own local ring.
    DSV4_C4_STATE = "dsv4_c4_state"


_DSV4_KVCACHE_STATE_TYPES = tuple(AscendStateType)


def _build_page_interleaved_dcp_plan(
    src_page_indices: npt.NDArray[np.int32],
    dst_page_indices: npt.NDArray[np.int32],
    *,
    page_size: int,
    dcp_size: int,
    dcp_rank: int,
    src_page_offset: int,
    decode_prefix_len: int,
    num_kv_tokens: int,
) -> Tuple[npt.NDArray[np.int64], ...]:
    """Map physical prefill pages to local/global decode page slots."""
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"Invalid DCP rank {dcp_rank} for size {dcp_size}")
    virtual_page_size = page_size * dcp_size
    if decode_prefix_len % virtual_page_size:
        raise ValueError(
            "Ascend PD DCP requires decode_prefix_len to align to the virtual "
            f"page size ({virtual_page_size}), got {decode_prefix_len}"
        )
    if src_page_offset < 0 or num_kv_tokens < 0:
        raise ValueError(
            "Ascend PD DCP page offset and token count must be nonnegative"
        )

    src_pages = np.asarray(src_page_indices, dtype=np.int64)
    dst_pages = np.asarray(dst_page_indices, dtype=np.int64)
    max_src_pages = (num_kv_tokens + page_size - 1) // page_size
    if src_pages.size > max_src_pages:
        raise ValueError(
            "Ascend PD DCP source page count exceeds the token count: "
            f"pages={src_pages.size}, tokens={num_kv_tokens}, page_size={page_size}"
        )
    if src_pages.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    # CP may assign this sender only a contiguous subset of the chunk pages;
    # index_slice.start still carries that subset's suffix-relative offset.
    relative_pages = src_page_offset + np.arange(src_pages.size, dtype=np.int64)
    dst_positions = relative_pages // dcp_size
    if dst_positions[-1] >= dst_pages.size:
        raise ValueError(
            "Ascend PD DCP destination does not contain enough virtual pages: "
            f"required={dst_positions[-1] + 1}, available={dst_pages.size}"
        )
    dst_super_pages = dst_pages[dst_positions]
    owners = (decode_prefix_len // page_size + relative_pages) % dcp_size
    local = owners == dcp_rank

    return (
        src_pages[local],
        dst_super_pages[local],
        src_pages,
        dst_super_pages * dcp_size + owners,
    )


class AscendKVManager(MooncakeKVManager):
    @staticmethod
    def prepare_prefill_mla_bridge_inputs(kv_args, pool, draft_pool):
        """Retain ordered native MLA plane views for a future bridge peer."""
        from sglang.srt.disaggregation.ascend.mla_bridge_collect import (
            collect_native_prefill_planes,
        )
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

        parallel = get_parallel()
        if (
            not isinstance(pool, NPUMLATokenToKVPool)
            or (
                draft_pool is not None
                and not isinstance(draft_pool, NPUMLATokenToKVPool)
            )
            or pool.enable_sparsity_driven_kv_offload
            or pool.kv_lora_rank != 512
            or pool.qk_rope_head_dim != 64
            or str(pool.store_dtype) != "torch.bfloat16"
            or pool.page_size != 128
            or parallel.attn_tp_size != 16
            or parallel.pp_size != 1
            or parallel.attn_dcp_size != 1
        ):
            return
        try:
            planes = collect_native_prefill_planes(pool, draft_pool)
        except ValueError as exc:
            logger.debug("NPU MLA bridge prefill geometry unsupported: %s", exc)
            return
        kv_args._npu_mla_prefill_planes = planes

    @staticmethod
    def prepare_decode_kv_buffers(kv_args, pool):
        """Prepare sparse target metadata without exporting absent KV planes."""
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
            LAYOUT,
            BridgeGeometry,
        )
        from sglang.srt.disaggregation.ascend.mla_bridge_resources import (
            SparseDecodeBridgeResources,
        )
        from sglang.srt.disaggregation.utils import get_kv_transfer_buf_infos
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

        if (
            not isinstance(pool, NPUMLATokenToKVPool)
            or not pool.enable_sparsity_driven_kv_offload
            or not envs.SGLANG_NPU_ENABLE_PD_MLA_BRIDGE.get()
        ):
            return get_kv_transfer_buf_infos(pool)
        parallel = get_parallel()
        geometry = BridgeGeometry(
            LAYOUT,
            parallel.attn_tp_size,
            parallel.pp_size,
            parallel.attn_dcp_size,
            pool.page_size,
            str(pool.store_dtype).removeprefix("torch."),
            pool.kv_lora_rank,
            pool.qk_rope_head_dim,
        )
        resources = SparseDecodeBridgeResources(
            pool, geometry, device=pool.index_k_buffer.device
        )
        kv_args._npu_mla_bridge_resources = resources
        entries = resources.entries
        return (
            list(entries.native_ptrs),
            list(entries.native_sizes),
            list(entries.native_strides),
        )

    def validate_npu_mla_bridge_registration(
        self, metadata: bytes, registration=None
    ) -> Optional[str]:
        """Reject incomplete or incompatible bridge manifests before transfer."""
        if not getattr(self, "_npu_mla_bridge_ready", False):
            return "NPU MLA bridge sender is not ready"
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
            BridgeMetadata,
            pair_bridge_entries,
        )

        try:
            decoded = BridgeMetadata.decode(metadata)
            planes = getattr(self.kv_args, "_npu_mla_prefill_planes", None)
            if not planes:
                raise ValueError("Prefill native plane manifest is missing")
            if tuple(p.tensor.data_ptr() for p in planes) != tuple(
                self.kv_args.kv_data_ptrs
            ):
                raise ValueError("Prefill native pointers changed after collection")
            pair_bridge_entries((p.entry for p in planes), decoded.entries)
            if registration is None:
                raise ValueError("Decode native registration is missing")
            native = [entry for entry in decoded.entries if entry.kind == "native"]
            if (
                len(registration.dst_kv_ptrs) != len(native)
                or any(
                    type(ptr) is not int or ptr <= 0 for ptr in registration.dst_kv_ptrs
                )
                or list(registration.dst_kv_item_lens)
                != [entry.item_stride for entry in native]
            ):
                raise ValueError(
                    "Decode native pointer or page stride manifest mismatch"
                )
        except (TypeError, ValueError, AttributeError) as exc:
            return f"Invalid NPU MLA bridge registration: {exc}"
        return None

    def prepare_npu_mla_sender_workspace(self):
        """Prepare one packed sender slot without enabling capability."""
        from sglang.srt.disaggregation.ascend.mla_bridge_workspace import (
            PackedBridgeWorkspace,
        )

        if getattr(self, "_npu_mla_sender_workspace", None) is not None:
            raise RuntimeError("NPU MLA sender workspace is already bound")
        if not getattr(self.kv_args, "_npu_mla_prefill_planes", None):
            raise ValueError("Prefill MLA native planes are unavailable")
        workspace = PackedBridgeWorkspace(f"npu:{self.kv_args.gpu_id}")
        self._npu_mla_sender_workspace = workspace
        workspace.transfer.register(self.engine.engine)
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import BridgeBudget

        budget = BridgeBudget()
        logger.info(
            "NPU MLA bridge P: allocated raw_scratch=%d usable_slot=%d gather=%d bytes; "
            "allocated indices=%d runtime_reserve=%d bytes; pool_budget_deducted=false",
            workspace.transfer.raw.nbytes,
            workspace.transfer.size,
            workspace.k_workspace.nbytes + workspace.v_workspace.nbytes,
            workspace.index_buffer.nbytes,
            budget.pack_transient_reserve_bytes,
        )

    def _do_npu_mla_bridge_transfer(
        self, kv_chunk, req, target_info, dst_indices, executor, queue, writer_id
    ):
        """Pack, send, ACK and then transfer native planes for one P segment.

        The experimental feature flag enables this path. A failed or uncertain
        operation quarantines room state and any held workspace lease.
        """
        import torch

        from sglang.srt.disaggregation.ascend.mla_bridge_collect import (
            plan_native_component_blocks,
        )
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
            BridgeMetadata,
            BridgeRoomDescriptor,
            bridge_segment_token_bounds,
            expand_physical_pages,
        )

        workspace = getattr(self, "_npu_mla_sender_workspace", None)
        if workspace is None or not getattr(self, "_npu_mla_bridge_ready", False):
            return -1, False
        with self._npu_mla_rpc_lock:
            if self._npu_mla_shutdown or req.room in self._npu_mla_unsettled_rooms:
                return -1, False
            if req.room in self._npu_mla_active_rooms:
                return -1, False
            self._npu_mla_active_rooms.add(req.room)
        try:
            if (
                self.validate_npu_mla_bridge_registration(
                    target_info.npu_mla_bridge_metadata, target_info
                )
                is not None
            ):
                raise ValueError("Peer bridge registration is invalid")
            metadata = BridgeMetadata.decode(target_info.npu_mla_bridge_metadata)
            descriptor = BridgeRoomDescriptor.decode(req.npu_mla_room)
            if descriptor.prefix_tokens != (req.decode_prefix_len or 0):
                raise ValueError("Bridge room prefix differs from transfer request")
            start, end = bridge_segment_token_bounds(
                kv_chunk.index_slice, kv_chunk.num_kv_tokens, descriptor.num_tokens
            )
            count = end - start
            planes = tuple(self.kv_args._npu_mla_prefill_planes)
            main_kv = {
                (p.entry.layer, p.entry.component): p
                for p in planes
                if p.entry.model == "main" and p.entry.component in ("ckv", "kpe")
            }
            staged = [entry for entry in metadata.entries if entry.kind == "staged"]
            src_capacity = min(p.entry.capacity * 128 for p in main_kv.values())
            dst_capacity = min(entry.capacity for entry in staged)
            src_ids, _ = expand_physical_pages(
                kv_chunk.prefill_kv_indices,
                dst_indices,
                segment_start=start,
                num_tokens=count,
                src_capacity=src_capacity,
                dst_capacity=dst_capacity,
            )
            torch.npu.set_device(self.kv_args.gpu_id)
            for entry in staged:
                ckv, kpe = main_kv[(entry.layer, "ckv")], main_kv[(entry.layer, "kpe")]
                for offset in range(0, count, workspace.capacity):
                    part = src_ids[offset : offset + workspace.capacity]
                    deadline = time.monotonic() + 30
                    lease = workspace.try_acquire()
                    while lease is None:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("NPU MLA packed workspace is busy")
                        time.sleep(0.001)
                        lease = workspace.try_acquire()
                    remote = self.request_npu_mla_lease(
                        req,
                        descriptor,
                        entry.layer,
                        start + offset,
                        len(part),
                        timeout=30,
                    )
                    while remote is None:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("NPU MLA receiver slots are busy")
                        time.sleep(0.001)
                        remote = self.request_npu_mla_lease(
                            req,
                            descriptor,
                            entry.layer,
                            start + offset,
                            len(part),
                            timeout=30,
                        )
                    slot, remote_lease = remote
                    indices = workspace.prepare_indices(
                        lease,
                        (
                            req.room,
                            descriptor.generation,
                            req.mooncake_session_id,
                            start,
                            end,
                        ),
                        part,
                    )
                    workspace.pack_native_planes(lease, ckv, kpe, indices)
                    pack_event = torch.npu.Event()
                    pack_event.record()
                    workspace.send(
                        lease,
                        self.engine.engine,
                        req.mooncake_session_id,
                        rows=len(part),
                        dst_base=metadata.ring_pointer,
                        dst_size=metadata.ring_bytes,
                        dst_offset=slot * workspace.transfer.size,
                        pack_event=pack_event,
                    )
                    self.notify_npu_mla_chunk_and_wait(
                        req,
                        descriptor,
                        entry.layer,
                        start + offset,
                        len(part),
                        slot,
                        remote_lease,
                        timeout=30,
                    )
                    workspace.release(
                        lease, transport_drained=True, device_work_drained=True
                    )
            native_pages = dst_indices
            if req.dst_device_kv_indices is not None:
                native_pages = req.dst_device_kv_indices[kv_chunk.index_slice]
            planned = plan_native_component_blocks(
                planes,
                metadata.entries,
                target_info.dst_kv_ptrs,
                kv_chunk.prefill_kv_indices,
                native_pages,
            )
            for _, blocks in planned:
                for offset in range(0, len(blocks), 128):
                    if (
                        self._transfer_data(
                            req.mooncake_session_id, blocks[offset : offset + 128]
                        )
                        != 0
                    ):
                        raise RuntimeError("NPU MLA native transfer failed")
            na = NetworkAddress(req.endpoint, req.dst_port)
            self._send_multipart_locked(
                na.to_tcp(),
                [
                    b"NPU_MLA_NATIVE_READY",
                    str(req.room).encode("ascii"),
                    str(descriptor.generation).encode("ascii"),
                    req.mooncake_session_id.encode("ascii"),
                    str(start).encode("ascii"),
                    str(count).encode("ascii"),
                    self.local_ip.encode("ascii"),
                    str(self.rank_port).encode("ascii"),
                ],
                is_ipv6=na.is_ipv6,
            )
            return 0, False
        except Exception:
            with self._npu_mla_rpc_lock:
                self._npu_mla_unsettled_rooms.add(req.room)
            self._poison_unsettled_send(req.room)
            logger.exception("NPU MLA bridge transfer failed; room remains quarantined")
            return -1, False
        finally:
            with self._npu_mla_rpc_lock:
                self._npu_mla_active_rooms.discard(req.room)

    def request_npu_mla_lease(self, req, descriptor, layer, start, count, *, timeout):
        """Send one lease request and retain an uncertain room on any timeout."""
        key = (
            req.room,
            descriptor.generation,
            req.mooncake_session_id,
            layer,
            start,
            count,
        )
        if timeout <= 0:
            raise ValueError("Lease timeout must be positive")
        waiter = {"event": threading.Event(), "result": None}
        with self._npu_mla_rpc_lock:
            if (
                req.room in self._npu_mla_unsettled_rooms
                or key in self._npu_mla_rpc_pending
            ):
                raise RuntimeError("Bridge room or lease is already unsettled")
            self._npu_mla_rpc_pending[key] = waiter
        parts = [
            b"NPU_MLA_LEASE_REQ",
            *(str(x).encode("ascii") for x in key[:2]),
            key[2].encode("ascii"),
            *(str(x).encode("ascii") for x in key[3:]),
            self.local_ip.encode("ascii"),
            str(self.rank_port).encode("ascii"),
        ]
        try:
            na = NetworkAddress(req.endpoint, req.dst_port)
            self._send_multipart_locked(na.to_tcp(), parts, is_ipv6=na.is_ipv6)
            if not waiter["event"].wait(timeout):
                raise TimeoutError("NPU MLA lease response timed out")
            return waiter["result"]
        except Exception:
            with self._npu_mla_rpc_lock:
                self._npu_mla_unsettled_rooms.add(req.room)
            raise
        finally:
            with self._npu_mla_rpc_lock:
                self._npu_mla_rpc_pending.pop(key, None)

    def handle_npu_mla_bridge_lease_response(self, msg):
        try:
            if len(msg) not in (8, 10):
                raise ValueError("Invalid lease response frame count")
            key = (
                int(msg[1]),
                int(msg[2]),
                msg[3].decode("ascii"),
                int(msg[4]),
                int(msg[5]),
                int(msg[6]),
            )
            if msg[7] == b"BUSY" and len(msg) == 8:
                result = None
            elif msg[7] == b"OK" and len(msg) == 10:
                slot, lease_id = int(msg[8]), int(msg[9])
                if slot not in (0, 1) or lease_id <= 0:
                    raise ValueError("Invalid lease response")
                result = (slot, lease_id)
            else:
                raise ValueError("Invalid lease response status")
            with self._npu_mla_rpc_lock:
                waiter = self._npu_mla_rpc_pending.get(key)
                if waiter is None or waiter["event"].is_set():
                    raise ValueError("Stale or duplicate lease response")
                waiter["result"] = result
                waiter["event"].set()
        except Exception:
            logger.exception("Rejected NPU MLA lease response")

    def notify_npu_mla_chunk_and_wait(
        self, req, descriptor, layer, start, count, slot, lease_id, *, timeout
    ):
        """Notify a completed sync write; only a matching event ACK permits reuse."""
        key = (
            req.room,
            descriptor.generation,
            req.mooncake_session_id,
            layer,
            start,
            count,
            slot,
            lease_id,
        )
        if timeout <= 0:
            raise ValueError("ACK timeout must be positive")
        event = threading.Event()
        with self._npu_mla_rpc_lock:
            if (
                req.room in self._npu_mla_unsettled_rooms
                or key in self._npu_mla_ack_pending
            ):
                raise RuntimeError("Bridge room or ACK is unsettled")
            self._npu_mla_ack_pending[key] = event
        parts = [
            b"NPU_MLA_CHUNK_READY",
            *(str(x).encode("ascii") for x in key[:2]),
            key[2].encode("ascii"),
            *(str(x).encode("ascii") for x in key[3:]),
            self.local_ip.encode("ascii"),
            str(self.rank_port).encode("ascii"),
        ]
        try:
            na = NetworkAddress(req.endpoint, req.dst_port)
            self._send_multipart_locked(na.to_tcp(), parts, is_ipv6=na.is_ipv6)
            if not event.wait(timeout):
                raise TimeoutError("NPU MLA scatter ACK timed out")
        except Exception:
            with self._npu_mla_rpc_lock:
                self._npu_mla_unsettled_rooms.add(req.room)
            raise
        finally:
            with self._npu_mla_rpc_lock:
                self._npu_mla_ack_pending.pop(key, None)

    def handle_npu_mla_bridge_ack(self, msg):
        try:
            if len(msg) != 9:
                raise ValueError("Invalid bridge ACK frame count")
            key = (
                int(msg[1]),
                int(msg[2]),
                msg[3].decode("ascii"),
                *(int(part) for part in msg[4:9]),
            )
            with self._npu_mla_rpc_lock:
                event = self._npu_mla_ack_pending.get(key)
                if event is None or event.is_set():
                    raise ValueError("Stale or duplicate bridge ACK")
                event.set()
        except Exception:
            logger.exception("Rejected NPU MLA bridge ACK")

    def handle_npu_mla_bridge_lease_request(self, msg):
        """Respond only to a registered room peer; failures quarantine its lease."""
        resources = getattr(self, "_sparse_decode_bridge_resources", None)
        if resources is None or getattr(self, "npu_mla_bridge_metadata", None) is None:
            logger.error("Ignoring NPU MLA lease before bridge activation")
            return
        room = generation = session = None
        authenticated = False
        try:
            if len(msg) != 9:
                raise ValueError("Invalid lease request frame count")
            room = int(msg[1].decode("ascii"))
            generation = int(msg[2].decode("ascii"))
            session = msg[3].decode("ascii")
            layer, start, count = (int(part.decode("ascii")) for part in msg[4:7])
            peer = (msg[7].decode("ascii"), int(msg[8].decode("ascii")))
            resources.require_peer(room, generation, session, peer)
            authenticated = True
            lease = resources.reserve_chunk(
                room=room,
                generation=generation,
                session=session,
                layer=layer,
                start=start,
                count=count,
            )
            fields = [b"NPU_MLA_LEASE_RSP", *msg[1:7]]
            if lease is None:
                fields += [b"BUSY"]
            else:
                slot, lease_id, _ = lease
                fields += [
                    b"OK",
                    str(slot).encode("ascii"),
                    str(lease_id).encode("ascii"),
                ]
            na = NetworkAddress(*peer)
            self._send_multipart_locked(na.to_tcp(), fields, is_ipv6=na.is_ipv6)
        except Exception:
            if authenticated:
                try:
                    resources.quarantine_room(room, generation, session)
                except Exception:
                    logger.exception("Failed to quarantine NPU MLA lease room")
            logger.exception("Rejected NPU MLA lease request")

    def handle_npu_mla_bridge_native_ready(self, msg):
        resources = getattr(self, "_sparse_decode_bridge_resources", None)
        if resources is None or getattr(self, "npu_mla_bridge_metadata", None) is None:
            logger.error("Ignoring NPU MLA native completion before activation")
            return
        room = generation = session = None
        authenticated = False
        try:
            if len(msg) != 8:
                raise ValueError("Invalid native completion frame count")
            room, generation = int(msg[1]), int(msg[2])
            session = msg[3].decode("ascii")
            start, count = int(msg[4]), int(msg[5])
            peer = (msg[6].decode("ascii"), int(msg[7]))
            resources.require_peer(room, generation, session, peer)
            authenticated = True
            resources.complete_native_range(
                room=room,
                generation=generation,
                session=session,
                start=start,
                count=count,
            )
        except Exception:
            if authenticated:
                try:
                    resources.quarantine_room(room, generation, session)
                except Exception:
                    logger.exception("Failed to quarantine native completion room")
            logger.exception("Rejected NPU MLA native completion")

    def handle_npu_mla_bridge_chunk(self, msg):
        """Scatter a completed write and ACK only after its device event."""
        resources = getattr(self, "_sparse_decode_bridge_resources", None)
        if resources is None or getattr(self, "npu_mla_bridge_metadata", None) is None:
            logger.error("Ignoring NPU MLA chunk before bridge activation")
            return
        room = generation = session = None
        authenticated = False
        try:
            if len(msg) != 11:
                raise ValueError("Invalid bridge chunk frame count")
            room, generation = int(msg[1]), int(msg[2])
            session = msg[3].decode("ascii")
            layer, start, count, slot, lease_id = (int(part) for part in msg[4:9])
            peer = (msg[9].decode("ascii"), int(msg[10]))
            resources.require_peer(room, generation, session, peer)
            authenticated = True
            rows = resources.reserved_rows(
                room=room,
                generation=generation,
                session=session,
                layer=layer,
                start=start,
                count=count,
                slot=slot,
                lease_id=lease_id,
            )
            if len(rows) != count:
                raise ValueError("Chunk row count differs from its lease")
            import torch

            torch.npu.set_device(self.kv_args.gpu_id)
            device = torch.device(f"npu:{self.kv_args.gpu_id}")
            src = torch.arange(count, dtype=torch.int64, device=device)
            dst = torch.tensor(rows, dtype=torch.int64, device=device)
            valid = torch.ones(count, dtype=torch.bool, device=device)
            host, host_ptr = resources.host_for_layer(layer)
            resources.ring.write_drained(slot, room, generation, session, lease_id)
            resources.ring.submit_scatter(
                slot,
                room,
                generation,
                session,
                lease_id,
                rows=count,
                host=host,
                src_indices=src,
                dst_indices=dst,
                valid=valid,
                host_dev_ptr=host_ptr,
                event_factory=torch.npu.Event,
            )
            deadline = time.monotonic() + 30
            _wait_for_npu_mla_bridge_chunk(
                resources,
                room=room,
                generation=generation,
                session=session,
                layer=layer,
                slot=slot,
                lease_id=lease_id,
                deadline=deadline,
            )
            na = NetworkAddress(*peer)
            self._send_multipart_locked(
                na.to_tcp(), [b"NPU_MLA_ACK", *msg[1:9]], is_ipv6=na.is_ipv6
            )
        except Exception:
            if authenticated:
                try:
                    resources.quarantine_room(room, generation, session)
                except Exception:
                    logger.exception("Failed to quarantine NPU MLA chunk room")
            logger.exception("Rejected or failed NPU MLA chunk")

    def bind_sparse_decode_bridge_resources(self, resources):
        """Retain an unadvertised ring owner before attempting registration.

        This alone does not enable the bridge; transport, completion and
        failure-quarantine integration must be installed before advertisement.
        """
        from sglang.srt.disaggregation.utils import DisaggregationMode

        if self.disaggregation_mode != DisaggregationMode.DECODE:
            raise ValueError("Sparse decode bridge requires decode mode")
        if getattr(self, "_sparse_decode_bridge_resources", None) is not None:
            raise RuntimeError("Sparse decode bridge resources already bound")
        self._sparse_decode_bridge_resources = resources
        resources.register(self.engine.engine)
        self._npu_mla_candidate_metadata = resources.candidate_metadata()
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import BridgeBudget

        budget = BridgeBudget()
        logger.info(
            "NPU MLA bridge D: allocated raw_ring=%d usable_ring=%d native=%d "
            "host=%d bytes; estimated max_indices_mask=%d bytes; pool_budget_deducted=false",
            resources.ring.transfer.raw.nbytes,
            resources.ring.transfer.size,
            resources.native_bytes,
            resources.host_bytes,
            budget.decode_index_mask_bytes,
        )

    def teardown(self):
        workspace = getattr(self, "_npu_mla_sender_workspace", None)
        if workspace is not None:
            with self._npu_mla_rpc_lock:
                self._npu_mla_shutdown = True
                if (
                    self._npu_mla_active_rooms
                    or self._npu_mla_unsettled_rooms
                    or self._npu_mla_rpc_pending
                    or self._npu_mla_ack_pending
                ):
                    raise RuntimeError(
                        "Cannot teardown NPU MLA sender with unsettled work"
                    )
        resources = getattr(self, "_sparse_decode_bridge_resources", None)
        if resources is not None:
            if getattr(self, "npu_mla_bridge_metadata", None) is not None:
                raise RuntimeError(
                    "Cannot teardown an advertised NPU MLA bridge without drain proof"
                )
            import torch

            resources.close(
                self.engine.engine,
                transport_drained=True,
                synchronize=torch.npu.synchronize,
            )
            self._sparse_decode_bridge_resources = None
        result = super().teardown()
        if workspace is not None:
            import torch

            workspace.close(
                self.engine.engine,
                transport_drained=True,
                synchronize=torch.npu.synchronize,
            )
            self._npu_mla_sender_workspace = None
        if hasattr(self.kv_args, "_npu_mla_prefill_planes"):
            self.kv_args._npu_mla_prefill_planes = None
        return result

    def __init__(
        self,
        args,
        disaggregation_mode,
        server_args,
        is_mla_backend: Optional[bool] = False,
        dcp_remote_decode_layout=None,
    ):
        from sglang.srt.disaggregation.utils import DisaggregationMode

        bridge_enabled = envs.SGLANG_NPU_ENABLE_PD_MLA_BRIDGE.get()
        self._npu_mla_bridge_pending = (
            bridge_enabled and disaggregation_mode == DisaggregationMode.PREFILL
        )
        if self._npu_mla_bridge_pending:
            if envs.SGLANG_RUST_SERVER.get():
                raise ValueError("NPU MLA bridge does not support Rust bootstrap")
            if not getattr(args, "_npu_mla_prefill_planes", None):
                raise ValueError("NPU MLA bridge requires supported prefill planes")
        self._npu_mla_rpc_lock = threading.Lock()
        self._npu_mla_rpc_pending = {}
        self._npu_mla_ack_pending = {}
        self._npu_mla_unsettled_rooms = set()
        self._npu_mla_active_rooms = set()
        self._npu_mla_shutdown = False
        self._dcp_remote_decode_layout = (
            None if dcp_remote_decode_layout is None else list(dcp_remote_decode_layout)
        )
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        resources = getattr(args, "_npu_mla_bridge_resources", None)
        if resources is not None:
            self.bind_sparse_decode_bridge_resources(resources)
            if bridge_enabled:
                self.npu_mla_bridge_metadata = self._npu_mla_candidate_metadata
        elif bridge_enabled and disaggregation_mode == DisaggregationMode.DECODE:
            raise ValueError("NPU MLA bridge requires sparse decode resources")
        if self._npu_mla_bridge_pending:
            from sglang.srt.disaggregation.ascend.mla_bridge_contract import LAYOUT

            self.prepare_npu_mla_sender_workspace()
            self._npu_mla_bridge_ready = True
            self.npu_mla_bridge_capability = LAYOUT
            self.register_to_bootstrap(strict=True)

    def _is_npu_dsa_layout(self) -> bool:
        return getattr(self.kv_args, "kv_buf_groups", 1) == 3

    def _requires_exact_state_index_match(self, st: StateType) -> bool:
        return (
            super()._requires_exact_state_index_match(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )

    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        # MemFabric aligns registered buffers to 2 MiB. Register everything in
        # one batch so overlapping aligned ranges from small tensors are merged
        # before they are published to the peer.
        ptrs = list(self.kv_args.kv_data_ptrs)
        lens = list(self.kv_args.kv_data_lens)
        ptrs.extend(self.kv_args.aux_data_ptrs)
        lens.extend(self.kv_args.aux_data_lens)
        for component_ptrs, component_lens in zip(
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_data_lens or [],
        ):
            ptrs.extend(component_ptrs)
            lens.extend(component_lens)
        if ptrs:
            if getattr(self.kv_args, "_npu_mla_bridge_resources", None) is not None:
                rc = self.engine.engine.batch_register_memory(ptrs, lens)
                if rc != 0:
                    raise RuntimeError(
                        f"NPU MLA bridge native registration failed: {rc!r}"
                    )
            else:
                self.engine.batch_register(ptrs, lens)

    def requires_dcp_relayout(self, dst_dcp_size: int, dst_dcp_rank: int) -> bool:
        if self._is_npu_dsa_layout() and self.dcp_size != dst_dcp_size:
            if self.dcp_size == 1 and dst_dcp_size > 1:
                return True
            raise RuntimeError(
                "NPU DSA PD supports prefill DCP 1 -> decode DCP N only, got "
                f"prefill={self.dcp_size}, decode={dst_dcp_size}"
            )
        return super().requires_dcp_relayout(dst_dcp_size, dst_dcp_rank)

    def _get_dcp_remote_decode_layout(self) -> List[bool]:
        layout = self._dcp_remote_decode_layout
        if layout is None or len(layout) != len(self.kv_args.kv_data_ptrs):
            raise RuntimeError(
                "Ascend PD DCP layout does not match its source KV entries"
            )
        return layout

    def prepare_dcp_token_item_lens(
        self, dst_page_item_lens: List[Optional[int]], dst_dcp_size: int
    ) -> List[int]:
        if not self._is_npu_dsa_layout():
            return super().prepare_dcp_token_item_lens(dst_page_item_lens, dst_dcp_size)
        self._get_dcp_remote_decode_layout()
        page_size = self.kv_args.page_size
        token_item_lens = []
        for entry, item_len in enumerate(self.kv_args.kv_item_lens):
            token_item_len, remainder = divmod(item_len, page_size)
            if remainder:
                raise RuntimeError(
                    f"Ascend PD DCP source entry {entry} is not page aligned"
                )
            token_item_lens.append(token_item_len)
        return token_item_lens

    def _init_dcp_pack_buffers_once(
        self, dcp_size: int, *, include_draft: bool = False
    ) -> None:
        # Page-interleaved NPU DCP transfers whole pages directly.
        self._dcp_pack_buffers = []

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int], state_type=None
    ) -> Tuple[List[int], List[int], int]:
        mla_ratios = getattr(self.kv_args, "mla_compression_ratios", None)
        if mla_ratios:
            if len(src_kv_ptrs) == len(dst_kv_ptrs):
                return src_kv_ptrs, dst_kv_ptrs, len(src_kv_ptrs)

            start_layer = self.kv_args.prefill_start_layer
            end_layer = self.kv_args.prefill_end_layer
            c4_full = sum(ratio == 4 for ratio in mla_ratios)
            c4_start = sum(ratio == 4 for ratio in mla_ratios[:start_layer])
            c4_end = sum(ratio == 4 for ratio in mla_ratios[:end_layer])
            c128_start = sum(ratio == 128 for ratio in mla_ratios[:start_layer])
            c128_end = sum(ratio == 128 for ratio in mla_ratios[:end_layer])

            if state_type == AscendStateType.DSV4_C128:
                dst = dst_kv_ptrs[c128_start:c128_end]
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            if state_type == AscendStateType.DSV4_C4_STATE:
                # Layout: [attn_state_0..attn_{c4_full-1},
                #          idx_state_0..idx_{c4_full-1}]
                # Two groups, each c4_full entries; slice both by PP stage.
                dst = []
                for offset in (0, c4_full):
                    dst.extend(dst_kv_ptrs[offset + c4_start : offset + c4_end])
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            # NPU main KV layout: [C4 KV, index K, index scale].
            if state_type is None and len(dst_kv_ptrs) == 3 * c4_full:
                dst = []
                for offset in (0, c4_full, 2 * c4_full):
                    dst.extend(dst_kv_ptrs[offset + c4_start : offset + c4_end])
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            # On A5 (CYCLE cache_mode), StateType.SWA only contains SWA KV
            # buffers (C4 compress state is registered separately as
            # DSV4_C4_STATE).  The common _mla_slice_ptrs_for_pp assumes
            # SWA + C4 state are bundled (swa_L + 2*c4_full), so intercept
            # here and slice SWA KV by layer index directly.
            if state_type == StateType.SWA and AscendStateType.DSV4_C4_STATE in (
                self.kv_args.state_types or []
            ):
                dst = list(dst_kv_ptrs[start_layer:end_layer])
                return src_kv_ptrs, dst, len(src_kv_ptrs)

            return super().get_mla_kv_ptrs_with_pp(src_kv_ptrs, dst_kv_ptrs, state_type)

        # src_kv_ptrs: k_data, v_data, index_k_data(optional)
        # dst_kv_ptrs: k_data, v_data, index_k_data(optional)
        # state_type is accepted for parity with the common disaggregation path;
        # the NPU kv_buf_groups slicing below is state-type agnostic.
        kv_buf_groups = getattr(self.kv_args, "kv_buf_groups", 1)
        hidden_kv_layers = getattr(self.kv_args, "hidden_kv_layers", 0)
        draft_kv_layers = getattr(self.kv_args, "draft_kv_layers", 0)
        src_layers = len(src_kv_ptrs) // kv_buf_groups
        dst_layers = len(dst_kv_ptrs) // kv_buf_groups
        if src_layers == dst_layers:
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            sliced_dst_kv_ptrs = []
            start_layer = self.kv_args.prefill_start_layer
            transfer_draft_kv = get_parallel().pp_group.is_last_rank and draft_kv_layers
            if transfer_draft_kv:
                end_layer = start_layer + src_layers - draft_kv_layers
            else:
                end_layer = start_layer + src_layers

            # target kv
            for i in range(kv_buf_groups):
                layer_offset = i * hidden_kv_layers
                sliced_dst_kv_ptrs.extend(
                    dst_kv_ptrs[layer_offset + start_layer : layer_offset + end_layer]
                )
            # draft kv
            if transfer_draft_kv:
                for i in range(kv_buf_groups):
                    layer_offset = (
                        i * draft_kv_layers + kv_buf_groups * hidden_kv_layers
                    )
                    sliced_dst_kv_ptrs.extend(
                        dst_kv_ptrs[layer_offset : layer_offset + draft_kv_layers]
                    )
        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: Optional[List[int]] = None,
        dst_device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
        dst_kv_item_len: Optional[int] = None,
        dst_attn_tp_size: Optional[int] = None,
    ):
        if dst_device_kv_indices is not None:
            raise NotImplementedError(
                "Ascend PD transfer does not support HiSparse "
                "destination device KV indices"
            )
        self._validate_envelope_kv_layout(
            dst_kv_ptrs, dst_kv_item_len, dst_attn_tp_size
        )
        # Hybrid MLA prefill stages expose PP-local entries, while a PP=1
        # decode peer registers all model layers. Pair only this layout by
        # global layer id; every other Ascend layout keeps the legacy path.
        if self.is_hybrid_mla_backend and self.pp_size > 1:
            return self._send_kvcache_generic(
                mooncake_session_id=mooncake_session_id,
                src_data_ptrs=self.kv_args.kv_data_ptrs,
                dst_data_ptrs=dst_kv_ptrs,
                item_lens=self.kv_args.kv_item_lens,
                prefill_data_indices=prefill_kv_indices,
                dst_data_indices=dst_kv_indices,
                executor=executor,
                src_layer_ids=self.kv_args.kv_layer_ids,
                dst_layer_ids=dst_layer_ids,
            )

        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        if self.pp_size > 1:
            if self.is_mla_backend:
                src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage = (
                    self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                )
                layers_params = [
                    (
                        src_kv_ptrs[layer_id],
                        sliced_dst_kv_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
            else:
                (
                    src_k_ptrs,
                    src_v_ptrs,
                    dst_k_ptrs,
                    dst_v_ptrs,
                    layers_current_pp_stage,
                ) = self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)

                layers_params = [
                    (
                        src_k_ptrs[layer_id],
                        dst_k_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ] + [
                    (
                        src_v_ptrs[layer_id],
                        dst_v_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layers_current_pp_stage + layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
        else:
            num_layers = len(self.kv_args.kv_data_ptrs)
            layers_params = [
                (
                    self.kv_args.kv_data_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(num_layers)
            ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0

    def send_kvcache_dcp(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        *,
        dcp_token_item_lens: List[int],
        dst_dcp_size: int,
        dst_dcp_rank: int,
        src_page_offset: int,
        decode_prefix_len: int,
        num_kv_tokens: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: List[int],
        pack_buffer=None,
        dst_kv_item_lens: Optional[List[int]] = None,
        dst_tp_rank: int = 0,
        dst_attn_tp_size: Optional[int] = None,
    ) -> int:
        if not self._is_npu_dsa_layout():
            return super().send_kvcache_dcp(
                mooncake_session_id,
                prefill_kv_indices,
                dst_kv_ptrs,
                dst_kv_indices,
                dcp_token_item_lens=dcp_token_item_lens,
                dst_dcp_size=dst_dcp_size,
                dst_dcp_rank=dst_dcp_rank,
                src_page_offset=src_page_offset,
                decode_prefix_len=decode_prefix_len,
                num_kv_tokens=num_kv_tokens,
                executor=executor,
                dst_layer_ids=dst_layer_ids,
                pack_buffer=pack_buffer,
                dst_kv_item_lens=dst_kv_item_lens,
                dst_tp_rank=dst_tp_rank,
                dst_attn_tp_size=dst_attn_tp_size,
            )

        src_kv_ptrs = self.kv_args.kv_data_ptrs
        _, dst_kv_ptrs, _ = self.get_mla_kv_ptrs_with_pp(src_kv_ptrs, dst_kv_ptrs)
        if dst_kv_item_lens:
            _, dst_kv_item_lens, _ = self.get_mla_kv_ptrs_with_pp(
                self.kv_args.kv_item_lens, dst_kv_item_lens
            )

        layout = self._get_dcp_remote_decode_layout()
        num_entries = len(src_kv_ptrs)
        if not (
            len(dst_kv_ptrs)
            == len(self.kv_args.kv_item_lens)
            == len(dcp_token_item_lens)
            == len(layout)
            == num_entries
        ):
            raise RuntimeError("Ascend PD DCP KV entry metadata is inconsistent")
        if dst_kv_item_lens and len(dst_kv_item_lens) != num_entries:
            raise RuntimeError("Ascend PD DCP destination KV metadata is inconsistent")

        local_src, local_dst, global_src, global_dst = _build_page_interleaved_dcp_plan(
            prefill_kv_indices,
            dst_kv_indices,
            page_size=self.kv_args.page_size,
            dcp_size=dst_dcp_size,
            dcp_rank=dst_dcp_rank,
            src_page_offset=src_page_offset,
            decode_prefix_len=decode_prefix_len,
            num_kv_tokens=num_kv_tokens,
        )

        transfer_blocks: List[Tuple[int, int, int]] = []
        for entry, uses_global_slots in enumerate(layout):
            page_bytes = self.kv_args.kv_item_lens[entry]
            if page_bytes != dcp_token_item_lens[entry] * self.kv_args.page_size:
                raise RuntimeError(
                    f"Ascend PD DCP source geometry differs at entry {entry}"
                )
            if dst_kv_item_lens:
                expected_dst_bytes = page_bytes * (
                    dst_dcp_size if uses_global_slots else 1
                )
                if dst_kv_item_lens[entry] != expected_dst_bytes:
                    raise RuntimeError(
                        "Ascend PD DCP destination geometry differs at entry "
                        f"{entry}: expected={expected_dst_bytes}, "
                        f"actual={dst_kv_item_lens[entry]}"
                    )

            src_pages = global_src if uses_global_slots else local_src
            dst_pages = global_dst if uses_global_slots else local_dst
            src_groups, dst_groups = group_concurrent_contiguous(src_pages, dst_pages)
            for src_group, dst_group in zip(src_groups, dst_groups):
                transfer_blocks.append(
                    (
                        src_kv_ptrs[entry] + int(src_group[0]) * page_bytes,
                        dst_kv_ptrs[entry] + int(dst_group[0]) * page_bytes,
                        len(src_group) * page_bytes,
                    )
                )

        return self._transfer_data(mooncake_session_id, transfer_blocks)

    def _is_generic_kvcache_state_type(self, st) -> bool:
        # DSV4 per-pool components also use the page-indexed send path.
        return (
            super()._is_generic_kvcache_state_type(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    def requires_strict_transfer_drain(self):
        return getattr(self.kv_mgr, "npu_mla_bridge_metadata", None) is not None

    def begin_strict_transfer_drain(self):
        coverage = getattr(self, "_npu_mla_bridge_coverage", None)
        resources = getattr(self.kv_mgr, "_sparse_decode_bridge_resources", None)
        if coverage is not None and resources is not None:
            resources.quarantine_room(
                coverage.room, coverage.generation, coverage.session
            )

    def is_strict_transfer_drain_safe(self):
        # No all-rank writer drain protocol has been proven yet.
        return False

    def filter_transfer_status(self, status):
        if not self.requires_strict_transfer_drain() or status != KVPoll.Success:
            return status
        coverage = getattr(self, "_npu_mla_bridge_coverage", None)
        resources = getattr(self.kv_mgr, "_sparse_decode_bridge_resources", None)
        if coverage is None or resources is None:
            return KVPoll.Transferring
        coverage.complete_terminal(
            room=coverage.room, generation=coverage.generation, session=coverage.session
        )
        if not coverage.success():
            return KVPoll.Transferring
        resources.ring.seal_room(coverage.room, coverage.generation, coverage.session)
        if not resources.ring.room_success(
            coverage.room, coverage.generation, coverage.session
        ):
            return KVPoll.Transferring
        return status

    def finish_npu_mla_transfer(self):
        """Retire a locally complete room only after the scheduler's all-rank poll."""
        if not self.requires_strict_transfer_drain():
            return
        coverage = self._npu_mla_bridge_coverage
        self.kv_mgr._sparse_decode_bridge_resources.close_room(
            coverage.room, globally_drained=True
        )
        self._npu_mla_bridge_coverage = None
        self._npu_mla_bridge_request = None

    def _npu_mla_room_frame(self):
        if getattr(self.kv_mgr, "npu_mla_bridge_metadata", None) is None:
            return None
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
            BridgeRoomDescriptor,
        )

        coverage = getattr(self, "_npu_mla_bridge_coverage", None)
        if coverage is None:
            raise RuntimeError("Bridge room was not registered before publication")
        return BridgeRoomDescriptor(
            generation=coverage.generation,
            num_tokens=coverage.num_tokens,
            prefix_tokens=coverage.prefix_tokens,
        ).encode()

    def prepare_npu_mla_bridge_room(
        self, decode_req, page_indices, num_tokens, prefix_tokens, draft_pool
    ):
        resources = getattr(self.kv_mgr, "_sparse_decode_bridge_resources", None)
        if (
            resources is None
            or getattr(self.kv_mgr, "npu_mla_bridge_metadata", None) is None
        ):
            raise RuntimeError("NPU MLA bridge resources are not negotiated")
        coverage = resources.register_room(
            room=self.bootstrap_room,
            session=self.session_id,
            pages=page_indices,
            num_tokens=num_tokens,
            prefix_tokens=prefix_tokens,
            draft_pool=draft_pool,
            peers=tuple(
                (info["rank_ip"], int(info["rank_port"]))
                for info in self.bootstrap_infos
                if not info["is_dummy"]
            ),
        )
        self._npu_mla_bridge_request = decode_req
        self._npu_mla_bridge_coverage = coverage


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
