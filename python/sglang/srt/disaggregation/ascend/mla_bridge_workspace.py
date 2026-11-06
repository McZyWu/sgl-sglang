"""Single-lease packed sender workspace for an experimental MLA PD bridge."""

from threading import Lock

import torch

from sglang.srt.disaggregation.ascend.mla_bridge_buffer import RegisteredBridgeBuffer
from sglang.srt.disaggregation.ascend.mla_bridge_contract import ALIGN, ROW_BYTES
from sglang.srt.disaggregation.ascend.mla_bridge_ops import pack_mla_rows


class PackedBridgeWorkspace:
    """One aligned transfer slot, shared by layers and protected by one lease.

    The caller retains this object before register, controls stream ordering,
    and may release a lease only when the remote write has drained.
    """

    def __init__(self, device):
        self.transfer = RegisteredBridgeBuffer(usable_bytes=ALIGN, device=device)
        self.capacity = ALIGN // ROW_BYTES
        self.k_workspace = torch.empty(
            (self.capacity, 512), dtype=torch.bfloat16, device=device
        )
        self.v_workspace = torch.empty(
            (self.capacity, 64), dtype=torch.bfloat16, device=device
        )
        self._lock = Lock()
        self._lease = None
        self._next_lease = 0
        self._packed_rows = None
        self._send_attempted = False
        self._inputs = None

    def try_acquire(self):
        with self._lock:
            if self.transfer.raw is None:
                raise RuntimeError("Packed workspace is closed")
            if self._lease is not None:
                return None
            self._next_lease += 1
            self._lease = self._next_lease
            self._packed_rows = None
            self._send_attempted = False
            return self._lease

    def pack(self, lease, k, v, indices):
        with self._lock:
            if self._lease != lease or lease is None:
                raise ValueError("Stale packed workspace lease")
            if self._packed_rows is not None or self._send_attempted:
                raise RuntimeError("A workspace lease can pack only one chunk")
            n = indices.numel()
            if not 0 < n <= self.capacity:
                raise ValueError("Packed workspace chunk is too large or empty")
            packed = (
                self.transfer.buffer[: n * ROW_BYTES].view(torch.bfloat16).view(n, 576)
            )
            self._packed_rows = -1
            self._inputs = (k, v, indices)
            pack_mla_rows(
                k, v, indices, packed, self.k_workspace[:n], self.v_workspace[:n]
            )
            self._packed_rows = n
            return packed

    def pack_native_planes(self, lease, ckv_plane, kpe_plane, indices):
        """Gather the actual page-major pool planes into the shared slot."""
        from sglang.srt.disaggregation.ascend.mla_bridge_collect import (
            flatten_main_mla_plane,
        )

        ckv, kpe = ckv_plane.entry, kpe_plane.entry
        if (
            ckv.component != "ckv"
            or kpe.component != "kpe"
            or ckv.layer != kpe.layer
            or ckv.capacity != kpe.capacity
        ):
            raise ValueError("Mismatched native MLA plane pair")
        return self.pack(
            lease,
            flatten_main_mla_plane(ckv_plane),
            flatten_main_mla_plane(kpe_plane),
            indices,
        )

    def release(self, lease, *, transport_drained: bool, device_work_drained: bool):
        with self._lock:
            if self._lease != lease or lease is None:
                raise ValueError("Stale packed workspace lease")
            if not transport_drained or not device_work_drained:
                raise RuntimeError("Cannot reuse a workspace while work may be active")
            self._lease = None
            self._inputs = None

    def send(
        self,
        lease,
        engine,
        session,
        *,
        rows,
        dst_base,
        dst_size,
        dst_offset,
        pack_event,
    ):
        """Wait for pack then synchronously send one bounded contiguous chunk."""
        with self._lock:
            if self._lease != lease or lease is None:
                raise ValueError("Stale packed workspace lease")
            if not self.transfer.registered or not session:
                raise RuntimeError("Packed workspace is not registered")
            if self._packed_rows != rows or self._send_attempted:
                raise RuntimeError("Packed chunk is missing or already sent")
            if any(type(x) is not int for x in (rows, dst_base, dst_size, dst_offset)):
                raise ValueError("Invalid packed transfer descriptor")
            length = rows * ROW_BYTES
            if (
                not 0 < rows <= self.capacity
                or dst_base <= 0
                or dst_base % ALIGN
                or dst_size != 2 * ALIGN
                or dst_offset not in (0, ALIGN)
                or dst_offset + length > dst_size
            ):
                raise ValueError("Packed transfer exceeds its advertised destination")
            pack_event.synchronize()
            self._send_attempted = True
            rc = engine.batch_transfer_sync_write(
                session, [self.transfer.pointer], [dst_base + dst_offset], [length]
            )
            if rc != 0:
                raise RuntimeError(f"NPU bridge transfer failed: {rc!r}")

    def close(self, engine, *, transport_drained: bool, synchronize):
        with self._lock:
            if self._lease is not None:
                raise RuntimeError("Cannot close a leased workspace")
            self.transfer.close(
                engine, transport_drained=transport_drained, synchronize=synchronize
            )
            self.k_workspace = None
            self.v_workspace = None
