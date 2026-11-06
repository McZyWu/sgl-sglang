"""Unwired NPU receiver ring and scatter-event ownership for MLA PD."""

from threading import Lock

import torch

from sglang.srt.disaggregation.ascend.mla_bridge_buffer import RegisteredBridgeBuffer
from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
    ALIGN,
    ROW_BYTES,
    BridgeLeases,
)
from sglang.srt.disaggregation.ascend.mla_bridge_ops import scatter_mla_rows


class BridgeReceiverRing:
    """Two owned slots; a lease remains live through scatter event completion.

    The transport must establish that all writes to a lease have drained before
    calling write_drained. The scheduler must retain request token allocations
    until all ranks are known drained; this class does not establish that proof.
    """

    def __init__(self, device):
        self.transfer = RegisteredBridgeBuffer(usable_bytes=2 * ALIGN, device=device)
        self._leases = BridgeLeases(2)
        self._events = {}
        self._submitted = set()
        self._submitted_inputs = {}
        self._lock = Lock()

    def open_room(self, room, generation, session, required):
        with self._lock:
            if self.transfer.raw is None:
                raise RuntimeError("Receiver ring is closed")
            self._leases.open(room, generation, session, required)

    def acquire(self, room, generation, session, item):
        with self._lock:
            if self.transfer.raw is None:
                raise RuntimeError("Receiver ring is closed")
            return self._leases.acquire(room, generation, session, item)

    def write_drained(self, slot, room, generation, session, lease_id):
        with self._lock:
            self._leases.write_drained(slot, room, generation, session, lease_id)

    def abort_room(self, room, generation, session):
        with self._lock:
            self._leases.abort(room, generation, session)

    def release_aborted(self, slot, room, generation, session, lease_id):
        with self._lock:
            self._leases.release_aborted(slot, room, generation, session, lease_id)

    def cancel_aborted_unsubmitted(self, slot, room, generation, session, lease_id):
        key = (slot, room, generation, session, lease_id)
        with self._lock:
            if key in self._submitted:
                raise RuntimeError("Scatter was already attempted")
            self._leases.cancel_aborted_unsubmitted(*key)

    def room_success(self, room, generation, session):
        with self._lock:
            return self._leases.success(room, generation, session)

    def seal_room(self, room, generation, session):
        with self._lock:
            self._leases.seal(room, generation, session)

    def close_room(self, room, generation, session):
        with self._lock:
            self._leases.close(room, generation, session)

    def submit_scatter(
        self,
        slot,
        room,
        generation,
        session,
        lease_id,
        *,
        rows,
        host,
        src_indices,
        dst_indices,
        valid,
        host_dev_ptr,
        event_factory,
        scatter=scatter_mla_rows,
    ):
        key = (slot, room, generation, session, lease_id)
        with self._lock:
            self._leases.require_scatter(*key)
            if key in self._submitted:
                raise ValueError("Scatter already submitted")
            if type(rows) is not int or not 0 < rows <= ALIGN // ROW_BYTES:
                raise ValueError("Invalid scatter row count")
            if any(x.numel() != rows for x in (src_indices, dst_indices, valid)):
                raise ValueError("Scatter index lengths do not match the chunk")
            packed = (
                self.transfer.buffer[slot * ALIGN : slot * ALIGN + rows * ROW_BYTES]
                .view(torch.bfloat16)
                .view(rows, 576)
            )
            # If enqueue or event recording fails, the lease remains live and
            # must be quarantined until the caller can prove stream drain.
            self._submitted.add(key)
            self._submitted_inputs[key] = (host, src_indices, dst_indices, valid)
            scatter(
                packed, host, src_indices, dst_indices, valid, host_dev_ptr=host_dev_ptr
            )
            event = event_factory()
            event.record()
            self._events[key] = event

    def finish_scatter(self, slot, room, generation, session, lease_id):
        """Return True only when event completion permits release and ACK."""
        key = (slot, room, generation, session, lease_id)
        with self._lock:
            event = self._events.get(key)
            if event is None:
                raise ValueError("Scatter event is missing")
            if not event.query():
                return False
            self._leases.scatter_drained(*key)
            self._leases.release(*key)
            del self._events[key]
            self._submitted.remove(key)
            del self._submitted_inputs[key]
            return True

    def close(self, engine, *, transport_drained: bool, synchronize):
        with self._lock:
            if (
                self._events
                or self._submitted_inputs
                or self._leases.has_live_leases()
                or self._leases.has_open_rooms()
            ):
                raise RuntimeError(
                    "Cannot close a receiver ring with live rooms or leases"
                )
            self.transfer.close(
                engine, transport_drained=transport_drained, synchronize=synchronize
            )
