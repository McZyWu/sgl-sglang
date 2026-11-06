"""Owned sparse decode bridge resources, not yet wired to transport."""

from sglang.srt.disaggregation.ascend.mla_bridge_collect import (
    collect_native_prefill_planes,
    collect_sparse_decode_entries,
)
from sglang.srt.disaggregation.ascend.mla_bridge_coverage import BridgeRoomCoverage
from sglang.srt.disaggregation.ascend.mla_bridge_receiver import BridgeReceiverRing
from threading import Lock


class SparseDecodeBridgeResources:
    """Retain native index, host mappings and the bounded device receive ring."""

    def __init__(self, pool, geometry, *, device):
        self.pool = pool
        self.geometry = geometry
        self.entries = collect_sparse_decode_entries(pool, geometry)
        self._draft_planes = None
        self.manager = pool.sparse_kv_manager
        self.ring = BridgeReceiverRing(device)
        self._rooms = {}
        self._room_peers = {}
        self._lease_rows = {}
        self._generation = 0
        self._lock = Lock()
        self._closed = False

    def register_room(
        self, *, room, session, pages, num_tokens, prefix_tokens, draft_pool, peers=()
    ):
        """Capture decode-owned pages before publishing them to prefill."""
        layers = [
            entry.layer for entry in self.entries.entries if entry.kind == "staged"
        ]
        native = [
            (entry.model, entry.layer, entry.component)
            for entry in self.entries.entries
            if entry.kind == "native"
        ]
        native.extend(
            (plane.entry.model, plane.entry.layer, plane.entry.component)
            for plane in collect_native_prefill_planes(None, draft_pool)
        )
        native.extend([("aux",), ("state",)])
        peers = tuple(peers)
        if len(set(peers)) != len(peers) or any(
            not isinstance(host, str)
            or not host
            or type(port) is not int
            or not 0 < port < 65536
            for host, port in peers
        ):
            raise ValueError("Invalid bridge room peer list")
        capacity = min(
            entry.capacity for entry in self.entries.entries if entry.kind == "staged"
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("Bridge resources are closed")
            if room in self._rooms:
                raise ValueError("Bridge room is already registered")
            generation = self._generation + 1
            coverage = BridgeRoomCoverage(
                room=room,
                generation=generation,
                session=session,
                pages=pages,
                num_tokens=num_tokens,
                prefix_tokens=prefix_tokens,
                physical_capacity=capacity,
                layers=layers,
                required_native=native,
            )
            self.ring.open_room(room, generation, session, None)
            self._generation = generation
            self._rooms[room] = coverage
            self._room_peers[room] = frozenset(peers)
            return coverage

    def bind_native_manifest(self, draft_pool, kv_args):
        """Retain draft views and validate the final native registration order."""
        draft_planes = collect_native_prefill_planes(None, draft_pool)
        pointers = self.entries.native_ptrs + tuple(
            p.tensor.data_ptr() for p in draft_planes
        )
        sizes = self.entries.native_sizes + tuple(p.tensor.nbytes for p in draft_planes)
        strides = self.entries.native_strides + tuple(
            p.entry.item_stride for p in draft_planes
        )
        if (
            pointers != tuple(kv_args.kv_data_ptrs)
            or sizes != tuple(kv_args.kv_data_lens)
            or strides != tuple(kv_args.kv_item_lens)
        ):
            raise ValueError("Decode native registration does not match its manifest")
        with self._lock:
            if self._draft_planes is not None:
                raise RuntimeError("Decode native manifest is already bound")
            self._draft_planes = draft_planes
            self.native_bytes = sum(sizes)
            self.host_bytes = sum(tensor.nbytes for tensor in self.entries.host_tensors)

    def candidate_metadata(self):
        """Build metadata only after owned ring registration; do not advertise it."""
        from sglang.srt.disaggregation.ascend.mla_bridge_contract import BridgeMetadata

        with self._lock:
            if self._draft_planes is None or not self.ring.transfer.registered:
                raise RuntimeError("Decode ring and native manifest are not ready")
            entries = self.entries.entries + tuple(p.entry for p in self._draft_planes)
            return BridgeMetadata(
                self.geometry,
                entries,
                self.ring.transfer.pointer,
                self.ring.transfer.size,
            ).encode()

    def host_for_layer(self, layer):
        """Return a zero-copy combined row view and its matching mapped DVA."""
        staged = [entry for entry in self.entries.entries if entry.kind == "staged"]
        matches = [i for i, entry in enumerate(staged) if entry.layer == layer]
        if len(matches) != 1:
            raise ValueError("Unknown or ambiguous target layer")
        i = matches[0]
        host = self.entries.host_tensors[i]
        if not host.is_contiguous() or host.numel() != staged[i].capacity * 576:
            raise ValueError("Sparse host storage no longer matches its manifest")
        rows = host.view(-1, 576)
        return rows, self.entries.host_device_ptrs[i]

    def complete_native_range(self, *, room, generation, session, start, count):
        with self._lock:
            self._rooms[room].complete_native_range(
                room=room,
                generation=generation,
                session=session,
                start=start,
                count=count,
            )

    def require_peer(self, room, generation, session, peer):
        with self._lock:
            coverage = self._rooms[room]
            if (coverage.generation, coverage.session) != (
                generation,
                session,
            ) or peer not in self._room_peers[room]:
                raise ValueError("Stale or unauthorized bridge peer")

    def quarantine_room(self, room, generation, session):
        with self._lock:
            coverage = self._rooms[room]
            if (coverage.generation, coverage.session) != (generation, session):
                raise ValueError("Stale bridge room")
            coverage.abort()
            self.ring.abort_room(room, generation, session)

    def reserve_chunk(self, *, room, generation, session, layer, start, count):
        """Lease a bounded slot and derive destination rows from the owned room."""
        with self._lock:
            coverage = self._rooms[room]
            if (generation, session) != (coverage.generation, coverage.session):
                raise ValueError("Stale bridge room")
            item = (layer, start, count)
            lease = self.ring.acquire(room, generation, session, item)
            if lease is None:
                return None
            slot, lease_id = lease
            try:
                rows = coverage.reserve(
                    room=room,
                    generation=generation,
                    session=session,
                    layer=layer,
                    start=start,
                    count=count,
                    lease_id=lease_id,
                )
            except Exception:
                # Once a lease exists, quarantine on an inconsistent request;
                # do not infer whether a peer could have observed the slot.
                coverage.abort()
                self.ring.abort_room(room, generation, session)
                raise
            self._lease_rows[(room, generation, session, slot, lease_id)] = (
                layer,
                start,
                count,
                rows,
            )
            return slot, lease_id, rows

    def reserved_rows(
        self, *, room, generation, session, layer, start, count, slot, lease_id
    ):
        with self._lock:
            item = self._lease_rows.get((room, generation, session, slot, lease_id))
            if item is None or item[:3] != (layer, start, count):
                raise ValueError("Stale or mismatched bridge lease")
            return item[3]

    def finish_chunk(self, *, room, generation, session, layer, slot, lease_id):
        """Commit coverage and permit ACK only after the scatter event completes."""
        with self._lock:
            coverage = self._rooms[room]
            if (generation, session) != (coverage.generation, coverage.session):
                raise ValueError("Stale bridge room")
            key = (room, generation, session, slot, lease_id)
            item = self._lease_rows.get(key)
            if item is None or item[0] != layer:
                raise ValueError("Stale or mismatched bridge lease")
            if not self.ring.finish_scatter(slot, room, generation, session, lease_id):
                return False
            try:
                coverage.complete_scatter(
                    room=room,
                    generation=generation,
                    session=session,
                    layer=layer,
                    lease_id=lease_id,
                )
            except Exception:
                coverage.abort()
                self.ring.abort_room(room, generation, session)
                raise
            del self._lease_rows[key]
            return True

    def close_room(self, room, *, globally_drained):
        """Close only after a real all-rank writer and scatter drain proof."""
        with self._lock:
            coverage = self._rooms[room]
            if not globally_drained:
                raise RuntimeError("Bridge room is not globally drained")
            if any(key[0] == room for key in self._lease_rows):
                raise RuntimeError("Bridge room still has reserved rows")
            if coverage.success():
                self.ring.seal_room(room, coverage.generation, coverage.session)
                if not self.ring.room_success(
                    room, coverage.generation, coverage.session
                ):
                    raise RuntimeError("Bridge ring has incomplete leases")
            else:
                coverage.abort()
                self.ring.abort_room(room, coverage.generation, coverage.session)
            self.ring.close_room(room, coverage.generation, coverage.session)
            del self._rooms[room]
            del self._room_peers[room]

    def register(self, engine):
        self.ring.transfer.register(engine)

    def close(self, engine, *, transport_drained, synchronize):
        with self._lock:
            if self._rooms:
                raise RuntimeError(
                    "Cannot release bridge resources with registered rooms"
                )
            self.ring.close(
                engine, transport_drained=transport_drained, synchronize=synchronize
            )
            self._closed = True
