"""Pure contracts for the proposed Ascend MLA PD staging bridge.

Nothing here enables the bridge. Transport and ownership integration must be
completed before these contracts can be used by a running server.
"""

import json
import operator
from dataclasses import asdict, dataclass
from enum import Enum, auto
from threading import Lock
from typing import Hashable

ALIGN = 2 * 1024 * 1024
ROW_BYTES = (512 + 64) * 2
LAYOUT = "ascend-mla-split-to-combined-v1"


@dataclass(frozen=True)
class BridgeRoomDescriptor:
    """Per-room decode allocation identity, sent only after capability agreement."""

    generation: int
    num_tokens: int
    prefix_tokens: int

    def encode(self) -> bytes:
        values = (self.generation, self.num_tokens, self.prefix_tokens)
        if any(type(value) is not int for value in values):
            raise ValueError("Invalid room descriptor types")
        if self.generation <= 0 or self.num_tokens <= 0 or self.prefix_tokens < 0:
            raise ValueError("Invalid room descriptor values")
        if self.prefix_tokens % 128:
            raise ValueError("Bridge prefix must be page aligned")
        return json.dumps(
            {
                "version": 1,
                "generation": self.generation,
                "num_tokens": self.num_tokens,
                "prefix_tokens": self.prefix_tokens,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")

    @classmethod
    def decode(cls, raw: bytes) -> "BridgeRoomDescriptor":
        if not isinstance(raw, bytes) or not 0 < len(raw) <= 256:
            raise ValueError("Invalid room descriptor size")
        try:
            value = json.loads(raw)
            if (
                not isinstance(value, dict)
                or set(value)
                != {"version", "generation", "num_tokens", "prefix_tokens"}
                or type(value["version"]) is not int
                or value["version"] != 1
            ):
                raise ValueError("Unsupported room descriptor")
            result = cls(
                value["generation"], value["num_tokens"], value["prefix_tokens"]
            )
            result.encode()
            return result
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError("Invalid room descriptor") from exc


@dataclass(frozen=True)
class BridgeGeometry:
    layout: str
    tp: int
    pp: int
    dcp: int
    page_size: int
    dtype: str
    ckv: int
    kpe: int

    def validate_peer(self, peer: "BridgeGeometry") -> None:
        expected = BridgeGeometry(LAYOUT, 16, 1, 1, 128, "bfloat16", 512, 64)
        if (
            self != expected
            or peer != expected
            or any(
                type(getattr(g, field)) is not int
                for g in (self, peer)
                for field in ("tp", "pp", "dcp", "page_size", "ckv", "kpe")
            )
        ):
            raise ValueError("Unsupported or mismatched Ascend MLA bridge geometry")


@dataclass(frozen=True)
class BridgeBudget:
    slot_bytes: int = ALIGN
    slot_count: int = 2

    # Both receive ring and prefill scratch need an owned alignment margin.
    @property
    def ring_allocation_bytes(self) -> int:
        return self.slot_bytes * self.slot_count + ALIGN

    @property
    def scratch_allocation_bytes(self) -> int:
        return self.slot_bytes + ALIGN

    @property
    def tokens_per_chunk(self) -> int:
        return self.slot_bytes // ROW_BYTES

    @property
    def gather_workspace_bytes(self) -> int:
        return self.tokens_per_chunk * ROW_BYTES

    @property
    def prefill_index_bytes(self) -> int:
        return self.tokens_per_chunk * 8

    @property
    def decode_index_mask_bytes(self) -> int:
        return self.tokens_per_chunk * (8 + 8 + 1)

    @property
    def pack_transient_reserve_bytes(self) -> int:
        # This is a conservative admission reserve, not a proven upper bound
        # on allocations made inside the device runtime.
        return ALIGN

    def validate(self) -> None:
        if self.slot_count != 2 or self.slot_bytes != ALIGN:
            raise ValueError("Unsupported Ascend MLA bridge budget")


@dataclass(frozen=True)
class BridgeEntry:
    """Explicit identity and geometry for a native or staged buffer."""

    kind: str
    model: str
    layer: int
    component: str
    dtype: str
    item_stride: int
    capacity: int
    addressing_unit: str
    page_size: int

    def validate(self) -> None:
        if self.kind not in ("staged", "native") or self.model not in ("main", "draft"):
            raise ValueError("Invalid bridge entry kind or model")
        if (
            any(
                type(x) is not int
                for x in (self.layer, self.item_stride, self.capacity)
            )
            or self.layer < 0
            or self.item_stride <= 0
            or self.capacity <= 0
            or not isinstance(self.dtype, str)
            or not self.dtype
        ):
            raise ValueError("Invalid bridge entry geometry")
        if self.page_size != 128 or type(self.page_size) is not int:
            raise ValueError("Unsupported bridge page size")
        expected_unit = "token" if self.kind == "staged" else "page"
        if self.addressing_unit != expected_unit:
            raise ValueError("Bridge addressing unit does not match buffer kind")
        if self.kind == "staged":
            if (self.model, self.component, self.dtype, self.item_stride) != (
                "main",
                "combined",
                "bfloat16",
                ROW_BYTES,
            ):
                raise ValueError("Only main combined MLA rows may be staged")
        elif self.component not in ("ckv", "kpe", "index", "index_scale"):
            raise ValueError("Unsupported native component")


def validate_bridge_entries(entries):
    """Reject ambiguous identities before any native or staged transfer."""
    entries = tuple(entries)
    identities = set()
    for entry in entries:
        entry.validate()
        identity = (entry.kind, entry.model, entry.layer, entry.component)
        if identity in identities:
            raise ValueError("Duplicate bridge entry")
        identities.add(identity)
    if not any(x.kind == "staged" for x in entries):
        raise ValueError("Bridge has no staged main MLA entries")
    return entries


def validate_bridge_manifest(entries, required_identities):
    """Match an authoritative complete identity set, including index and draft."""
    entries = validate_bridge_entries(entries)
    required = tuple(required_identities)
    if not required or len(set(required)) != len(required):
        raise ValueError("Invalid required bridge identities")
    actual = {(x.kind, x.model, x.layer, x.component) for x in entries}
    if actual != set(required):
        raise ValueError("Bridge manifest does not cover the required components")
    return entries


def pair_bridge_entries(source, destination):
    """Pair every source plane with its destination; reject omissions/extras.

    Staged main combined rows map to two source planes. Native index, scale,
    and draft entries retain their model, layer and component identities.
    Capacities may differ across peers; each peer validates its physical IDs.
    """
    source = tuple(source)
    destination = validate_bridge_entries(destination)
    source_by_id = {}
    for entry in source:
        entry.validate()
        if entry.kind != "native":
            raise ValueError("Prefill entries must be native")
        key = (entry.model, entry.layer, entry.component)
        if key in source_by_id:
            raise ValueError("Duplicate source bridge entry")
        source_by_id[key] = entry
    used = set()
    pairs = []
    for dst in destination:
        if dst.kind == "staged":
            keys = (("main", dst.layer, "ckv"), ("main", dst.layer, "kpe"))
            expected_strides = (128 * 1024, 128 * 128)
        else:
            if dst.model == "main" and dst.component in ("ckv", "kpe"):
                raise ValueError("Main KV must use the staged destination")
            keys = ((dst.model, dst.layer, dst.component),)
            expected_strides = (dst.item_stride,)
        for key, stride in zip(keys, expected_strides):
            src = source_by_id.get(key)
            if (
                src is None
                or key in used
                or src.item_stride != stride
                or src.dtype != dst.dtype
                or src.page_size != dst.page_size
            ):
                raise ValueError("Missing or incompatible bridge component")
            used.add(key)
            pairs.append((src, dst))
    if used != set(source_by_id):
        raise ValueError("Unpaired source bridge component")
    return tuple(pairs)


@dataclass(frozen=True)
class BridgeMetadata:
    geometry: BridgeGeometry
    entries: tuple[BridgeEntry, ...]
    ring_pointer: int
    ring_bytes: int

    def encode(self) -> bytes:
        self.geometry.validate_peer(self.geometry)
        if (
            type(self.ring_pointer) is not int
            or self.ring_pointer <= 0
            or self.ring_pointer % ALIGN
            or type(self.ring_bytes) is not int
            or self.ring_bytes != 2 * ALIGN
        ):
            raise ValueError("Invalid registered bridge ring")
        validate_bridge_entries(self.entries)
        data = {
            "geometry": asdict(self.geometry),
            "entries": [asdict(x) for x in self.entries],
            "ring_pointer": self.ring_pointer,
            "ring_bytes": self.ring_bytes,
        }
        payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(payload) > 65536:
            raise ValueError("Bridge metadata exceeds size limit")
        return payload

    @classmethod
    def decode(cls, payload: bytes) -> "BridgeMetadata":
        if not payload or len(payload) > 65536:
            raise ValueError("Invalid bridge metadata length")
        try:
            data = json.loads(payload)
            if not isinstance(data, dict) or set(data) != {
                "geometry",
                "entries",
                "ring_pointer",
                "ring_bytes",
            }:
                raise ValueError("Invalid bridge metadata fields")
            if not isinstance(data["entries"], list):
                raise ValueError("Invalid bridge entry list")
            metadata = cls(
                geometry=BridgeGeometry(**data["geometry"]),
                entries=tuple(BridgeEntry(**x) for x in data["entries"]),
                ring_pointer=data["ring_pointer"],
                ring_bytes=data["ring_bytes"],
            )
            metadata.geometry.validate_peer(metadata.geometry)
            validate_bridge_entries(metadata.entries)
            metadata.encode()
            return metadata
        except (TypeError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid bridge metadata") from exc


def _physical_ids(values):
    result = []
    for value in values:
        if type(value) is bool:
            raise ValueError("Boolean is not a physical ID")
        try:
            result.append(operator.index(value))
        except TypeError as exc:
            raise ValueError("Physical IDs must be integers") from exc
    return tuple(result)


def checked_token_pairs(src, dst, src_capacity: int, dst_capacity: int):
    """Validate physical IDs before enqueueing any transfer or scatter."""
    src, dst = _physical_ids(src), _physical_ids(dst)
    if not src or len(src) != len(dst) or src_capacity <= 0 or dst_capacity <= 0:
        raise ValueError("Invalid source and destination token counts")
    if any(type(x) is not int or not 0 <= x < src_capacity for x in src):
        raise ValueError("Invalid source physical token ID")
    if any(type(x) is not int or not 0 <= x < dst_capacity for x in dst):
        raise ValueError("Invalid destination physical token ID")
    if len(set(dst)) != len(dst):
        raise ValueError("Duplicate destination physical token IDs")
    return src, dst


def bridge_segment_token_bounds(index_slice, num_tokens, total_suffix_tokens):
    """Translate a suffix-relative page slice and exact segment length to tokens."""
    if (
        not isinstance(index_slice, slice)
        or index_slice.step not in (None, 1)
        or type(index_slice.start) is not int
        or type(index_slice.stop) is not int
        or type(num_tokens) is not int
        or num_tokens <= 0
        or type(total_suffix_tokens) is not int
        or total_suffix_tokens <= 0
    ):
        raise ValueError("Invalid bridge segment")
    if (
        index_slice.start < 0
        or index_slice.stop <= index_slice.start
        or index_slice.stop - index_slice.start != (num_tokens + 127) // 128
    ):
        raise ValueError("Bridge page slice differs from segment length")
    start = index_slice.start * 128
    end = start + num_tokens
    if end > total_suffix_tokens or (end < total_suffix_tokens and num_tokens % 128):
        raise ValueError(
            "Bridge segment is outside or partially overlaps a suffix page"
        )
    return start, end


def expand_physical_pages(
    src_pages, dst_pages, *, segment_start, num_tokens, src_capacity, dst_capacity
):
    """Expand matched page IDs to physical rows, preserving a partial tail."""
    src_pages, dst_pages = _physical_ids(src_pages), _physical_ids(dst_pages)
    if type(segment_start) is not int or segment_start < 0 or segment_start % 128:
        raise ValueError("Bridge segment must start on a page boundary")
    if type(num_tokens) is not int or num_tokens <= 0:
        raise ValueError("Token count must be positive")
    expected_pages = (num_tokens + 127) // 128
    if len(src_pages) != expected_pages or len(dst_pages) != expected_pages:
        raise ValueError("Page counts do not match the token count")
    if any(type(x) is not int or x < 0 for x in src_pages + dst_pages):
        raise ValueError("Invalid physical page ID")
    src, dst = [], []
    for i, (sp, dp) in enumerate(zip(src_pages, dst_pages)):
        count = min(128, num_tokens - i * 128)
        src.extend(sp * 128 + offset for offset in range(count))
        dst.extend(dp * 128 + offset for offset in range(count))
    return checked_token_pairs(src, dst, src_capacity, dst_capacity)


def plan_native_page_transfers(
    src_pages,
    dst_pages,
    *,
    src_base,
    dst_base,
    src_bytes,
    dst_bytes,
    src_stride,
    dst_stride,
    item_bytes,
):
    """Plan a native index/draft plane using its own physical page stride."""
    values = (
        src_base,
        dst_base,
        src_bytes,
        dst_bytes,
        src_stride,
        dst_stride,
        item_bytes,
    )
    if any(type(x) is not int or x <= 0 for x in values):
        raise ValueError("Invalid native transfer geometry")
    if item_bytes > src_stride or item_bytes > dst_stride:
        raise ValueError("Native item exceeds page stride")
    src_pages, dst_pages = _physical_ids(src_pages), _physical_ids(dst_pages)
    if (
        not src_pages
        or len(src_pages) != len(dst_pages)
        or len(set(dst_pages)) != len(dst_pages)
    ):
        raise ValueError("Invalid native page pairs")
    result = []
    for sp, dp in zip(src_pages, dst_pages):
        if sp < 0 or dp < 0:
            raise ValueError("Negative native page ID")
        so, do = sp * src_stride, dp * dst_stride
        if so + item_bytes > src_bytes or do + item_bytes > dst_bytes:
            raise ValueError("Native transfer exceeds its owned region")
        result.append((src_base + so, dst_base + do, item_bytes))
    return tuple(result)


class LeaseState(Enum):
    WRITING = auto()
    SCATTERING = auto()
    DRAINED = auto()


@dataclass
class _Lease:
    room: Hashable
    generation: int
    session: str
    lease_id: int
    item: Hashable
    state: LeaseState = LeaseState.WRITING


class BridgeLeases:
    """Local lease bookkeeping; callers must provide real drain evidence.

    The lock protects state, not transport writes or device work. Aborted rooms
    retain their physical allocations until the caller proves both are drained.
    """

    def __init__(self, count: int = 2):
        if count <= 0:
            raise ValueError("Slot count must be positive")
        self._lock = Lock()
        self._slots = [None] * count
        self._next_id = 0
        self._last_generation = -1
        self._rooms = {}

    def open(self, room, generation: int, session: str, required):
        dynamic = required is None
        required = set() if dynamic else set(required)
        if (
            type(generation) is not int
            or generation < 0
            or not isinstance(session, str)
            or not session
            or (not dynamic and not required)
        ):
            raise ValueError("Invalid room descriptor")
        with self._lock:
            previous = self._rooms.get(room)
            if generation <= self._last_generation:
                raise ValueError("Stale room generation")
            if previous is not None:
                raise ValueError("Room must be closed before reopening")
            if any(slot is not None and slot.room == room for slot in self._slots):
                raise ValueError("Room has outstanding leases")
            self._rooms[room] = dict(
                generation=generation,
                session=session,
                required=required,
                completed=set(),
                aborted=False,
                dynamic=dynamic,
                sealed=not dynamic,
            )
            self._last_generation = generation

    def acquire(self, room, generation: int, session: str, item):
        with self._lock:
            state = self._room(room, generation, session)
            if (
                state["aborted"]
                or (item not in state["required"] and state["sealed"])
                or item in state["completed"]
            ):
                raise ValueError("Item cannot be leased")
            if any(
                x is not None and x.room == room and x.item == item for x in self._slots
            ):
                raise ValueError("Item already leased")
            for index, slot in enumerate(self._slots):
                if slot is None:
                    state["required"].add(item)
                    self._next_id += 1
                    self._slots[index] = _Lease(
                        room, generation, session, self._next_id, item
                    )
                    return index, self._next_id
            return None

    def seal(self, room, generation, session):
        with self._lock:
            state = self._room(room, generation, session)
            if not state["dynamic"] or state["aborted"]:
                raise ValueError("Room cannot be sealed")
            state["sealed"] = True

    def _room(self, room, generation, session):
        state = self._rooms.get(room)
        if (
            state is None
            or state["generation"] != generation
            or state["session"] != session
        ):
            raise ValueError("Stale room or session")
        return state

    def _lease(self, slot, room, generation, session, lease_id):
        self._room(room, generation, session)
        if type(slot) is not int or not 0 <= slot < len(self._slots):
            raise ValueError("Invalid slot")
        lease = self._slots[slot]
        if lease is None or (
            lease.room,
            lease.generation,
            lease.session,
            lease.lease_id,
        ) != (room, generation, session, lease_id):
            raise ValueError("Stale lease")
        return lease

    def write_drained(self, slot, room, generation, session, lease_id):
        with self._lock:
            lease = self._lease(slot, room, generation, session, lease_id)
            if lease.state is not LeaseState.WRITING:
                raise ValueError("Write already completed")
            lease.state = (
                LeaseState.DRAINED
                if self._rooms[room]["aborted"]
                else LeaseState.SCATTERING
            )

    def scatter_drained(self, slot, room, generation, session, lease_id):
        with self._lock:
            lease = self._lease(slot, room, generation, session, lease_id)
            if lease.state is not LeaseState.SCATTERING:
                raise ValueError("Scatter is not pending")
            lease.state = LeaseState.DRAINED

    def require_scatter(self, slot, room, generation, session, lease_id):
        with self._lock:
            lease = self._lease(slot, room, generation, session, lease_id)
            if lease.state is not LeaseState.SCATTERING:
                raise ValueError("Scatter is not pending")

    def has_live_leases(self):
        with self._lock:
            return any(slot is not None for slot in self._slots)

    def has_open_rooms(self):
        with self._lock:
            return bool(self._rooms)

    def release(self, slot, room, generation, session, lease_id):
        with self._lock:
            lease = self._lease(slot, room, generation, session, lease_id)
            if lease.state is not LeaseState.DRAINED:
                raise ValueError("Lease is not drained")
            state = self._rooms[room]
            if not state["aborted"]:
                state["completed"].add(lease.item)
            self._slots[slot] = None

    def release_aborted(self, slot, room, generation, session, lease_id):
        with self._lock:
            lease = self._lease(slot, room, generation, session, lease_id)
            if (
                not self._rooms[room]["aborted"]
                or lease.state is not LeaseState.DRAINED
            ):
                raise ValueError("Aborted lease is not drained")
            self._slots[slot] = None

    def cancel_aborted_unsubmitted(self, slot, room, generation, session, lease_id):
        with self._lock:
            lease = self._lease(slot, room, generation, session, lease_id)
            if (
                not self._rooms[room]["aborted"]
                or lease.state is not LeaseState.SCATTERING
            ):
                raise ValueError("Aborted unsubmitted scatter is not ready")
            self._slots[slot] = None

    def abort(self, room, generation, session):
        with self._lock:
            self._room(room, generation, session)["aborted"] = True

    def success(self, room, generation, session):
        with self._lock:
            state = self._room(room, generation, session)
            return (
                not state["aborted"]
                and state["sealed"]
                and state["completed"] == state["required"]
                and not any(x is not None and x.room == room for x in self._slots)
            )

    def close(self, room, generation, session):
        """Discard a completed or aborted room only after its leases drain."""
        with self._lock:
            state = self._room(room, generation, session)
            if any(x is not None and x.room == room for x in self._slots):
                raise ValueError("Room has outstanding leases")
            if not state["aborted"] and (
                not state["sealed"] or state["completed"] != state["required"]
            ):
                raise ValueError("Room is incomplete")
            del self._rooms[room]
