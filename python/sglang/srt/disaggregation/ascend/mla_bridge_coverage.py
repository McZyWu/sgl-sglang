"""Fail-closed per-room coverage for NPU MLA bridge receive slots."""

from threading import Lock

from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
    ALIGN,
    ROW_BYTES,
    _physical_ids,
)


class BridgeRoomCoverage:
    """Derive target rows from decode-owned pages, never from sender pointers.

    Chunks advance each layer in logical order. A reservation is not committed
    until the caller observes the scatter event. Native completion may only be
    recorded after its corresponding write is known complete.
    """

    def __init__(
        self,
        *,
        room,
        generation,
        session,
        pages,
        num_tokens,
        prefix_tokens,
        physical_capacity,
        layers,
        required_native,
    ):
        if (
            type(room) is not int
            or room < 0
            or type(generation) is not int
            or generation <= 0
            or not isinstance(session, str)
            or not session
        ):
            raise ValueError("Invalid bridge room identity")
        if type(num_tokens) is not int or num_tokens <= 0:
            raise ValueError("Invalid bridge token count")
        if type(prefix_tokens) is not int or prefix_tokens < 0 or prefix_tokens % 128:
            raise ValueError("Bridge prefix must be page aligned")
        if type(physical_capacity) is not int or physical_capacity <= 0:
            raise ValueError("Invalid physical token capacity")
        pages = _physical_ids(pages)
        layers = tuple(layers)
        native = tuple(required_native)
        if (
            len(pages) != (num_tokens + 127) // 128
            or any(x < 0 for x in pages)
            or len(set(pages)) != len(pages)
            or any(
                page * 128 + min(128, num_tokens - i * 128) > physical_capacity
                for i, page in enumerate(pages)
            )
        ):
            raise ValueError("Invalid decode-owned page allocation")
        if (
            not layers
            or len(set(layers)) != len(layers)
            or not native
            or len(set(native)) != len(native)
        ):
            raise ValueError("Incomplete or duplicate bridge manifest")
        self.room, self.generation, self.session = room, generation, session
        self.pages, self.num_tokens, self.prefix_tokens = (
            pages,
            num_tokens,
            prefix_tokens,
        )
        self._cursor = dict.fromkeys(layers, 0)
        self._required_native = frozenset(native)
        self._done_native = set()
        self._native_cursor = {
            item: 0 for item in native if isinstance(item, tuple) and len(item) == 3
        }
        self._pending = {}
        self._last_lease_id = 0
        self._aborted = False
        self._lock = Lock()

    def reserve(self, *, room, generation, session, layer, start, count, lease_id):
        with self._lock:
            if (room, generation, session) != (
                self.room,
                self.generation,
                self.session,
            ) or self._aborted:
                raise ValueError("Stale or aborted bridge room")
            if (
                type(start) is not int
                or type(count) is not int
                or type(lease_id) is not int
                or lease_id <= self._last_lease_id
            ):
                raise ValueError("Invalid chunk identity")
            if (
                layer not in self._cursor
                or layer in self._pending
                or start != self._cursor[layer]
                or not 0 < count <= ALIGN // ROW_BYTES
                or start + count > self.num_tokens
            ):
                raise ValueError("Invalid or duplicate layer range")
            if any(pending[2] == lease_id for pending in self._pending.values()):
                raise ValueError("Duplicate live lease ID")
            self._pending[layer] = (start, count, lease_id)
            self._last_lease_id = lease_id
            return tuple(
                self.pages[i // 128] * 128 + i % 128
                for i in range(start, start + count)
            )

    def complete_scatter(self, *, room, generation, session, layer, lease_id):
        with self._lock:
            if (room, generation, session) != (
                self.room,
                self.generation,
                self.session,
            ):
                raise ValueError("Stale bridge room")
            pending = self._pending.get(layer)
            if pending is None or pending[2] != lease_id:
                raise ValueError("Stale scatter completion")
            if not self._aborted:
                self._cursor[layer] = pending[0] + pending[1]
            del self._pending[layer]

    def complete_native_range(self, *, room, generation, session, start, count):
        """Commit all native planes only after each matching sync write succeeds."""
        with self._lock:
            if (
                (room, generation, session)
                != (self.room, self.generation, self.session)
                or self._aborted
                or type(start) is not int
                or type(count) is not int
                or count <= 0
                or start + count > self.num_tokens
                or not self._native_cursor
                or any(cursor != start for cursor in self._native_cursor.values())
            ):
                raise ValueError("Invalid native transfer range")
            end = start + count
            for component in self._native_cursor:
                self._native_cursor[component] = end
                if end == self.num_tokens:
                    self._done_native.add(component)

    def complete_terminal(self, *, room, generation, session):
        """Record final aux/state only after the sender reports success."""
        with self._lock:
            if (room, generation, session) != (
                self.room,
                self.generation,
                self.session,
            ) or self._aborted:
                raise ValueError("Invalid terminal bridge completion")
            terminal = {("aux",), ("state",)}
            if not terminal.issubset(self._required_native):
                raise ValueError("Missing terminal bridge components")
            self._done_native.update(terminal)

    def complete_native(self, *, room, generation, session, component):
        with self._lock:
            if (
                (room, generation, session)
                != (self.room, self.generation, self.session)
                or self._aborted
                or component not in self._required_native
                or component in self._done_native
                or component in self._native_cursor
            ):
                raise ValueError("Invalid native completion")
            self._done_native.add(component)

    def abort(self):
        with self._lock:
            self._aborted = True

    def success(self):
        with self._lock:
            return (
                not self._aborted
                and not self._pending
                and all(x == self.num_tokens for x in self._cursor.values())
                and self._done_native == self._required_native
            )
