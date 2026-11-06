"""Explicitly owned aligned memory for the experimental NPU PD bridge."""

import torch

from sglang.srt.disaggregation.ascend.mla_bridge_contract import ALIGN


class RegisteredBridgeBuffer:
    """A registered subrange of one owned allocation.

    Construct and retain this owner before calling register. A failed register
    may have partly succeeded, so close must successfully unregister before the
    tensor can be released. The caller establishes transport and stream drain.
    """

    def __init__(self, *, usable_bytes: int, device):
        if usable_bytes <= 0 or usable_bytes % ALIGN:
            raise ValueError("Bridge buffer size must be positive and 2 MiB aligned")
        self.raw = torch.empty(usable_bytes + ALIGN, dtype=torch.uint8, device=device)
        offset = (-self.raw.data_ptr()) % ALIGN
        if offset + usable_bytes > self.raw.numel():
            raise RuntimeError("Aligned range exceeds its owned allocation")
        self.buffer = self.raw[offset : offset + usable_bytes]
        self.pointer = self.buffer.data_ptr()
        self.size = usable_bytes
        self._registration_attempted = False
        self.registered = False

    def register(self, engine):
        if self.raw is None or self._registration_attempted:
            raise RuntimeError("Bridge buffer is closed or already registered")
        self._registration_attempted = True
        rc = engine.batch_register_memory([self.pointer], [self.size])
        if rc != 0:
            raise RuntimeError(f"NPU bridge registration failed: {rc!r}")
        self.registered = True

    def close(self, engine, *, transport_drained: bool, synchronize):
        if self.raw is None:
            return
        if not transport_drained:
            raise RuntimeError(
                "Cannot release bridge buffer while a writer may be active"
            )
        synchronize()
        if self._registration_attempted:
            rc = engine.unregister_memory(self.pointer)
            if rc != 0:
                raise RuntimeError(f"NPU bridge unregister failed: {rc!r}")
            self._registration_attempted = False
            self.registered = False
        self.buffer = None
        self.raw = None
