# NPU backend checklist

Apply only the items reachable from the changed code path. Trace the call chain and verify against the active backend, device, runtime versions, and checked-out branch.

- Device/backend guards preserve CPU and GPU behavior and select the intended NPU path.
- Shapes, dtypes, strides/layouts, dynamic sequence lengths, tail cases, and output or in-place contracts match callers.
- KV cache page/slot indices, capacity versus concurrent requests, length bounds, cancellation, reuse, and resource lifetime are sound.
- Streams/events establish required happens-before ordering; H2D/D2H buffers remain alive; implicit `.item()` or equivalent synchronization is understood.
- Graph capture and replay preserve required pointer and shape stability.
- For material behavior/performance changes, compare equivalent model, hardware, environment, parallelism, batch, input/output tokens, and configuration. Check accuracy as well as throughput/latency; startup or HTTP 200 is insufficient.
- State evidence for performance claims. Treat unmeasured gains as hypotheses; profile only to resolve a concrete question.

Before distributed or hardware validation, inspect the actual environment and available devices. Do not assume a remote host, path, container, or device mapping.
