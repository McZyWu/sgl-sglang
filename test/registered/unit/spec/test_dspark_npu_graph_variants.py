"""CPU tests for NPU graph selection and input-update ordering."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Variants:
    npu_graph_variants = ("sampling", "greedy")
    npu_graph_variant = "greedy"
    capturing = None

    @contextmanager
    def npu_graph_capture_variant(self, variant):
        self.capturing = variant
        try:
            yield
        finally:
            self.capturing = None


@pytest.fixture
def backend(monkeypatch):
    events = []

    class Graph:
        def __init__(self):
            self.name = None

        def replay(self):
            events.append(("replay", self.name))

        def update(self, **kwargs):
            events.append(("update", self.name, kwargs))

    @contextmanager
    def capture(graph, **kwargs):
        graph.name = (
            instance._variant_provider.capturing if instance._variant_provider else None
        )
        events.append(("capture", graph.name, kwargs["pool"]))
        yield

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(NPUGraph=Graph, graph=capture), raising=False
    )
    instance = NPUCudaGraphBackend.__new__(NPUCudaGraphBackend)
    instance._graphs = {}
    instance._outputs = {}
    instance._pool = object()
    instance._capture_stream = None
    instance._variant_provider = _Variants()
    instance._device_module = SimpleNamespace(synchronize=lambda: None)
    instance._tp_group = SimpleNamespace(barrier=lambda: None)
    instance._enable_torch_compile = False
    instance._memory_saver_adapter = None
    # Defer the actual update until result() to catch missing update/replay waits.
    instance._update_executor = Mock()
    instance._update_executor.submit.side_effect = lambda fn, **kwargs: SimpleNamespace(
        result=lambda: fn(**kwargs)
    )
    return instance, events


def test_capture_and_replay_select_staged_mode_for_every_shape(backend):
    instance, events = backend
    provider = instance._variant_provider
    reset = Mock()
    keys = [ShapeKey(size=4), ShapeKey(size=8, stream_idx=1, variant_label="test")]
    for key in keys:
        instance.capture_one(key, lambda: provider.capturing, post_warmup_hook=reset)
    assert reset.call_count == 8  # Two warmups per variant per shape.
    assert len(instance._graphs) == len(instance._outputs) == 4
    assert provider.capturing is None
    for key in keys:
        for mode in ("greedy", "sampling", "greedy", "sampling"):
            provider.npu_graph_variant = mode
            assert instance.can_run(None, key)
            assert instance.replay(key, None) == mode
            assert events[-1] == ("replay", mode)
            assert (
                instance.replay_with_input_update(key, [1, 2], attr_name="seq_lens")
                == mode
            )
            assert events[-2:] == [
                ("update", mode, {"cpu_update_input": [{"seq_lens": [1, 2]}]}),
                ("replay", mode),
            ]
    assert len([e for e in events if e[0] == "capture"]) == 4
    assert all(e[2] is instance._pool for e in events if e[0] == "capture")
    assert not instance.can_run(None, ShapeKey(size=16))
    instance.cleanup()
    assert not instance._graphs and not instance._outputs


def test_failed_input_update_does_not_replay_either_variant(backend):
    instance, events = backend
    key = ShapeKey(size=4)
    instance.capture_one(key, lambda: None)
    instance._update_executor.submit.side_effect = None
    instance._update_executor.submit.return_value.result.side_effect = RuntimeError(
        "update failed"
    )
    with pytest.raises(RuntimeError, match="update failed"):
        instance.replay_with_input_update(key, [1], attr_name="seq_lens")
    assert not any(e[0] == "replay" for e in events)


def test_non_specialized_model_retains_single_graph(backend):
    instance, events = backend
    instance._variant_provider = None
    key = ShapeKey(size=4)
    instance.capture_one(key, lambda: "original")
    assert list(instance._graphs) == [key]
    assert instance.can_run(None, key)
    assert instance.replay(key, None) == "original"
    assert events[-1] == ("replay", None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
