"""NPUCudaGraphBackend — Ascend NPU full-graph capture (torch.npu.NPUGraph).

Mirrors FullCudaGraphBackend with two differences:
  - Captures via torch.npu.graph(...) into torch.npu.NPUGraph.
  - replay_with_input_update(shape_key, seq_lens, attr_name) rebinds
    the recorded graph's input bindings for variable seq_lens at replay
    time (NPU's NPUGraph.update(...) API).

torch.npu is imported lazily inside methods so the module loads on
non-NPU hosts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.utils import empty_context, get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class NPUCudaGraphBackend(BaseCudaGraphBackend):
    """Capture attention metadata inside each shape's torch.npu.NPUGraph.

    A folded tail can provide capture contexts and a host-selected replay
    variant (DSpark DP1 uses greedy/sampling). replay_with_input_update
    substitutes fresh seq_lens in the selected graph without re-recording.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, Any] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._device_id = self._device_module.current_device()
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._variant_provider = getattr(
            cuda_graph_runner.model_runner, "npu_graph_variant_provider", None
        )
        self._capture_stream = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        self._enable_torch_compile = getattr(
            cuda_graph_runner, "enable_torch_compile", False
        )
        # Reuse one device-bound worker for graph input updates.
        self._update_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="npu-graph-update",
            initializer=self._device_module.set_device,
            initargs=(self._device_id,),
        )

    @contextmanager
    def capture_session(self, stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # The provider's capture context selects Python branches while recording;
        # its replay property uses already-staged host metadata, without a D2H
        # read or recapture. Variants share the graph pool and run exclusively.
        provider = self._variant_provider
        if provider is None:
            self._capture_one(shape_key, forward_fn, post_warmup_hook)
            return
        for variant in provider.npu_graph_variants:
            with provider.npu_graph_capture_variant(variant):
                self._capture_one((shape_key, variant), forward_fn, post_warmup_hook)

    def _capture_one(self, graph_key, forward_fn, post_warmup_hook) -> None:
        import torch_npu  # noqa: F401  (verifies NPU availability)

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.npu.NPUGraph()

        if self._enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = torch.npu.graph

        with (
            skip_guard_context,
            graph_ctx(
                graph,
                pool=self._pool,
                stream=self._capture_stream,
                auto_dispatch_capture=True,
            ),
        ):
            out = forward_fn()

        self._graphs[graph_key] = graph
        self._outputs[graph_key] = out

    def _replay_key(self, shape_key):
        if self._variant_provider is None:
            return shape_key
        return (shape_key, self._variant_provider.npu_graph_variant)

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return self._replay_key(shape_key) in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        graph_key = self._replay_key(shape_key)
        self._graphs[graph_key].replay()
        return self._outputs[graph_key]

    def replay_with_input_update(
        self,
        shape_key: ShapeKey,
        seq_lens: Any,
        attr_name: str = None,
        attr_type: Any = None,
        cpu_update_input: list = None,
    ) -> Any:
        """Rebind seq_lens on the recorded NPU graph, then replay.

        NPUGraph.update must complete before replay can consume the updated
        KV lengths. Used when the model is not deepseek-nsa.

        Two calling conventions:
        1. (legacy) seq_lens + attr_name + attr_type:
           Constructs cpu_update_input=[{attr_name: seq_lens}] internally.
        2. cpu_update_input: A list of {attr_name: seq_lens} dicts,
           one per speculative step.  Used by EAGLE draft runners.
        """
        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        graph_key = self._replay_key(shape_key)
        graph = self._graphs[graph_key]

        update_future = self._update_executor.submit(
            graph.update, cpu_update_input=cpu_update_input
        )
        update_future.result()
        graph.replay()
        return self._outputs[graph_key]

    def cleanup(self) -> None:
        self._update_executor.shutdown(wait=True, cancel_futures=True)
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
