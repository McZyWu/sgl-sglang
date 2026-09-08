"""NPU parity for persistent KDA state, including live graph replay indices."""

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.attention import kda_state_commit as impl
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=20, suite="stage-a-unit-test-npu")


def _load_backend_commit():
    # Execute the production override without constructing a model or HCCL
    # groups. Both the ordinary and opt-in branches come from the source.
    source = Path(impl.__file__).with_name("ascend_kda_backend.py").read_text()
    method = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "update_mamba_state_after_mtp_verify"
    )
    scope = {"torch": torch}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<kda-backend>", "exec"),
        scope,
    )
    return scope[method.name]


class TestKDAStateCommit(unittest.TestCase):
    def test_persistent_state_and_graph_replay(self):
        torch.manual_seed(31)
        layers, pool, requests, steps, channels, window = 2, 12, 3, 8, 256, 3
        for snapshots in (False, True):
            for fallback in (False, True, "baseline"):
                for tracking in (False, True):
                    with self.subTest(
                        snapshots=snapshots, fallback=fallback, tracking=tracking
                    ):
                        conv_shape = (
                            (layers, pool, channels, window)
                            if snapshots
                            else (layers, pool, steps + window - 1, channels)
                        )
                        original_conv = torch.randn(conv_shape, dtype=torch.bfloat16)
                        original_temporal = torch.randn(layers, pool, 2, 16, 16)
                        source = torch.randn(layers, requests, steps, 2, 16, 16)
                        conv_source = torch.randn(
                            layers,
                            requests,
                            steps,
                            channels,
                            window,
                            dtype=torch.bfloat16,
                        )
                        caches = SimpleNamespace(
                            conv=[original_conv.to("npu")],
                            temporal=original_temporal.transpose(-1, -2)
                            .contiguous()
                            .to("npu")
                            .transpose(-1, -2),
                            intermediate_ssm=source.to("npu"),
                            intermediate_conv_window=[conv_source.to("npu")],
                        )
                        dst = torch.tensor([0, 1, 2], device="npu", dtype=torch.int32)
                        accepted = torch.tensor(
                            [0, 3, -1], device="npu", dtype=torch.int32
                        )
                        track = (
                            torch.tensor([6, 7, 8], device="npu", dtype=torch.int32)
                            if tracking
                            else None
                        )
                        track_steps = (
                            torch.tensor([2, -1, 6], device="npu", dtype=torch.int32)
                            if tracking
                            else None
                        )

                        backend = SimpleNamespace(
                            linear_attn_backend=SimpleNamespace(
                                use_fast_state_commit=fallback != "baseline",
                                supports_speculative_conv_state_snapshots=snapshots,
                                forward_metadata=SimpleNamespace(
                                    mamba_cache_indices=dst
                                ),
                                req_to_token_pool=SimpleNamespace(
                                    get_speculative_mamba2_params_all_layers=lambda: caches
                                ),
                            )
                        )
                        backend_commit = _load_backend_commit()

                        def call():
                            backend_commit(backend, accepted, track, track_steps, None)

                        with contextlib.ExitStack() as stack:
                            if fallback is True:
                                for name in (
                                    "move_kda_temporal_snapshot",
                                    "scatter_kda_conv_snapshot",
                                    "commit_kda_extended_conv_state",
                                ):
                                    stack.enter_context(
                                        patch.object(impl, name, return_value=False)
                                    )
                            call()
                            torch.npu.synchronize()
                            graph = torch.npu.NPUGraph()
                            with torch.npu.graph(graph):
                                call()

                        for primary_steps, tracking_steps in (
                            ([0, 3, -1], [2, -1, 6]),
                            ([7, 1, 5], [-1, 7, 0]),
                        ):
                            caches.conv[0].copy_(original_conv)
                            caches.temporal.copy_(original_temporal)
                            accepted.copy_(
                                torch.tensor(primary_steps, dtype=torch.int32)
                            )
                            if tracking:
                                track_steps.copy_(
                                    torch.tensor(tracking_steps, dtype=torch.int32)
                                )
                            graph.replay()
                            torch.npu.synchronize()
                            expected_temporal = original_temporal.clone()
                            expected_conv = original_conv.clone()
                            destinations = [(list(range(requests)), primary_steps)]
                            if tracking:
                                destinations.append(([6, 7, 8], tracking_steps))
                            for slots, accepted_steps in destinations:
                                for row, (slot, step) in enumerate(
                                    zip(slots, accepted_steps)
                                ):
                                    if step < 0:
                                        continue
                                    expected_temporal[:, slot] = source[:, row, step]
                                    if snapshots:
                                        expected_conv[:, slot] = conv_source[
                                            :, row, step
                                        ]
                                    else:
                                        expected_conv[:, slot, -window:] = (
                                            original_conv[:, row, step : step + window]
                                        )
                            torch.testing.assert_close(
                                caches.temporal.cpu(), expected_temporal, rtol=0, atol=0
                            )
                            actual = caches.conv[0].cpu()
                            if not snapshots:
                                actual = actual[:, :, -window:]
                                expected_conv = expected_conv[:, :, -window:]
                            torch.testing.assert_close(
                                actual, expected_conv, rtol=0, atol=0
                            )
                        del graph


if __name__ == "__main__":
    unittest.main()
