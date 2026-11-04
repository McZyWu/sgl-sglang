import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.runner import base_cuda_graph_runner as graph_runner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestDraftGraphBuckets(CustomTestCase):
    def test_tp16_phase_widths_and_padding(self):
        runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(size=20), server_args=object()
        )
        graph = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=[20])),
            torch_compile_max_bs=32,
        )
        exec_config = SimpleNamespace(
            graph=graph, overlap=SimpleNamespace(enable_two_batch_overlap=False)
        )
        flags = SimpleNamespace(capture=SimpleNamespace(enable_torch_compile=False))
        with (
            patch.object(graph_runner, "get_exec", return_value=exec_config),
            patch.object(graph_runner, "get_flags", return_value=flags),
            patch.object(
                graph_runner.DllmConfig, "from_server_args", return_value=None
            ),
            patch.object(
                graph_runner, "get_cuda_graph_batch_size_alignment", return_value=16
            ),
        ):
            capture = graph_runner.get_batch_sizes_to_capture
            self.assertEqual(capture(runner, 1, pad_unaligned=True)[0], [32])
            # Target verify and draft extend both have four rows per request.
            self.assertEqual(capture(runner, 4)[0], [20])
            with self.assertRaisesRegex(AssertionError, r"capture_bs=\[\]"):
                capture(runner, 1)
            graph.cuda_graph_config.decode.bs = [16, 20]
            self.assertEqual(capture(runner, 1, pad_unaligned=True)[0], [16, 32])
            self.assertEqual(capture(runner, 4)[0], [16, 20])
            graph.cuda_graph_config.decode.bs = [16]
            self.assertEqual(capture(runner, 1, pad_unaligned=True)[0], [16])
            self.assertEqual(capture(runner, 1)[0], [16])
            # Even for a smaller pool, the padded bucket is clamped to its
            # alignment-rounded capacity, rather than the requested size.
            graph.cuda_graph_config.decode.bs = [64]
            self.assertEqual(capture(runner, 1, pad_unaligned=True)[0], [32])


if __name__ == "__main__":
    unittest.main()
