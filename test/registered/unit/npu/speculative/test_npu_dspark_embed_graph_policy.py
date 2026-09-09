"""Exercise production capture/staging branches without importing the runtime."""

import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]
RUNNER = ROOT / "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
NPU_RUNNER = (
    ROOT / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py"
)


def _branch(path, *, capture):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if capture:
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(
                isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Subscript)
                    and ast.unparse(target) == "kwargs['input_embeds']"
                    for target in stmt.targets
                )
                for stmt in node.body
            )
        ]
    else:
        execute = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "execute"
        )
        matches = [
            node
            for node in ast.walk(execute)
            if isinstance(node, ast.If)
            and "forward_batch.input_embeds is not None" in ast.unparse(node.test)
        ]
    assert len(matches) == 1, (path, len(matches))
    module = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    return compile(module, str(path), "exec")


def _algorithm(name):
    return SimpleNamespace(
        is_dflash=lambda: name == "DFLASH",
        is_dflash_family=lambda: name in {"DFLASH", "DSPARK"},
        is_dspark=lambda: name == "DSPARK",
    )


class TestNpuDsparkEmbedGraphPolicy(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        self.capture_branch = _branch(RUNNER, capture=True)
        self.stage_branch = _branch(NPU_RUNNER, capture=False)
        self.buffer = torch.zeros((224, 16), device=self.device)
        self.runner = SimpleNamespace(
            model_runner=SimpleNamespace(
                spec_algorithm=_algorithm("DSPARK"),
                is_draft_worker=True,
                model=SimpleNamespace(forward_embed=lambda _: None),
            ),
            buffers=SimpleNamespace(input_embeds=self.buffer),
            raw_num_token=224,
        )

    def capture_kwargs(self, *, enabled, npu=True):
        def forward(input_ids, positions, batch, input_embeds=None):
            pass

        namespace = dict(
            self=self.runner,
            kwargs={},
            num_tokens=224,
            forward=forward,
            inspect=inspect,
            is_npu=lambda: npu,
            envs=SimpleNamespace(
                SGLANG_DSPARK_EMBED_IN_GRAPH=SimpleNamespace(get=lambda: enabled)
            ),
        )
        exec(self.capture_branch, namespace)
        return namespace["kwargs"]

    def stage(self, live):
        exec(
            self.stage_branch,
            {"self": self.runner, "forward_batch": SimpleNamespace(input_embeds=live)},
        )

    def test_off_captures_the_staged_buffer_and_bypasses_embedding(self):
        kwargs = self.capture_kwargs(enabled=False)
        self.assertIn("input_embeds", kwargs)
        self.assertEqual(kwargs["input_embeds"].data_ptr(), self.buffer.data_ptr())
        for value in (2, 7, -3):
            self.stage(torch.full_like(self.buffer, value))
            torch.testing.assert_close(kwargs["input_embeds"], self.buffer)
            self.assertTrue(torch.all(kwargs["input_embeds"] == value).item())

    def test_on_keeps_embedding_inside_the_graph(self):
        self.assertEqual(self.capture_kwargs(enabled=True), {})

    def test_other_backends_and_algorithms_keep_existing_capture_policy(self):
        self.assertEqual(self.capture_kwargs(enabled=False, npu=False), {})
        for name in ("DFLASH", "EAGLE"):
            self.runner.model_runner.spec_algorithm = _algorithm(name)
            self.assertEqual(self.capture_kwargs(enabled=False), {})

    def test_models_without_forward_embed_keep_staged_input(self):
        self.runner.model_runner.model = SimpleNamespace()
        for name in ("DFLASH", "DSPARK"):
            self.runner.model_runner.spec_algorithm = _algorithm(name)
            self.assertIn("input_embeds", self.capture_kwargs(enabled=True))

    def test_target_graph_never_captures_draft_embeddings(self):
        self.runner.model_runner.is_draft_worker = False
        self.assertEqual(self.capture_kwargs(enabled=False), {})

    def test_preplanned_staging_handles_shrinking_batches_and_none(self):
        self.buffer.fill_(-11)
        self.runner.raw_num_token = 28
        self.stage(torch.full_like(self.buffer[:28], 5))
        self.stage(None)
        self.assertTrue(torch.all(self.buffer[:28] == 5).item())
        self.assertTrue(torch.all(self.buffer[28:] == -11).item())
        self.runner.model_runner.spec_algorithm = _algorithm("EAGLE")
        self.stage(torch.zeros_like(self.buffer[:28]))
        self.assertTrue(torch.all(self.buffer[:28] == 5).item())

    def test_npu_graph_consumes_new_eager_embeddings_on_replay(self):
        if self.device != "npu":
            self.skipTest("requires an NPU graph runtime")
        kwargs = self.capture_kwargs(enabled=False)
        # Capture the same static input view the production runner supplies.
        # Real graph replay must observe every subsequent preplanned update.
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = kwargs["input_embeds"] * 3 + 1
        torch.npu.current_stream().wait_stream(stream)
        for value in (2, 7, -3, 0):
            self.stage(torch.full_like(self.buffer, value))
            graph.replay()
            torch.testing.assert_close(output, torch.full_like(output, value * 3 + 1))


if __name__ == "__main__":
    unittest.main()
