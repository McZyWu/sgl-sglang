import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_greedy_top1_npu import (
    select_vanilla_global_top1_npu,
    select_vanilla_local_top1_npu,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=15, suite="base-b-test-1-npu-a3")


class TestVanillaTop1(unittest.TestCase):
    @torch.inference_mode()
    def test_bf16_rounding_ties_padding_and_graph_input_updates(self):
        for width in (17, 5120, 10240):
            with self.subTest(width=width):
                storage = torch.randn(4, 7, width, dtype=torch.bfloat16, device="npu")
                base = storage[:, 2, :]
                bias = torch.randn_like(base)
                graph = torch.npu.NPUGraph()
                select_vanilla_local_top1_npu(base, bias, vocab_offset=10240)
                with torch.npu.graph(graph):
                    candidates = select_vanilla_local_top1_npu(
                        base, bias, vocab_offset=10240
                    )
                for update in range(3):
                    base.normal_()
                    bias.normal_()
                    # FP32 distinguishes these two sums, but BF16 makes a tie.
                    base[0].fill_(-100)
                    base[0, :2] = 1
                    bias[0, 0] = 0
                    bias[0, 1] = 0.001
                    if update == 1:
                        base[1].fill_(float("-inf"))
                    if update == 2:
                        base[2, 5] = float("nan")
                    graph.replay()
                    torch.npu.synchronize()
                    values, ids = (base + bias).max(-1)
                    expected = torch.stack((values.float(), (ids + 10240).float()), -1)
                    torch.testing.assert_close(
                        candidates, expected, rtol=0, atol=0, equal_nan=True
                    )

    @torch.inference_mode()
    def test_global_first_id_ties_and_nan_under_replay(self):
        candidates = torch.tensor(
            [
                [[1.0, 10.0], [1.0, 2.0], [0.0, 0.0]],
                [[float("-inf"), 10.0], [float("-inf"), 2.0], [float("-inf"), 0.0]],
                [[float("nan"), 10.0], [float("nan"), 2.0], [5.0, 0.0]],
            ],
            device="npu",
            dtype=torch.float32,
        )
        graph = torch.npu.NPUGraph()
        select_vanilla_global_top1_npu(candidates, vocab_size=16)
        with torch.npu.graph(graph):
            tokens = select_vanilla_global_top1_npu(candidates, vocab_size=16)
        graph.replay()
        torch.testing.assert_close(tokens, torch.tensor([2, 0, 2], device="npu"))
        candidates[:, 2, 0] = float("nan")
        graph.replay()
        torch.testing.assert_close(
            tokens, torch.zeros(3, dtype=torch.long, device="npu")
        )


if __name__ == "__main__":
    unittest.main()
