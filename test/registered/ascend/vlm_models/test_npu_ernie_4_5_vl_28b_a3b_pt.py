import unittest

from sglang.test.ascend.test_ascend_utils import ERNIE_4_5_VL_28B_A3B_PT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=400,
    suite="full-2-npu-a3",
    nightly=True,
)


class TestDeepseekVl2(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT model on the MMMU dataset is no less than 0.

    [Test Category] Model
    [Test Target] PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT
    """

    model = ERNIE_4_5_VL_28B_A3B_PT_WEIGHTS_PATH
    mmmu_accuracy = 0
    other_args = [
        "--trust-remote-code",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--tp-size",
        2,
        "--mem-fraction-static",
        0.8,
        "--dtype",
        "bfloat16",
        "--enable-multimodal",
    ]

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
