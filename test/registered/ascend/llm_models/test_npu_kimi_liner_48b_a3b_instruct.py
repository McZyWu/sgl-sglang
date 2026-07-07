import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import KIMI_LINEAR_48B_A3B_INSTRUCT_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestAFM(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the moonshotai/Kimi-Linear-48B-A3B-Instruct model on the GSM8K dataset is no less than 0.

    [Test Category] Model
    [Test Target] moonshotai/Kimi-Linear-48B-A3B-Instruct
    """

    model = KIMI_LINEAR_48B_A3B_INSTRUCT_MODEL_PATH
    accuracy = 0


if __name__ == "__main__":
    unittest.main()
