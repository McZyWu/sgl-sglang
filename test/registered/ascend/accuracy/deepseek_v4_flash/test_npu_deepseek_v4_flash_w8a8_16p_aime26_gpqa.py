import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

# Environment variables for DSV4-Flash single-node PD-mix deployment.
# Derived from run_dsv4_flash.sh (latest deployment script from dev).
DEEPSEEK_V4_FLASH_W8A8_16P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    # deepep
    "HCCL_BUFFSIZE": "1000",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    # skip gpu branch
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    # MTP (EAGLE) related envs
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

# Server launch arguments for DSV4-Flash W8A8 single-node 16-card PD-mix.
# Derived from run_dsv4_flash.sh (latest deployment script from dev) and the
# test case design (Excel) which requires max-running-requests=160 and MTP
# (EAGLE) enabled.
DEEPSEEK_V4_FLASH_W8A8_16P_OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.7,
    "--prefill-max-requests",
    2,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--max-running-requests",
    160,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    10,
    # MTP (EAGLE) configuration, required by the test case design (Excel S2).
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    2,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    3,
]

# Generation config shared by AIME26 and GPQA-Diamond, required by the test
# case design (Excel S4): max_tokens=125000, top_p=1, temperature=1, thinking=true.
DEEPSEEK_V4_FLASH_W8A8_GENERATION_CONFIG = {
    "max_tokens": 125000,
    "top_p": 1,
    "temperature": 1,
    "n": 1,
    "extra_body": {"chat_template_kwargs": {"thinking": True}},
}


class TestNPUDeepSeekV4FlashW8A816PAIME26(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for DeepSeek-V4-Flash W8A8 16p PD-mix on AIME26.

    Single-node 16-card PD mixed deployment with TP=16, DP=16, EP=16 and MTP
    (EAGLE) enabled. Baseline accuracy 0.933 (gap < 1% compared with paper).
    """

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_W8A8_16P_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_W8A8_16P_ENVS
    accuracy = 0.933
    datasets = ["aime26"]
    few_shot_num = 0
    generation_config = DEEPSEEK_V4_FLASH_W8A8_GENERATION_CONFIG
    eval_batch_size = 30
    limit = 30
    stream = True
    timeout = 6000
    seed = 1

    def test_npu_deepseek_v4_flash_w8a8_16p_aime26(self):
        """Run NPU accuracy test for DeepSeek-V4-Flash W8A8 16p on AIME26."""
        self.run_accuracy()


class TestNPUDeepSeekV4FlashW8A816PGPQA(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for DeepSeek-V4-Flash W8A8 16p PD-mix on GPQA-Diamond.

    Single-node 16-card PD mixed deployment with TP=16, DP=16, EP=16 and MTP
    (EAGLE) enabled. Baseline accuracy 0.712 (gap < 1% compared with paper).
    """

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_W8A8_16P_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_W8A8_16P_ENVS
    accuracy = 0.712
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    generation_config = DEEPSEEK_V4_FLASH_W8A8_GENERATION_CONFIG
    eval_batch_size = 128
    stream = True
    timeout = 6000
    seed = 1

    def test_npu_deepseek_v4_flash_w8a8_16p_gpqa(self):
        """Run NPU accuracy test for DeepSeek-V4-Flash W8A8 16p on GPQA-Diamond."""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
