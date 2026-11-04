"""Run selected SplitKV CopySFA tests and report their CPU-oracle errors."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

CASES = (
    "test_copy_sfa_dynamic_tnd_tail_crosses_page_boundary",
    "test_copy_sfa_dynamic_tnd_head_count_128",
    "test_copy_sfa_dynamic_tnd_lim_sized_first_fill_graph",
)


def load_tests(kernel_checkout):
    tests_dir = kernel_checkout.resolve(strict=True) / "tests"
    path = tests_dir / "test_copy_sfa_dynamic_tnd.py"
    if not path.is_file() or not (tests_dir / "split_kv_fixture.py").is_file():
        raise ValueError(f"Not a SplitKV kernel checkout: {kernel_checkout}")
    sys.path.insert(0, str(tests_dir))
    spec = importlib.util.spec_from_file_location("splitkv_dynamic_tnd_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load kernel tests from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-checkout", required=True, type=Path)
    args = parser.parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("An available NPU is required; skipped tests are not passes")
    torch.npu.set_device(0)
    module = load_tests(args.kernel_checkout)
    original = module._DynamicTndCase.assert_result
    measurements = []

    def measure(self, actual, label):
        original(self, actual, label)
        diff = (actual.cpu().float() - self.expected).abs()
        result = {
            "type": "oracle_measurement",
            "case": current_case,
            "label": label,
            "query_shape": list(self.query.shape),
            "dtype": str(actual.dtype),
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
        }
        measurements.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    module._DynamicTndCase.assert_result = measure
    results = []
    for current_case in CASES:
        try:
            getattr(module, current_case)()
        except pytest.skip.Exception as exc:
            raise RuntimeError(f"Kernel test skipped: {current_case}: {exc}") from exc
        result = {
            "type": "case",
            "case": current_case,
            "passed": True,
            "cache_exact_zero_tolerance": True,
            "eager_graph_exact_zero_tolerance": True,
            "measurements": [x for x in measurements if x["case"] == current_case],
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    print(
        json.dumps(
            {"type": "summary", "passed": True, "cases": results}, sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
