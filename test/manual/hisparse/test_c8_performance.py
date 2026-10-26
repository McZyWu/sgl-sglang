"""CPU contracts for the manual A5 whole-model throughput experiment.

No model or accelerator is launched. Run with python -m pytest <this file>.
"""

import copy
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "benchmark/hisparse/lim_copy_sfa_c8_a5/performance.py"
)
spec = importlib.util.spec_from_file_location("c8_performance", SOURCE)
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)


def record(mode, concurrencies, tpots):
    result = {
        "status": "pass",
        "mode": mode,
        "objective": {"metric": "mean_tpot_ms", "strict_upper_bound_ms": 60},
        "protocol": {
            "input_lengths": [131072],
            "concurrencies": concurrencies,
            "new_tokens": 1024,
            "rounds": 2,
            "waves": 4,
            "seed": 42,
            "dataset_sha256": "same-prompts",
        },
        "server_configuration": {key: None for key in perf.SERVER_KEYS},
        "launch": {
            "sglang_revision": "same",
            "kernel_revision": "same",
            "packages": {},
            "environment": {key: None for key in perf.ENV_KEYS},
        },
        "cases": [],
    }
    result["launch"]["environment"]["SGLANG_NPU_SPARSE_KV_CACHE_SIZE"] = "2048"
    result["server_configuration"].update(
        model_path="same-model", mem_fraction_static=0.8
    )
    for concurrency, rounds in zip(concurrencies, tpots):
        for repeat, tpot in enumerate(rounds, 1):
            result["cases"].append(
                {
                    "input_tokens": 131072,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    **{key: 1.0 for key in perf.METRICS},
                    "output_throughput": concurrency * 100.0,
                    "mean_tpot_ms": tpot,
                }
            )
    return result


class TestC8Performance(CustomTestCase):
    def test_full_output_needs_scheduler_headroom_and_actual_pool(self):
        """131072+1024 context produced 1022 outputs in the reported A5 run."""
        config = {"context_length": 132096, "max_total_num_tokens": 147456}
        with self.assertRaisesRegex(ValueError, "132098"):
            perf.validate_capacity(config, [131072], 1024)
        config["context_length"] = 135168
        self.assertEqual(perf.validate_capacity(config, [131072], 1024), 147456)
        config.update(max_total_num_tokens=131072, max_total_tokens=147456)
        with self.assertRaisesRegex(ValueError, "Actual token pool"):
            perf.validate_capacity(config, [131072], 1024)

    def test_best_throughput_can_use_different_concurrency_but_every_round_must_pass(
        self,
    ):
        """Do not choose a fast point whose slow repetition violates the limit."""
        base = record("baseline", [2, 4], [[40, 40], [59, 61]])
        opt = record("lim_copy_sfa", [4, 8], [[49, 51], [55, 56]])
        base["server_configuration"].update(
            max_total_tokens=147456,
            max_running_requests=4,
            cuda_graph_bs_decode=[1, 2, 4],
        )
        opt["server_configuration"].update(
            max_total_tokens=1048576,
            max_running_requests=8,
            cuda_graph_bs_decode=[1, 2, 4, 8],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, value in (("base", base), ("opt", opt)):
                (root / name).mkdir()
                perf.write_json(root / name / "performance.json", value)
            args = SimpleNamespace(
                baseline=[root / "base"],
                optimized=[root / "opt"],
                output=root / "comparison",
            )
            perf.compare(args)
            row = json.loads((args.output / "comparison.json").read_text())["rows"][0]
            self.assertEqual(
                (
                    row["baseline_client_concurrency"],
                    row["optimized_client_concurrency"],
                ),
                (2, 8),
            )
            self.assertEqual(row["throughput_speedup"], 4)
            changed = copy.deepcopy(opt)
            changed["server_configuration"]["model_path"] = "different-model"
            perf.write_json(root / "opt/performance.json", changed)
            with self.assertRaisesRegex(ValueError, "model_path"):
                perf.compare(args)
        opt["cases"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            perf.slo_summary(opt)

    def test_no_eligible_point_is_not_reported_as_a_speedup(self):
        result = record("baseline", [2], [[60, 60]])
        self.assertIsNone(perf.slo_summary(result)["best"][0]["point"])

    def test_total_indexer_pool_is_independent_of_per_request_native_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "base.env"
            source.write_text("MAX_TOTAL_TOKENS=147456\nGRAPH_BATCH_SIZE=16\n")
            args = SimpleNamespace(
                config=source,
                output=root / "configs",
                context_length=135168,
                max_total_tokens=1048576,
                max_running_requests=16,
                concurrencies=[2, 4, 8, 16],
            )
            perf.prepare_configs(args)
            generated = (args.output / "c2048-lim_copy_sfa.env").read_text()
            self.assertIn("MAX_TOTAL_TOKENS=1048576", generated)
            self.assertIn('GRAPH_BATCH_SIZES="1 2 4 8 16"', generated)
            args.context_length = 262144
            with self.assertRaisesRegex(ValueError, "262140"):
                perf.prepare_configs(args)


if __name__ == "__main__":
    import unittest

    unittest.main()
