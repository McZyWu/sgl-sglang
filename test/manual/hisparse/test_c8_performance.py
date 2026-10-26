"""CPU contracts for the manual A5 whole-model throughput experiment.

No model or accelerator is launched. Run with python -m pytest <this file>.
"""

import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.benchmark.stream_metrics import BatchStreamRecorder
from sglang.test.test_utils import CustomTestCase

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "benchmark/hisparse/lim_copy_sfa_c8_a5/performance.py"
)
spec = importlib.util.spec_from_file_location("c8_performance", SOURCE)
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)
spec = importlib.util.spec_from_file_location(
    "c8_trace_summary", SOURCE.with_name("summarize_offload_trace.py")
)
trace_summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace_summary)


def load_hbm_tools():
    spec = importlib.util.spec_from_file_location(
        "monitor_hbm", SOURCE.with_name("monitor_hbm.py")
    )
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    with patch.dict(sys.modules, {"performance": perf, "monitor_hbm": monitor}):
        spec = importlib.util.spec_from_file_location(
            "hbm_capacity", SOURCE.with_name("hbm_capacity.py")
        )
        capacity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capacity)
    return monitor, capacity


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
    def test_hbm_parser_and_sampler_use_each_cards_peak_not_total_or_average(self):
        monitor, _ = load_hbm_tools()
        text = "| 0 | Ascend950DT | OK |\n| 0 | NA | 100 | 0 / 0 | 77322 / 98304 |\n| 1 | Ascend950DT | OK |\n| 1 | NA | 100 | 0 / 0 | 92000 / 98304 |\n"
        self.assertEqual(monitor.parse_hbm(text)["1"]["used_mib"], 92000)
        with tempfile.TemporaryDirectory() as temporary:
            sampler = monitor.Sampler(temporary, [0, 1])
            values = monitor.parse_hbm(text)
            with patch.object(monitor, "read_hbm", return_value=values):
                sampler.sample()
                values["1"]["used_mib"] = 80000
                sampler.sample()
            summary = sampler.summary()
            self.assertEqual(summary["cases"]["case"]["max_card_peak_mib"], 92000)
            self.assertEqual(
                summary["cases"]["case"]["by_npu"]["1"]["min_free_mib"], 6304
            )
            with patch.object(monitor, "read_hbm", return_value={"0": values["0"]}):
                sampler.sample()
            self.assertEqual(len(sampler.summary()["errors"]), 1)
            sampler.performance = Path(temporary) / "performance.json"
            with patch.object(monitor, "read_hbm", return_value=values):
                for name in ("i16384-c2", "i65536-c4"):
                    perf.write_json(
                        sampler.performance, {"active_case": {"name": name}}
                    )
                    sampler.sample()
            self.assertIn("i16384-c2", sampler.summary()["cases"])
            self.assertIn("i65536-c4", sampler.summary()["cases"])
            with self.assertRaisesRegex(ValueError, "fresh HBM"):
                monitor.Sampler(temporary, [0, 1])

    def test_host_kv_guard_checks_all_ranks_before_launching_a_server(self):
        monitor, capacity = load_hbm_tools()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"num_hidden_layers": 78}))
            expected = 5 * 17664 * (656 * 78 + 4) * 8
            self.assertEqual(
                capacity.host_kv_estimate(root, 4, 16384, 1024, 8), expected
            )
            args = SimpleNamespace(
                output=root,
                devices=[0],
                interval=0.1,
                new_tokens=1024,
                mem_fraction=0.98,
            )
            settings = {
                "MODEL_PATH": str(root),
                "TP_SIZE": "8",
                "PORT": "0",
                "HOST": "127.0.0.1",
            }
            with (
                patch.object(
                    monitor,
                    "read_hbm",
                    return_value={"0": {"used_mib": 1000, "total_mib": 98304}},
                ),
                patch.object(
                    capacity.psutil,
                    "virtual_memory",
                    return_value=SimpleNamespace(available=1),
                ),
                patch.object(
                    capacity.psutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=10**15),
                ),
                patch.object(
                    capacity.subprocess,
                    "Popen",
                    side_effect=AssertionError(
                        "host guard must run before process launch"
                    ),
                ),
            ):
                result = capacity.probe(
                    args, "", settings, "lim_copy_sfa", 2048, 16384, 4
                )
            self.assertEqual(result["status"], "host_memory_limited")
            self.assertEqual(result["host_kv_estimate_bytes"], expected)

    def test_owned_process_cleanup_reaches_separate_session_child_and_spares_sibling(
        self,
    ):
        import subprocess

        import psutil

        _, capacity = load_hbm_tools()
        sibling = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
        code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);print(p.pid,flush=True);time.sleep(60)"
        parent = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        child_pid = int(parent.stdout.readline())
        try:
            capacity.remember_children(parent)
            parent.terminate()
            parent.wait(timeout=5)
            capacity.stop_owned(parent)
            self.assertIsNone(sibling.poll())
            self.assertTrue(
                not psutil.pid_exists(child_pid)
                or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
            )
        finally:
            sibling.terminate()
            sibling.wait(timeout=5)
            parent.stdout.close()
            if psutil.pid_exists(child_pid):
                psutil.Process(child_pid).kill()

    def test_capacity_admission_requires_hbm_pool_and_formal_decode_batch(self):
        _, capacity = load_hbm_tools()
        info = {"max_total_num_tokens": 10000}
        measured = [{"max": 4}]
        self.assertEqual(
            capacity.classify_point(info, measured, 4, 9000, 90000, 92160, []), "pass"
        )
        self.assertEqual(
            capacity.classify_point(info, measured, 4, 9000, 93000, 92160, []),
            "hbm_limit",
        )
        self.assertEqual(
            capacity.classify_point(info, measured, 4, 11000, 90000, 92160, []),
            "token_pool_limited",
        )
        self.assertEqual(
            capacity.classify_point(info, [{"max": 2}], 4, 9000, 90000, 92160, []),
            "batch_not_observed",
        )
        self.assertEqual(
            capacity.classify_point(info, measured, 4, 9000, None, 92160, []),
            "hbm_monitor_error",
        )

    def test_capacity_search_refines_resource_boundary_but_not_scheduler_failure(self):
        _, capacity = load_hbm_tools()
        found = capacity.search(
            [1, 2, 4, 8, 16],
            lambda batch: {
                "batch": batch,
                "status": "pass" if batch <= 6 else "hbm_limit",
            },
        )
        self.assertEqual(
            (found["max_admitted_batch"], found["next_rejected_batch"]), (6, 7)
        )
        found = capacity.search(
            [1, 2, 4, 8],
            lambda batch: {
                "batch": batch,
                "status": "pass" if batch <= 4 else "batch_not_observed",
            },
        )
        self.assertEqual([point["batch"] for point in found["points"]], [1, 2, 4, 8])
        self.assertEqual(found["boundary_reason"], "batch_not_observed")
        edge = capacity.search([1, 2], lambda batch: {"batch": batch, "status": "pass"})
        self.assertEqual(edge["boundary_reason"], "scan_edge")

    def test_measured_batch_boundary_excludes_warmup_logs(self):
        import subprocess

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server_log = root / "server.log"
            server_log.write_text("Decode batch, #running-req: 32\n")
            code = (
                "import time; from pathlib import Path; print('Starting main benchmark run...',flush=True); time.sleep(.2); p=Path("
                + repr(str(server_log))
                + "); p.write_text(p.read_text()+'Decode batch, #running-req: 4\\n'); print('report',flush=True)"
            )
            offset = perf.run_client(
                [sys.executable, "-c", code], root / "client.log", 5, server_log
            )
            self.assertEqual(
                perf.batch_observations(server_log, offset, measured=True)["max"], 4
            )
            self.assertEqual(perf.batch_observations(server_log, 0)["max"], 32)
            with self.assertRaises(subprocess.TimeoutExpired):
                perf.run_client(
                    [sys.executable, "-c", "import time; time.sleep(10)"],
                    root / "timeout.log",
                    0.1,
                    server_log,
                )

    def test_hbm_comparison_separates_max_batch_from_tpot_and_rejects_other_budget(
        self,
    ):
        _, capacity = load_hbm_tools()

        def point(batch, throughput, tpot):
            return dict(
                batch=batch,
                status="pass",
                output_throughput=throughput,
                mean_tpot_ms=tpot,
                p95_tpot_ms=tpot,
                worst_round_mean_tpot_ms=tpot,
                peak_hbm_mib=90000,
            )

        base = dict(
            status="complete",
            settings=dict(tpot_limit_ms=60, hbm_limit_gib=90),
            groups=[
                dict(
                    mode="baseline",
                    cache_tokens=2048,
                    input_tokens=16384,
                    boundary_reason="hbm_limit",
                    points=[point(2, 90, 50), point(4, 100, 80)],
                )
            ],
        )
        opt = copy.deepcopy(base)
        opt["groups"][0].update(
            mode="lim_copy_sfa", points=[point(2, 110, 50), point(8, 120, 80)]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, value in (("base", base), ("opt", opt)):
                (root / name).mkdir()
                perf.write_json(root / name / "capacity.json", value)
            args = SimpleNamespace(
                compare=[root / "base", root / "opt"], output=root / "comparison"
            )
            capacity.compare_runs(args)
            result = json.loads((args.output / "capacity.json").read_text())
            rows = {row["selection"]: row for row in result["comparisons"]}
            self.assertEqual(rows["max_batch_point"]["optimized_batch"], 8)
            self.assertEqual(rows["best_tpot_point"]["optimized_batch"], 2)
            self.assertAlmostEqual(
                rows["best_tpot_point"]["output_throughput_ratio"], 110 / 90
            )
            opt["settings"]["hbm_limit_gib"] = 91
            perf.write_json(root / "opt/capacity.json", opt)
            with self.assertRaisesRegex(ValueError, "HBM budget"):
                capacity.compare_runs(args)

    def test_decode_window_excludes_prefill_and_accounts_for_boundary_delivery(self):
        """Adding prefill delay must not lower decode throughput; chunks straddle boundaries."""
        chunks = [
            (0, 1, None, 1),
            (0, 2, None, 2),
            (1, 1, None, 3),
            (0, 4, None, 4),
            (1, 2, None, 4),
            (1, 3, None, 5),
            (0, 6, "length", 6),
            (1, 5, "length", 7),
        ]
        for prefill in (0, 100):
            recorder = BatchStreamRecorder(2)
            for index, count, finish, now in chunks:
                recorder.record_chunk(
                    index=index,
                    completion_tokens=count,
                    finish_type=finish,
                    now=now + prefill,
                )
            window = recorder.steady_state_window()
            self.assertEqual(window.duration, 3)
            self.assertEqual(window.output_throughput, 2)
            self.assertEqual(recorder.tpots_ms(), [1000, 1000])
        incomplete = BatchStreamRecorder(1)
        incomplete.record_chunk(index=0, completion_tokens=2, finish_type=None, now=1)
        with self.assertRaisesRegex(ValueError, "finish chunk"):
            incomplete.tpots_ms()

    def test_decode_rejects_clamped_output_and_marks_missing_common_window(self):
        value = dict(
            batch_size=2,
            input_len=131072,
            output_len=1024,
            total_output_tokens=2044,
            num_early_stopped=0,
            steady_state_output_throughput=100,
            steady_state_window_s=10,
            mean_tpot_ms=20,
            p95_tpot_ms=21,
        )
        with self.assertRaisesRegex(ValueError, "lengths"):
            perf.decode_metrics(value, 2, 131072, 1024)
        value["total_output_tokens"] = 2048
        self.assertEqual(
            perf.decode_metrics(value, 2, 131072, 1024)["steady_batch_step_ms"], 20
        )
        value.update(steady_state_output_throughput=None, steady_state_window_s=None)
        metrics = perf.decode_metrics(value, 2, 131072, 1024)
        self.assertEqual(metrics["status"], "no_full_batch_window")
        self.assertIsNone(metrics["decode_output_throughput"])

    def test_decode_winner_uses_decode_rate_not_prefill_amortized_rate(self):
        value = record("baseline", [2, 4, 8], [[40, 40], [45, 45], [None, None]])
        value["protocol"]["measurement"] = "decode"
        for row in value["cases"]:
            batch = row["concurrency"]
            row.update(
                decode_output_throughput={2: 300, 4: 250, 8: None}[batch],
                steady_state_window_s=10,
                steady_batch_step_ms=10,
            )
            if batch == 8:
                row["status"] = "capacity_limited"
        summary = perf.slo_summary(value)
        self.assertEqual(summary["best"][0]["point"]["client_concurrency"], 2)
        self.assertFalse(summary["points"][2]["meets_slo"])

    @staticmethod
    def trace_line(rank, forward, selected, misses):
        value = dict.fromkeys(trace_summary.COUNTERS, 0)
        value.update(
            forwards=forward,
            max_decode_seq_len=8192,
            prefix_selected_tokens=selected,
            selected_miss_tokens=misses,
            decode_layer_calls=forward,
        )
        return (
            f"[2026-10-26 12:00:00 TP{rank} EP{rank}] C8_OFFLOAD_TRACE "
            + json.dumps(value)
        )

    def test_trace_summary_uses_latest_per_rank_and_weighted_deltas(self):
        """Summing cumulative prints or averaging their percentages double-counts samples."""
        a = self.trace_line(0, 1, 10, 2)
        b = self.trace_line(0, 2, 110, 12)
        c = self.trace_line(1, 1, 20, 0)
        d = self.trace_line(1, 2, 220, 20)
        ranks = trace_summary.parse_trace("\n".join([a, c, b, b, d]))
        latest = trace_summary.summarize(ranks, "latest")["aggregate"]
        self.assertEqual(latest["prefix_selected_tokens"], 330)
        self.assertAlmostEqual(latest["prefix_cache_hit_rate_pct"], 100 * 298 / 330)
        window = trace_summary.summarize(ranks, "window")["aggregate"]
        self.assertEqual(window["prefix_selected_tokens"], 300)
        self.assertAlmostEqual(window["prefix_cache_hit_rate_pct"], 90)

    def test_trace_summary_distinguishes_no_samples_from_complete_hits(self):
        empty = trace_summary.parse_trace(self.trace_line(0, 100, 0, 0))
        row = trace_summary.summarize(empty, "latest")["aggregate"]
        self.assertEqual(row["status"], "no_cache_management_samples")
        self.assertIsNone(row["prefix_cache_hit_rate_pct"])
        hit = trace_summary.parse_trace(self.trace_line(0, 100, 20, 0))
        self.assertEqual(
            trace_summary.summarize(hit, "latest")["aggregate"][
                "prefix_cache_hit_rate_pct"
            ],
            100,
        )
        self.assertEqual(
            trace_summary.summarize(hit, "window")["aggregate"]["status"],
            "no_snapshot_window",
        )
        with self.assertRaisesRegex(ValueError, "reset"):
            trace_summary.parse_trace(
                self.trace_line(0, 100, 20, 0) + "\n" + self.trace_line(0, 1, 0, 0)
            )

    def test_gsm8k_actual_lengths_determine_cache_management_not_output_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            pairs = [(500, 512), (2200, 10), (2047, 2), (2047, 3), (8192, 1)]
            path.write_text(
                "\n".join(
                    json.dumps({"prompt_tokens": p, "completion_tokens": n})
                    for p, n in pairs
                )
            )
            stats = trace_summary.prediction_lengths(path, 2048)
            # The long prefill with one output token has no subsequent decode.
            self.assertEqual(
                stats["questions_with_possible_cache_management_decode"], 2
            )
            self.assertEqual(stats["max_completion_tokens"], 512)

    def test_trace_log_offset_excludes_previous_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.log"
            previous = self.trace_line(0, 1, 10000, 9000) + "\n"
            path.write_text(previous)
            offset = path.stat().st_size
            with path.open("a") as file:
                file.write(self.trace_line(0, 2, 10100, 9010) + "\n")
                file.write(self.trace_line(0, 3, 10200, 9015) + "\n")
            result = trace_summary.report(path, "window", offset)
            self.assertEqual(result["aggregate"]["prefix_selected_tokens"], 100)
            self.assertEqual(result["aggregate"]["prefix_cache_hit_rate_pct"], 95)
            with self.assertRaisesRegex(ValueError, "truncated"):
                trace_summary.report(path, offset=path.stat().st_size + 1)

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
