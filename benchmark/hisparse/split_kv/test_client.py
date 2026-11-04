"""CPU-only checks of client validation and the automatic phase sequence."""

import errno
import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "splitkv_runner", Path(__file__).with_name("run.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestClient(unittest.TestCase):
    def test_host_validation_rejects_noncanonical_or_unsupported_addresses(self):
        for bad_host in (
            "809",
            "127.0.0.1:41000",
            "http://127.0.0.1",
            "::1",
            "0.0.0.0",
            "224.0.0.1",
            "255.255.255.255",
        ):
            with self.subTest(host=bad_host), self.assertRaises(ValueError):
                runner.validate_host_addresses(bad_host, "192.168.1.20", "prefill")

    def test_host_validation_binds_only_this_role_address(self):
        class FakeSocket:
            def __init__(self):
                self.bound = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def bind(self, address):
                self.bound.append(address)

        fake_socket = FakeSocket()
        with patch.object(runner.socket, "socket", return_value=fake_socket):
            runner.validate_host_addresses("192.168.1.10", "192.168.1.20", "decode")
        self.assertEqual(fake_socket.bound, [("192.168.1.20", 0)])

    def test_host_validation_reports_nonlocal_bind_with_address_guidance(self):
        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def bind(self, address):
                raise OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")

        with patch.object(runner.socket, "socket", return_value=FakeSocket()):
            with self.assertRaisesRegex(
                ValueError,
                r"--prefill-host 192\.168\.1\.10 cannot be bound.*EADDRNOTAVAIL.*ip -br -4 addr",
            ):
                runner.validate_host_addresses(
                    "192.168.1.10", "192.168.1.20", "prefill"
                )

    def test_host_validation_accepts_private_ipv4_and_real_loopback_bind(self):
        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def bind(self, address):
                pass

        with patch.object(runner.socket, "socket", return_value=FakeSocket()):
            runner.validate_host_addresses("10.1.2.3", "10.1.2.4", "prefill")
        runner.validate_host_addresses("127.0.0.1", "127.0.0.2", "prefill")

    def test_invalid_cli_host_fails_before_creating_run_or_loading_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            output = root / "runs"
            command = [
                sys.executable,
                str(Path(__file__).with_name("run.py")),
                "prefill",
                "--model",
                str(model),
                "--prefill-host",
                "809",
                "--decode-host",
                "192.168.1.20",
                "--devices",
                "0",
                "--interface",
                "eth0",
                "--tp",
                "1",
                "--output-dir",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--prefill-host must be a full IPv4 address", result.stderr)
            self.assertFalse(output.exists())

    def test_default_frozen_controller_runs_all_rounds_after_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            output = Path(tmp) / "runs"
            sequence = []
            phase = {"value": None}
            server_envs = {}

            def start(command, env, log_path):
                name = Path(log_path).name
                if name.startswith("decode-"):
                    phase["value"] = name.removeprefix("decode-").removesuffix(".log")
                    server_envs[phase["value"]] = env.copy()
                    if phase["value"] == "performance":
                        Path(log_path).write_text(
                            "Decode batch, #running-req: 2, #token: 1\n"
                        )
                return object()

            def generate(router, ids, length, rid):
                return {
                    "request_id": rid,
                    "response": {
                        "output_ids": [7] * length,
                        "meta_info": {
                            "prompt_tokens": len(ids),
                            "completion_tokens": length,
                            "finish_reason": {"type": "length"},
                            "num_retractions": int(phase["value"] == "functional"),
                        },
                    },
                    "elapsed_seconds": 1,
                }

            def frozen_round(router, model_path, out, ids, bs, label):
                sequence.append(label)
                return {
                    "round": label,
                    "output_tokens_per_second": 42.0,
                    "requests": bs,
                }

            argv = [
                "run.py",
                "decode",
                "--model",
                str(model),
                "--prefill-host",
                "127.0.0.2",
                "--decode-host",
                "127.0.0.1",
                "--devices",
                "0,1",
                "--interface",
                "lo",
                "--tp",
                "2",
                "--bs",
                "2",
                "--output-dir",
                str(output),
            ]
            original_stdout, original_stderr = sys.stdout, sys.stderr
            try:
                with (
                    patch.object(sys, "argv", argv),
                    patch.dict(
                        sys.modules,
                        {"frozen_bench": types.SimpleNamespace(run_round=frozen_round)},
                    ),
                    patch.object(
                        runner, "runtime_info", return_value={"visible_npu_count": 2}
                    ),
                    patch.object(runner, "check_split_abi", return_value={}),
                    patch.object(runner, "fixture", return_value=[1] * 64000),
                    patch.object(runner, "wait_http"),
                    patch.object(runner, "port_free"),
                    patch.object(
                        runner,
                        "flush",
                        side_effect=lambda *args: sequence.append("flush"),
                    ),
                    patch.object(runner, "start", side_effect=start),
                    patch.object(runner, "stop"),
                    patch.object(runner, "generate", side_effect=generate),
                ):
                    runner.main()
            finally:
                if isinstance(sys.stdout, runner.Tee):
                    sys.stdout.log.close()
                sys.stdout, sys.stderr = original_stdout, original_stderr
            directory = next(output.iterdir())
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(
                [item["round"] for item in summary["rounds"]],
                [
                    "warm-1",
                    "warm-2",
                    "formal-1",
                    "formal-2",
                    "formal-3",
                    "formal-4",
                    "formal-5",
                ],
            )
            self.assertEqual(summary["formal_output_tps"]["median"], 42.0)
            self.assertIn(
                "not a full model benchmark accuracy evaluation",
                summary["greedy_accuracy_validation"]["scope"],
            )
            self.assertEqual(
                server_envs["reference"]["SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD"], "0"
            )
            self.assertEqual(
                server_envs["reference"]["SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA"], "0"
            )
            for phase_name in ("functional", "performance"):
                self.assertEqual(
                    server_envs[phase_name]["SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD"],
                    "1",
                )
                self.assertEqual(
                    server_envs[phase_name]["SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA"],
                    "1",
                )
            self.assertEqual(
                summary["observed_decode_batch"]["max_running_requests"], 2
            )
            self.assertEqual(sequence.count("flush"), 11)
            for index, item in enumerate(sequence):
                if item != "flush":
                    self.assertGreater(index, 0)
                    self.assertEqual(sequence[index - 1], "flush")

    def test_observed_decode_batch_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decode.log"
            path.write_text(
                "Decode batch [9], #running-req: 7, #token: 123\nDecode batch, #running-req: 16, #token: 321\n"
            )
            self.assertEqual(
                runner.observed_decode_batch(path, 16)["max_running_requests"], 16
            )
            with self.assertRaises(ValueError):
                runner.observed_decode_batch(path, 17)
            path.write_text("no decode batch\n")
            with self.assertRaises(ValueError):
                runner.observed_decode_batch(path, 1)

    def test_entrypoint_records_failure_and_exitcode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            output = root / "runs"
            path = Path(__file__).with_name("run.py")
            bootstrap = "import runpy,sys; sys.modules['memfabric_hybrid']=None; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
            command = [
                sys.executable,
                "-c",
                bootstrap,
                str(path),
                "decode",
                "--model",
                str(model),
                "--prefill-host",
                "127.0.0.2",
                "--decode-host",
                "127.0.0.1",
                "--devices",
                "0,1",
                "--interface",
                "lo",
                "--tp",
                "2",
                "--output-dir",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            directory = next(output.iterdir())
            self.assertEqual((directory / "exitcode.txt").read_text().strip(), "1")
            self.assertIn(
                "memfabric_hybrid", (directory / "controller.log").read_text()
            )

    def test_real_sse_done_and_timing(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for obj in (
                    {"output_ids": [3], "meta_info": {}},
                    {
                        "output_ids": [3, 4],
                        "meta_info": {
                            "prompt_tokens": 1,
                            "completion_tokens": 2,
                            "cached_tokens": 0,
                            "finish_reason": {"type": "length"},
                        },
                    },
                ):
                    self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                    self.wfile.flush()
                if payload["rid"] != "no-done":
                    self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.object(runner, "EVENT_DIR", Path(tmp)),
            ):
                result = runner.generate(
                    f"http://127.0.0.1:{server.server_port}", [1], 2, "r"
                )
                with gzip.open(
                    result["events_path"], "rt", encoding="utf-8"
                ) as recorded:
                    events = [json.loads(line) for line in recorded]
                self.assertEqual(events[-1]["data"], "[DONE]")
            runner.validate([result], 1, 2)
            self.assertEqual(result["response"]["output_ids"], [3, 4])
            self.assertGreaterEqual(result["ttft_seconds"], 0)
            self.assertGreaterEqual(result["tpot_seconds"], 0)
            with patch.object(runner, "EVENT_DIR", None):
                missing = runner.generate(
                    f"http://127.0.0.1:{server.server_port}", [1], 2, "no-done"
                )
            self.assertIn("without [DONE]", missing["error"])
            with self.assertRaises(ValueError):
                runner.validate_cached([result], 63872)
            result["response"]["meta_info"]["finish_reason"] = {"type": "abort"}
            with self.assertRaises(ValueError):
                runner.validate([result], 1, 2)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_real_http_errors_are_recorded(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"unavailable")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = runner.generate(
                f"http://127.0.0.1:{server.server_port}", [1], 1, "r"
            )
            self.assertEqual(result["error"], "HTTP 503")
            self.assertEqual(result["body"], "unavailable")
            with self.assertRaises(RuntimeError):
                runner.validate([result], 1, 1)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_bundled_fixture_hash_and_token_range(self):
        tokenizer = types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: range(154856)
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                sys.modules,
                {"transformers": types.SimpleNamespace(AutoTokenizer=tokenizer)},
            ),
        ):
            args = types.SimpleNamespace(token_ids_json=None, model="unused")
            ids = runner.fixture(args, Path(tmp))
            self.assertEqual(len(ids), 64000)
            self.assertEqual((min(ids), max(ids)), (1, 154855))

    def test_validation_rejects_errors_and_wrong_lengths(self):
        with self.assertRaises(RuntimeError):
            runner.validate([{"request_id": "r", "error": "HTTP 500"}], 1, 1)
        with self.assertRaises(ValueError):
            runner.validate(
                [
                    {
                        "request_id": "r",
                        "response": {
                            "output_ids": [],
                            "meta_info": {"prompt_tokens": 1, "completion_tokens": 1},
                        },
                    }
                ],
                1,
                1,
            )
        runner.validate(
            [
                {
                    "request_id": "r",
                    "response": {
                        "output_ids": [1],
                        "meta_info": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "finish_reason": {"type": "length"},
                        },
                    },
                }
            ],
            1,
            1,
        )

    def test_full_sequence_keeps_seven_rounds_and_checks_retraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            output = Path(tmp) / "runs"
            calls = []
            commands = {}
            phase = {"value": None}
            generated_inputs = []

            def start(command, env, log_path):
                name = Path(log_path).name
                calls.append(name)
                commands[name] = command
                if name.startswith("decode-"):
                    phase["value"] = name.removeprefix("decode-").removesuffix(".log")
                    if phase["value"] == "performance":
                        Path(log_path).write_text(
                            "Decode batch, #running-req: 2, #token: 1\n"
                        )
                return object()

            def generate(router, ids, length, rid):
                if phase["value"] in ("reference", "functional"):
                    generated_inputs.append((phase["value"], len(ids), length))
                time.sleep(0.02)
                return {
                    "request_id": rid,
                    "response": {
                        "output_ids": [7] * length,
                        "meta_info": {
                            "prompt_tokens": len(ids),
                            "completion_tokens": length,
                            "cached_tokens": 63872,
                            "num_retractions": int(phase["value"] == "functional"),
                            "finish_reason": {"type": "length"},
                        },
                    },
                    "elapsed_seconds": 1,
                }

            argv = [
                "run.py",
                "decode",
                "--model",
                str(model),
                "--prefill-host",
                "127.0.0.2",
                "--decode-host",
                "127.0.0.1",
                "--devices",
                "0,1",
                "--interface",
                "lo",
                "--tp",
                "2",
                "--bs",
                "2",
                "--client",
                "stdlib",
                "--output-dir",
                str(output),
            ]
            original_stdout, original_stderr = sys.stdout, sys.stderr
            try:
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(
                        runner, "runtime_info", return_value={"visible_npu_count": 2}
                    ),
                    patch.object(runner, "check_split_abi", return_value={}),
                    patch.object(runner, "fixture", return_value=[1] * 64000),
                    patch.object(runner, "wait_http"),
                    patch.object(runner, "port_free"),
                    patch.object(runner, "flush"),
                    patch.object(runner, "start", side_effect=start),
                    patch.object(runner, "stop"),
                    patch.object(runner, "generate", side_effect=generate),
                ):
                    runner.main()
            finally:
                if isinstance(sys.stdout, runner.Tee):
                    sys.stdout.log.close()
                sys.stdout, sys.stderr = original_stdout, original_stderr
            directory = next(output.iterdir())
            self.assertEqual(
                calls,
                [
                    "decode-reference.log",
                    "router-reference.log",
                    "decode-functional.log",
                    "router-functional.log",
                    "decode-performance.log",
                    "router-performance.log",
                ],
            )
            self.assertCountEqual(
                generated_inputs,
                [
                    (phase_name, input_length, 64)
                    for phase_name in ("reference", "functional")
                    for input_length in (512, 512, 512, 512, 4096, 4096, 4096, 4096)
                ],
            )
            for name in (
                "reference-short.json",
                "reference-long.json",
                "functional-short.json",
                "functional-long.json",
            ):
                report = json.loads((directory / name).read_text())
                self.assertEqual(len(report["results"]), 4)
                expected_prompt_length = 512 if "short" in name else 4096
                self.assertTrue(
                    all(
                        item["response"]["meta_info"]["prompt_tokens"]
                        == expected_prompt_length
                        for item in report["results"]
                    )
                )
            self.assertEqual(len(list(directory.glob("warm-[12].json"))), 2)
            self.assertEqual(len(list(directory.glob("formal-[1-5].json"))), 5)
            self.assertEqual(len(list(directory.glob("*-seed.json"))), 7)
            for name in (
                "decode-reference.log",
                "decode-functional.log",
                "decode-performance.log",
            ):
                self.assertIn("--disaggregation-decode-extra-slots", commands[name])
                self.assertEqual(
                    commands[name][
                        commands[name].index("--disaggregation-decode-extra-slots") + 1
                    ],
                    "0",
                )
            self.assertIn(
                "--disaggregation-decode-enable-radix-cache",
                commands["decode-performance.log"],
            )
            self.assertNotIn(
                "--disaggregation-decode-enable-radix-cache",
                commands["decode-functional.log"],
            )
            self.assertIn(
                "--disaggregation-decode-retraction-backup",
                commands["decode-functional.log"],
            )
            for item in directory.glob("formal-[1-5].json"):
                self.assertEqual(len(json.loads(item.read_text())["results"]), 2)


if __name__ == "__main__":
    unittest.main()
