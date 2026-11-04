"""CPU-only checks of client validation and the automatic phase sequence."""

import errno
import gzip
import hashlib
import importlib.util
import io
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
    @staticmethod
    def precision_fixture_stub():
        sizes = {"short": (512, 4), "long": (4096, 4), "target64k": (64000, 2)}
        cases = {
            name: {
                "input_ids": [index + 1] * count,
                "requests": requests,
                "token_count": count,
                "gzip_sha256": f"{index + 1:064x}",
                "raw_sha256": f"{index + 11:064x}",
            }
            for index, (name, (count, requests)) in enumerate(sizes.items())
        }
        manifest = {
            "output_tokens": 64,
            "answer": "731954",
            "scope": "Fixed retrieval case check; not a full model benchmark accuracy evaluation.",
            "cases": {
                name: {"token_count": count, "requests": requests}
                for name, (count, requests) in sizes.items()
            },
        }
        return {"manifest": manifest, "cases": cases}

    def test_router_waits_for_both_healthy_exact_role_workers(self):
        expected_prefill = "http://127.0.0.1:41000"
        expected_decode = "http://127.0.0.1:30001"
        snapshots = [
            {"workers": [], "total": 0, "stats": {}},
            {
                "workers": [
                    {
                        "url": expected_prefill,
                        "worker_type": "prefill",
                        "is_healthy": True,
                    },
                    {
                        "url": expected_decode,
                        "worker_type": "prefill",
                        "is_healthy": True,
                    },
                    {
                        "url": "http://127.0.0.9:30001",
                        "worker_type": "decode",
                        "is_healthy": True,
                    },
                    {
                        "url": expected_decode,
                        "worker_type": "decode",
                        "is_healthy": False,
                    },
                ],
                "total": 4,
                "stats": {"prefill_count": 2, "decode_count": 2},
            },
            {
                "workers": [
                    {
                        "url": expected_prefill,
                        "worker_type": "prefill",
                        "is_healthy": True,
                    },
                    {
                        "url": expected_decode,
                        "worker_type": "decode",
                        "is_healthy": True,
                    },
                ],
                "total": 2,
                "stats": {"prefill_count": 1, "decode_count": 1},
            },
        ]

        class Handler(BaseHTTPRequestHandler):
            polls = 0

            def do_GET(self):
                self.__class__.polls += 1
                payload = snapshots[min(self.__class__.polls - 1, len(snapshots) - 1)]
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = runner.wait_router_workers(
                f"http://127.0.0.1:{server.server_port}",
                expected_prefill,
                expected_decode,
                process=None,
                timeout=5,
            )
            self.assertEqual(result, snapshots[-1])
            self.assertGreaterEqual(Handler.polls, 3)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_router_worker_timeout_includes_last_snapshot(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(
                    {
                        "workers": [
                            {
                                "url": "http://127.0.0.1:41000",
                                "worker_type": "prefill",
                                "is_healthy": True,
                            }
                        ],
                        "total": 1,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(
                TimeoutError, r"last_workers=.*127\.0\.0\.1:41000"
            ):
                runner.wait_router_workers(
                    f"http://127.0.0.1:{server.server_port}",
                    "http://127.0.0.1:41000",
                    "http://127.0.0.1:30001",
                    process=None,
                    timeout=0,
                )
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_router_exit_before_worker_registration_is_reported(self):
        process = types.SimpleNamespace(poll=lambda: 1, returncode=1)
        with self.assertRaisesRegex(RuntimeError, "Router exited.*last_workers=None"):
            runner.wait_router_workers(
                "http://127.0.0.1:1",
                "http://127.0.0.1:41000",
                "http://127.0.0.1:30001",
                process=process,
                timeout=0,
            )

    @staticmethod
    def proc_stat(state, session_id=123, birth="101", flags="4194304"):
        fields = [state, "1", "123", str(session_id)] + ["0"] * 16
        fields[6] = flags
        fields[19] = birth
        return fields

    def make_proc_fixture(self, root, environment=b""):
        process_path = Path(root) / "123"
        process_path.mkdir()
        (process_path / "environ").write_bytes(environment)
        return process_path

    def make_stop_process(self):
        return types.SimpleNamespace(
            pid=123,
            run_token="owned-token",
            start_ticks="100",
            log_file=io.StringIO(),
            poll=lambda: 0,
            returncode=0,
        )

    def test_unknown_live_member_waits_for_delayed_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            process_path = self.make_proc_fixture(Path(tmp))
            process = self.make_stop_process()

            class FakeClock:
                now = 0.0
                sleeps = 0

                def sleep(self, delay):
                    self.now += delay
                    self.sleeps += 1
                    if self.sleeps == 1:
                        (process_path / "environ").write_bytes(
                            b"SPLITKV_RUN_TOKEN=owned-token\0"
                        )

            clock = FakeClock()
            with (
                patch.object(
                    runner, "_read_proc_stat", return_value=self.proc_stat("S")
                ),
                patch.object(runner.time, "monotonic", side_effect=lambda: clock.now),
                patch.object(runner.time, "sleep", side_effect=clock.sleep),
            ):
                self.assertTrue(
                    runner._wait_for_owned_token_or_exit(
                        process_path, process, "101", 123, timeout=1.0
                    )
                )
            self.assertEqual(clock.sleeps, 1)

    def test_unknown_live_member_times_out_with_proc_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            process_path = self.make_proc_fixture(Path(tmp))
            process = self.make_stop_process()

            class FakeClock:
                now = 0.0

                def sleep(self, delay):
                    self.now += delay

            clock = FakeClock()
            with (
                patch.object(
                    runner, "_read_proc_stat", return_value=self.proc_stat("S")
                ),
                patch.object(runner.time, "monotonic", side_effect=lambda: clock.now),
                patch.object(runner.time, "sleep", side_effect=clock.sleep),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"after 0.2s; state='S',ppid='1',pgrp='123',session='123',"
                    r"birth='101',flags='4194304'; environ_error=None",
                ):
                    runner._wait_for_owned_token_or_exit(
                        process_path, process, "101", 123, timeout=0.2
                    )

    def test_unknown_member_that_exits_or_changes_identity_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            process_path = self.make_proc_fixture(Path(tmp))
            process = self.make_stop_process()
            for stat in (self.proc_stat("Z"), self.proc_stat("S", birth="999")):
                with (
                    self.subTest(stat=stat[:1] + stat[19:20]),
                    patch.object(runner, "_read_proc_stat", return_value=stat),
                ):
                    self.assertFalse(
                        runner._wait_for_owned_token_or_exit(
                            process_path, process, "101", 123, timeout=1.0
                        )
                    )

    def test_stop_skips_process_that_turns_zombie_while_reading_empty_environ(self):
        with tempfile.TemporaryDirectory() as tmp:
            process_path = self.make_proc_fixture(Path(tmp))
            states = iter(("S", "Z", "Z"))
            process = self.make_stop_process()
            with (
                patch.object(runner, "PROC_ROOT", Path(tmp)),
                patch.object(runner.signal, "SIGKILL", 9, create=True),
                patch.object(
                    runner,
                    "_read_proc_stat",
                    side_effect=lambda path: self.proc_stat(next(states)),
                ),
            ):
                runner.stop(process)
            self.assertTrue(process.log_file.closed)
            self.assertTrue((process_path / "environ").exists())

    def test_stop_still_rejects_live_same_session_without_owner_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_proc_fixture(Path(tmp))
            process = self.make_stop_process()
            with (
                patch.object(runner, "PROC_ROOT", Path(tmp)),
                patch.object(runner.signal, "SIGKILL", 9, create=True),
                patch.object(
                    runner,
                    "_read_proc_stat",
                    side_effect=lambda path: self.proc_stat("S"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "run token missing"):
                    runner.stop(process)
            self.assertTrue(process.log_file.closed)

    def test_stop_waits_for_pf_exiting_without_signalling_until_zombie(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_proc_fixture(Path(tmp))
            exiting = self.proc_stat("S", flags="4195652")
            states = iter(
                (
                    exiting,
                    exiting,
                    exiting,
                    self.proc_stat("Z", flags="4195652"),
                    self.proc_stat("Z", flags="4195652"),
                    self.proc_stat("Z", flags="4195652"),
                )
            )
            process = self.make_stop_process()

            class FakeClock:
                now = 0.0
                sleeps = 0

                def sleep(self, delay):
                    self.now += delay
                    self.sleeps += 1

            clock = FakeClock()
            with (
                patch.object(runner, "PROC_ROOT", Path(tmp)),
                patch.object(runner.signal, "SIGKILL", 9, create=True),
                patch.object(
                    runner,
                    "_read_proc_stat",
                    side_effect=lambda path: next(states),
                ),
                patch.object(runner.time, "monotonic", side_effect=lambda: clock.now),
                patch.object(runner.time, "sleep", side_effect=clock.sleep),
                patch.object(runner.os, "kill") as kill,
            ):
                runner.stop(process)
            kill.assert_not_called()
            self.assertEqual(clock.sleeps, 1)
            self.assertTrue(process.log_file.closed)

    def test_stop_rechecks_state_before_signalling_zombie(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_proc_fixture(Path(tmp), b"SPLITKV_RUN_TOKEN=owned-token\0")
            states = iter(("S", "S", "Z", "Z", "Z", "Z"))
            process = self.make_stop_process()
            with (
                patch.object(runner, "PROC_ROOT", Path(tmp)),
                patch.object(runner.signal, "SIGKILL", 9, create=True),
                patch.object(
                    runner,
                    "_read_proc_stat",
                    side_effect=lambda path: self.proc_stat(next(states)),
                ),
                patch.object(runner.os, "kill") as kill,
            ):
                runner.stop(process)
            kill.assert_not_called()

    def test_stop_signals_owned_token_child_in_a_different_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            child_path = Path(tmp) / "234"
            child_path.mkdir()
            (child_path / "environ").write_bytes(b"SPLITKV_RUN_TOKEN=owned-token\0")
            state = {"count": 0}
            process = self.make_stop_process()

            def read_stat(path):
                state["count"] += 1
                status = "S" if state["count"] <= 5 else "Z"
                return self.proc_stat(status, session_id=999)

            with (
                patch.object(runner, "PROC_ROOT", Path(tmp)),
                patch.object(runner.signal, "SIGKILL", 9, create=True),
                patch.object(
                    runner,
                    "_read_proc_stat",
                    side_effect=read_stat,
                ),
                patch.object(runner.os, "kill") as kill,
            ):
                runner.stop(process)
            kill.assert_called_once_with(234, runner.signal.SIGTERM)

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

    def test_invalid_decode_graph_shape_fails_before_runtime_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            output = root / "runs"
            command = [
                sys.executable,
                str(Path(__file__).with_name("run.py")),
                "decode",
                "--model",
                str(model),
                "--prefill-host",
                "127.0.0.2",
                "--decode-host",
                "127.0.0.1",
                "--devices",
                ",".join(str(index) for index in range(16)),
                "--interface",
                "lo",
                "--tp",
                "16",
                "--bs",
                "5",
                "--output-dir",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("decode graph capture has no valid bucket", result.stderr)
            self.assertIn("--graph-bs * 4 must be divisible by --tp", result.stderr)
            self.assertFalse(output.exists())

    def test_graph_bs_must_cover_runtime_batch_before_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            output = root / "runs"
            command = [
                sys.executable,
                str(Path(__file__).with_name("run.py")),
                "decode",
                "--model",
                str(model),
                "--prefill-host",
                "127.0.0.2",
                "--decode-host",
                "127.0.0.1",
                "--devices",
                ",".join(str(index) for index in range(16)),
                "--interface",
                "lo",
                "--tp",
                "16",
                "--bs",
                "6",
                "--graph-bs",
                "5",
                "--output-dir",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("graph-bs must be greater than or equal to bs", result.stderr)
            self.assertFalse(output.exists())

    def test_server_env_records_deepep_dispatch_limit_when_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            runner.record_server_env(output, "decode", {})
            recorded = json.loads((output / "env-decode.json").read_text())
        self.assertIsNone(recorded["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"])

    def test_native_kv_mode_disables_sparse_and_lim_env_for_every_phase(self):
        args = types.SimpleNamespace(
            role="decode",
            prefill_host="127.0.0.2",
            decode_host="127.0.0.1",
            devices="0,1",
            store_port=24669,
            transfer_protocol="device_sdma",
            interface="lo",
            kv_mode="native",
        )
        sparse_env = {
            "SGLANG_NPU_SPARSE_KV_INDEXER": "lim",
            "SGLANG_NPU_SPARSE_KV_CACHE_SIZE": "12288",
            "SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB": "512",
            "SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE": "1",
            "SGLANG_NPU_MTP_SEPARATE_KV_TAIL": "0",
            "SGLANG_NPU_FUSED_LIM_PRE_METADATA": "1",
            "SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA": "1",
        }
        with patch.dict("os.environ", sparse_env):
            for functional, reference in ((False, True), (True, False), (False, False)):
                env = runner.server_env(
                    args, functional=functional, reference=reference
                )
                self.assertEqual(env["SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD"], "0")
                self.assertEqual(env["SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA"], "0")
                self.assertTrue(all(name not in env for name in sparse_env))
                self.assertEqual(
                    env.get("SGLANG_TEST_RETRACT"), "1" if functional else None
                )

    def test_server_command_uses_graph_bs_and_preserves_runtime_bs(self):
        args = types.SimpleNamespace(
            model="/model",
            model_name="test-model",
            tp=16,
            mem_fraction=0.92,
            graph_bs=8,
            bs=6,
            max_total_tokens=163840,
            prefill_port=41000,
            decode_port=30001,
            bootstrap_port=8998,
            disable_decode_prefix_cache=False,
        )
        command = runner.server_command(args, "decode")
        self.assertEqual(command[command.index("--cuda-graph-bs-decode") + 1], "8")
        self.assertEqual(command[command.index("--max-running-requests") + 1], "6")

    def test_default_frozen_controller_runs_all_rounds_after_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            output = Path(tmp) / "runs"
            sequence = []
            phase = {"value": None}
            server_envs = {}
            performance_fixture_inputs = []

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
                        "text": "731954\n" * 12 + "731954",
                        "meta_info": {
                            "prompt_tokens": len(ids),
                            "completion_tokens": length,
                            "cached_tokens": 1,
                            "finish_reason": {"type": "length"},
                            "num_retractions": int(phase["value"] == "functional"),
                        },
                    },
                    "elapsed_seconds": 1,
                }

            def frozen_round(router, model_path, out, ids, bs, label):
                sequence.append(label)
                performance_fixture_inputs.append(ids)
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
                    patch.object(
                        runner,
                        "precision_fixtures",
                        return_value=self.precision_fixture_stub(),
                    ),
                    patch.object(runner, "wait_http"),
                    patch.object(runner, "wait_router_workers"),
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
            self.assertEqual(
                summary["performance_decode_prefix_cache_accuracy"],
                {
                    "status": "passed",
                    "cache_mode": "enabled",
                    "requests": 2,
                    "answer": "731954",
                    "exact_match": True,
                    "cache_hit": True,
                    "passed": True,
                },
            )
            self.assertTrue(
                (next(output.iterdir()) / "performance-prefix-seed.json").is_file()
            )
            prefix_report = json.loads(
                (
                    next(output.iterdir()) / "performance-prefix-accuracy.json"
                ).read_text()
            )
            self.assertEqual(prefix_report["cache_mode"], "enabled")
            self.assertEqual(len(prefix_report["results"]), 2)
            self.assertTrue(prefix_report["passed"])
            self.assertEqual(len(performance_fixture_inputs), 7)
            self.assertTrue(
                all(ids == [1] * 64000 for ids in performance_fixture_inputs)
            )
            self.assertIn(
                "not a full model benchmark accuracy evaluation",
                summary["greedy_accuracy_validation"]["scope"],
            )
            self.assertEqual(
                summary["greedy_accuracy_validation"]["cases"],
                [
                    {"input_tokens": 512, "output_tokens": 64, "requests": 4},
                    {"input_tokens": 4096, "output_tokens": 64, "requests": 4},
                    {"input_tokens": 64000, "output_tokens": 64, "requests": 2},
                ],
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
            self.assertEqual(sequence.count("flush"), 14)
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

    def test_precision_fixtures_verify_hashes_lengths_vocab_and_copy_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            fixture_dir = root / "fixtures" / "precision"
            fixture_dir.mkdir(parents=True)
            tokenizer_files = {
                "tokenizer.json": b"tokenizer fixture",
                "tokenizer_config.json": b"tokenizer config fixture",
            }
            tokenizer_hashes = {}
            for name, contents in tokenizer_files.items():
                (model / name).write_bytes(contents)
                tokenizer_hashes[name] = hashlib.sha256(contents).hexdigest()
            cases = {}
            fixture_payloads = {}
            for name, token_count, requests in (
                ("short", 512, 4),
                ("long", 4096, 4),
                ("target64k", 64000, 2),
            ):
                filename = f"{name}.json.gz"
                raw = json.dumps(
                    {"input_ids": [1, 2] * (token_count // 2) + [1] * (token_count % 2)}
                ).encode()
                compressed = gzip.compress(raw)
                (fixture_dir / filename).write_bytes(compressed)
                cases[name] = {
                    "file": filename,
                    "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "token_count": token_count,
                    "requests": requests,
                }
                fixture_payloads[name] = compressed
            manifest = {
                "version": 1,
                "answer": "731954",
                "output_tokens": 64,
                "scope": "test scope",
                "tokenizer_files_sha256": tokenizer_hashes,
                "cases": cases,
            }
            (fixture_dir / "manifest.json").write_text(json.dumps(manifest))
            tokenizer = types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: range(10)
            )
            args = types.SimpleNamespace(model=str(model))
            with (
                patch.object(runner, "HERE", root),
                patch.dict(
                    sys.modules,
                    {"transformers": types.SimpleNamespace(AutoTokenizer=tokenizer)},
                ),
            ):
                loaded = runner.precision_fixtures(args, root / "output")
            self.assertEqual(
                {
                    name: len(case["input_ids"])
                    for name, case in loaded["cases"].items()
                },
                {"short": 512, "long": 4096, "target64k": 64000},
            )
            copied = root / "output" / "precision-fixtures"
            self.assertEqual(
                (copied / "manifest.json").read_bytes(),
                (fixture_dir / "manifest.json").read_bytes(),
            )
            for name, compressed in fixture_payloads.items():
                self.assertEqual(
                    (copied / cases[name]["file"]).read_bytes(), compressed
                )

    def test_precision_fixture_loader_rejects_tokenizer_hash_length_and_vocab(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            fixture_dir = root / "fixtures" / "precision"
            fixture_dir.mkdir(parents=True)
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (model / name).write_text(name)
            tokenizer = types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: range(5)
            )
            args = types.SimpleNamespace(model=str(model))
            with patch.dict(
                sys.modules,
                {"transformers": types.SimpleNamespace(AutoTokenizer=tokenizer)},
            ):
                for failure in ("tokenizer", "gzip", "raw", "length", "vocab"):
                    with self.subTest(failure=failure):
                        tokenizer_hashes = {
                            name: hashlib.sha256(
                                (model / name).read_bytes()
                            ).hexdigest()
                            for name in ("tokenizer.json", "tokenizer_config.json")
                        }
                        if failure == "tokenizer":
                            tokenizer_hashes["tokenizer.json"] = "0" * 64
                        case_specs = {
                            "short": (512, 4),
                            "long": (4096, 4),
                            "target64k": (64000, 2),
                        }
                        case_manifest = {}
                        for case_name, (count, request_count) in case_specs.items():
                            ids = [1] * count
                            if case_name == "short" and failure == "length":
                                ids = [1] * (count - 1)
                            elif case_name == "short" and failure == "vocab":
                                ids[0] = 5
                            raw = json.dumps({"input_ids": ids}).encode()
                            compressed = gzip.compress(raw)
                            filename = f"{case_name}.json.gz"
                            (fixture_dir / filename).write_bytes(compressed)
                            case_manifest[case_name] = {
                                "file": filename,
                                "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                                "token_count": count,
                                "requests": request_count,
                            }
                        if failure == "gzip":
                            case_manifest["short"]["gzip_sha256"] = "0" * 64
                        elif failure == "raw":
                            case_manifest["short"]["raw_sha256"] = "0" * 64
                        manifest = {
                            "version": 1,
                            "answer": "731954",
                            "output_tokens": 64,
                            "tokenizer_files_sha256": tokenizer_hashes,
                            "cases": case_manifest,
                        }
                        (fixture_dir / "manifest.json").write_text(json.dumps(manifest))
                        with patch.object(runner, "HERE", root):
                            with self.assertRaises(ValueError):
                                runner.precision_fixtures(args, root / "out")

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

    def test_retrieval_answer_requires_exact_repeated_line_prefix(self):
        valid_text = "731954\n" * 12 + "731954"
        for text in ("731954\n", valid_text):
            runner.validate_retrieval_answer(
                [{"request_id": "ok", "response": {"text": text}}], "731954"
            )
        for text in (
            "prefix 731954\n",
            "731954\n731954x",
            "x731954\n",
            "731954",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                runner.validate_retrieval_answer(
                    [{"request_id": "bad", "response": {"text": text}}],
                    "731954",
                )

    def test_prefix_accuracy_gate_rejects_wrong_answer_tokens_and_cache_miss(self):
        valid_result = {
            "request_id": "ok",
            "response": {
                "output_ids": [7] * 64,
                "text": "731954\n" * 12 + "731954",
                "meta_info": {"cached_tokens": 100},
            },
        }
        passed = runner.validate_prefix_cache_accuracy(
            [valid_result], [7] * 64, "731954"
        )
        self.assertEqual(
            passed, {"exact_match": True, "cache_hit": True, "cached_tokens": [100]}
        )
        wrong_answer = json.loads(json.dumps(valid_result))
        wrong_answer["response"]["text"] = "other\n" * 12 + "other"
        with self.assertRaisesRegex(ValueError, "valid repeated-answer prefix"):
            runner.validate_prefix_cache_accuracy([wrong_answer], [7] * 64, "731954")
        wrong_tokens = json.loads(json.dumps(valid_result))
        wrong_tokens["response"]["output_ids"] = [8] * 64
        with self.assertRaisesRegex(ValueError, "differ from the native"):
            runner.validate_prefix_cache_accuracy([wrong_tokens], [7] * 64, "731954")
        cache_miss = json.loads(json.dumps(valid_result))
        cache_miss["response"]["meta_info"]["cached_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "positive integer cached_tokens"):
            runner.validate_prefix_cache_accuracy([cache_miss], [7] * 64, "731954")

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
                    generated_inputs.append((phase["value"], len(ids), ids[0], length))
                time.sleep(0.02)
                return {
                    "request_id": rid,
                    "response": {
                        "output_ids": [7] * length,
                        "text": "731954\n" * 12 + "731954",
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
                    patch.object(
                        runner,
                        "precision_fixtures",
                        return_value=self.precision_fixture_stub(),
                    ),
                    patch.object(runner, "wait_http"),
                    patch.object(runner, "wait_router_workers"),
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
                    (phase_name, input_length, first_id, 64)
                    for phase_name in ("reference", "functional")
                    for input_length, first_id, count in (
                        (512, 1, 4),
                        (4096, 2, 4),
                        (64000, 3, 2),
                    )
                    for _ in range(count)
                ],
            )
            self.assertEqual(
                json.loads(
                    (directory / "performance-prefix-accuracy.json").read_text()
                )["passed"],
                True,
            )
            for name, expected_count in (
                ("reference-short.json", 4),
                ("reference-long.json", 4),
                ("reference-target64k.json", 2),
                ("functional-short.json", 4),
                ("functional-long.json", 4),
                ("functional-target64k.json", 2),
            ):
                report = json.loads((directory / name).read_text())
                self.assertEqual(len(report["results"]), expected_count)
                expected_prompt_length = {
                    "short": 512,
                    "long": 4096,
                    "target64k": 64000,
                }[name.split("-")[1].removesuffix(".json")]
                self.assertTrue(
                    all(
                        item["response"]["meta_info"]["prompt_tokens"]
                        == expected_prompt_length
                        for item in report["results"]
                    )
                )
                self.assertIs(report["answer_validation"]["passed"], True)
                outputs = [item["response"]["output_ids"] for item in report["results"]]
                if name.startswith("reference-"):
                    self.assertTrue(all(output == outputs[0] for output in outputs))
                else:
                    reference_name = name.replace("functional-", "reference-", 1)
                    reference_report = json.loads(
                        (directory / reference_name).read_text()
                    )
                    reference_output = reference_report["results"][0]["response"][
                        "output_ids"
                    ]
                    self.assertTrue(
                        all(output == reference_output for output in outputs)
                    )
                    self.assertGreater(
                        sum(
                            item["response"]["meta_info"].get("num_retractions", 0)
                            for item in report["results"]
                        ),
                        0,
                    )
            self.assertEqual(len(list(directory.glob("warm-[12].json"))), 2)
            self.assertEqual(len(list(directory.glob("formal-[1-5].json"))), 5)
            self.assertEqual(len(list(directory.glob("*-seed.json"))), 8)
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
