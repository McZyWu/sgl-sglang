"""Run a two-host split-KV PD experiment from within the two containers."""

import argparse
import concurrent.futures
import errno
import gzip
import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROC_ROOT = Path("/proc")
# `/proc/<pid>/stat` field 9 exposes PF_EXITING in its decimal bitmask.
PF_EXITING = 0x00000004
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
EVENT_DIR = None
RUN_OUT = None
INHERITED_ENV_TO_CLEAR = (
    "RANK_TABLE_FILE",
    "HCCL_IF_IP",
    "HCCL_SOCKET_FAMILY",
    "ASCEND_LAUNCH_BLOCKING",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


def request(url, payload=None, timeout=7200):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"} if body else {}
    )
    with OPENER.open(req, timeout=timeout) as response:
        return response.read().decode()


def wait_http(url, process=None, timeout=3600):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Process exited {process.returncode} while waiting for {url}"
            )
        try:
            request(url, timeout=5)
            return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def wait_router_workers(
    router, expected_prefill, expected_decode, process, timeout=3600
):
    deadline = time.monotonic() + timeout
    last_snapshot = None
    last_error = None
    expected = {
        "prefill": expected_prefill.rstrip("/"),
        "decode": expected_decode.rstrip("/"),
    }
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "Router exited before both healthy PD workers registered: "
                f"exit={process.returncode}; last_workers={last_snapshot!r}; "
                f"last_error={last_error!r}"
            )
        try:
            request_timeout = max(0.1, min(5.0, max(0.0, deadline - time.monotonic())))
            payload = json.loads(
                request(router.rstrip("/") + "/workers", timeout=request_timeout)
            )
            last_snapshot = payload
            workers = payload.get("workers") if isinstance(payload, dict) else None
            if not isinstance(workers, list):
                raise ValueError("/workers response must contain a workers list")
            healthy = {
                role: any(
                    isinstance(worker, dict)
                    and isinstance(worker.get("url"), str)
                    and worker["url"].rstrip("/") == expected_url
                    and worker.get("worker_type") == role
                    and worker.get("is_healthy") is True
                    for worker in workers
                )
                for role, expected_url in expected.items()
            }
            if all(healthy.values()):
                return payload
            last_error = (
                f"expected one healthy prefill and decode worker; matched={healthy}"
            )
        except (OSError, urllib.error.URLError, ValueError, TypeError) as error:
            last_error = repr(error)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for both healthy PD router workers at "
                f"{router.rstrip('/')}/workers; last_workers={last_snapshot!r}; "
                f"last_error={last_error!r}"
            )
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _read_proc_stat(path):
    return (path / "stat").read_text().rsplit(") ", 1)[1].split()


def _same_live_session(path, expected_birth, session_id):
    try:
        stat = _read_proc_stat(path)
    except (FileNotFoundError, ProcessLookupError):
        return False
    return (
        len(stat) > 19
        and stat[0] not in ("Z", "X")
        and stat[3] == str(session_id)
        and stat[19] == expected_birth
    )


def _proc_is_exiting(stat):
    return len(stat) > 6 and bool(int(stat[6]) & PF_EXITING)


def _proc_identity_evidence(path, stat):
    flags = stat[6] if len(stat) > 6 else "unavailable"
    return (
        f"state={stat[0]!r},ppid={stat[1]!r},pgrp={stat[2]!r},"
        f"session={stat[3]!r},birth={stat[19]!r},flags={flags!r}"
    )


def _wait_for_owned_token_or_exit(
    path, proc, expected_birth, expected_session, timeout=1.0
):
    token = f"SPLITKV_RUN_TOKEN={proc.run_token}".encode()
    deadline = time.monotonic() + timeout
    last_stat = None
    last_environment_error = None
    while True:
        try:
            stat = _read_proc_stat(path)
        except (FileNotFoundError, ProcessLookupError):
            return False
        if len(stat) <= 19:
            return False
        if (
            stat[19] != expected_birth
            or stat[3] != str(expected_session)
            or stat[0] in ("Z", "X")
        ):
            return False
        last_stat = stat
        if _proc_is_exiting(stat):
            return "exiting"
        try:
            environment = (path / "environ").read_bytes().split(b"\0")
            last_environment_error = None
            if token in environment:
                return True
        except PermissionError as error:
            last_environment_error = repr(error)
        except (FileNotFoundError, ProcessLookupError) as error:
            last_environment_error = repr(error)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.1, remaining))

    evidence = _proc_identity_evidence(path, last_stat)
    raise RuntimeError(
        f"Cannot verify ownership of session member {path.name}: run token "
        f"missing after {timeout:.1f}s; {evidence}; "
        f"environ_error={last_environment_error}"
    )


def port_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as error:
            raise RuntimeError(f"Local port {port} is unavailable: {error}") from error


def validate_host_addresses(prefill_host, decode_host, role):
    """Validate IPv4 CLI addresses and confirm this container can bind its own IP."""
    addresses = {}
    for option, value in (
        ("--prefill-host", prefill_host),
        ("--decode-host", decode_host),
    ):
        try:
            address = ipaddress.IPv4Address(value)
        except ipaddress.AddressValueError as error:
            raise ValueError(
                f"{option} must be a full IPv4 address (got {value!r}); "
                "hostnames, abbreviated addresses, ports, and IPv6 are unsupported"
            ) from error
        if address.is_unspecified or address.is_multicast or int(address) == 0xFFFFFFFF:
            raise ValueError(
                f"{option} must be a usable unicast IPv4 address (got {value!r})"
            )
        addresses[option] = address

    if addresses["--prefill-host"] == addresses["--decode-host"]:
        raise ValueError("two distinct host addresses are required")

    local_option = "--prefill-host" if role == "prefill" else "--decode-host"
    local_address = addresses[local_option]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((str(local_address), 0))
    except OSError as error:
        reason = error.strerror or str(error)
        errno_name = getattr(error, "errno", None)
        errno_label = (
            errno.errorcode.get(errno_name, str(errno_name))
            if errno_name
            else "unknown errno"
        )
        raise ValueError(
            f"{local_option} {local_address} cannot be bound in this container "
            f"({errno_label}: {reason}). Check `ip -br -4 addr` inside this "
            "container and use an address assigned in its network namespace."
        ) from error


def start(command, env, log_path):
    print("Starting:", subprocess.list2cmdline(command), "log:", log_path, flush=True)
    log = open(log_path, "w", encoding="utf-8", buffering=1)
    try:
        env = env.copy()
        env["SPLITKV_RUN_TOKEN"] = uuid.uuid4().hex
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log.close()
        raise
    proc.log_file = log
    proc.run_token = env["SPLITKV_RUN_TOKEN"]
    try:
        proc.start_ticks = (
            Path(f"/proc/{proc.pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
        )
    except (FileNotFoundError, ProcessLookupError) as error:
        proc.start_ticks = "0"
        stop(proc)
        raise RuntimeError(
            f"Process {proc.pid} exited before its identity could be recorded"
        ) from error
    return proc


def stop(proc):
    if proc is None:
        return

    tracked_owned_members = set()
    exiting_members = set()

    def members():
        found = []
        exiting_members.clear()
        for path in PROC_ROOT.iterdir():
            if not path.name.isdigit():
                continue
            try:
                stat = _read_proc_stat(path)
                if (
                    len(stat) <= 19
                    or stat[0] in ("Z", "X")
                    or int(stat[19]) < int(proc.start_ticks)
                ):
                    continue
                pid = int(path.name)
                identity = (pid, stat[19], stat[3])
                if _proc_is_exiting(stat):
                    if stat[3] == str(proc.pid) or identity in tracked_owned_members:
                        tracked_owned_members.add(identity)
                        exiting_members.add(identity)
                        found.append(identity)
                    continue
                same_leader_session = _same_live_session(path, stat[19], proc.pid)
                try:
                    environment = (path / "environ").read_bytes().split(b"\0")
                except PermissionError:
                    if same_leader_session:
                        raise RuntimeError(
                            f"Cannot verify ownership of session member {path.name}"
                        )
                    continue
                if f"SPLITKV_RUN_TOKEN={proc.run_token}".encode() in environment:
                    tracked_owned_members.add(identity)
                    found.append(identity)
                elif same_leader_session:
                    token_state = _wait_for_owned_token_or_exit(
                        path, proc, stat[19], proc.pid
                    )
                    if token_state is True:
                        tracked_owned_members.add(identity)
                        found.append(identity)
                    elif token_state == "exiting":
                        tracked_owned_members.add(identity)
                        exiting_members.add(identity)
                        found.append(identity)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return found

    try:
        for sig, seconds in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            current = members()
            if not current:
                break
            print(
                f"Stopping owned session={proc.pid} signal={sig.name} members={[p for p, _, _ in current]}",
                flush=True,
            )
            for pid, birth, session_id in current:
                path = PROC_ROOT / str(pid)
                try:
                    stat = _read_proc_stat(path)
                    if (
                        len(stat) <= 19
                        or stat[0] in ("Z", "X")
                        or stat[3] != session_id
                        or stat[19] != birth
                    ):
                        continue
                    if _proc_is_exiting(stat):
                        exiting_members.add((pid, birth, session_id))
                        continue
                    env = (path / "environ").read_bytes().split(b"\0")
                    if (
                        f"SPLITKV_RUN_TOKEN={proc.run_token}".encode() in env
                        and _same_live_session(path, birth, session_id)
                    ):
                        current_stat = _read_proc_stat(path)
                        if (
                            len(current_stat) > 19
                            and current_stat[0] not in ("Z", "X")
                            and current_stat[3] == session_id
                            and current_stat[19] == birth
                            and not _proc_is_exiting(current_stat)
                        ):
                            os.kill(pid, sig)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            deadline = time.monotonic() + seconds
            while members() and time.monotonic() < deadline:
                proc.poll()
                time.sleep(0.2)
        remaining = members()
        proc.poll()
        print(
            f"Owned leader pid={proc.pid} exit={proc.returncode}, remaining={remaining}",
            flush=True,
        )
        if remaining:
            pending_exit = [
                identity for identity in remaining if identity in exiting_members
            ]
            if pending_exit:
                raise RuntimeError(
                    "Owned processes remained in PF_EXITING after the cleanup "
                    f"grace periods: {pending_exit}"
                )
            raise RuntimeError(f"Could not stop owned processes: {remaining}")
    finally:
        proc.log_file.close()


def server_env(args, functional=False, reference=False):
    env = os.environ.copy()
    for name in INHERITED_ENV_TO_CLEAR:
        env.pop(name, None)
    env.update(
        {
            "SGLANG_HOST_IP": args.prefill_host
            if args.role == "prefill"
            else args.decode_host,
            "ASCEND_RT_VISIBLE_DEVICES": args.devices,
            "ASCEND_MF_STORE_URL": f"tcp://{args.prefill_host}:{args.store_port}",
            "ASCEND_MF_TRANSFER_PROTOCOL": args.transfer_protocol,
            "HCCL_SOCKET_IFNAME": args.interface,
            "GLOO_SOCKET_IFNAME": args.interface,
            "MF_HYBM_USE_VMM_SEGMENT": "1",
            "SGLANG_SET_CPU_AFFINITY": "1",
            "STREAMS_PER_DEVICE": "32",
            "HCCL_BUFFSIZE": "1000",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1",
            "SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA": "1",
            "SGLANG_NPU_SPARSE_KV_INDEXER": "lim",
            "SGLANG_NPU_SPARSE_KV_CACHE_SIZE": "12288",
            "SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB": "0",
            "SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE": "1",
            "SGLANG_NPU_MTP_SEPARATE_KV_TAIL": "0",
            "SGLANG_NPU_FUSED_LIM_PRE_METADATA": "1",
            "SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA": "1",
            "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
            "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
            "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
            "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
            "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
            "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        }
    )
    for name in (
        "SGLANG_TEST_RETRACT",
        "SGLANG_TEST_RETRACT_INTERVAL",
        "SGLANG_NPU_SPARSE_KV_POLICY",
        "SGLANG_NPU_SPARSE_KV_CACHE_CAPACITY",
        "SGLANG_NPU_GVA_DRAM_SIZE_GB",
        "SGLANG_D2RH_VERIFY",
    ):
        env.pop(name, None)
    if functional:
        env.update(SGLANG_TEST_RETRACT="1", SGLANG_TEST_RETRACT_INTERVAL="3")
    if reference:
        env.update(
            SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD="0",
            SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA="0",
        )
    return env


def server_command(args, role, functional=False, performance=False):
    command = [
        sys.executable,
        str(HERE / "no_thp_launch.py"),
        "--model-path",
        args.model,
        "--served-model-name",
        args.model_name,
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--tp-size",
        str(args.tp),
        "--nnodes",
        "1",
        "--dp-size",
        "1",
        "--chunked-prefill-size",
        "1024",
        "--max-prefill-tokens",
        "8192",
        "--prefill-max-requests",
        "1",
        "--trust-remote-code",
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--cuda-graph-bs-decode",
        str(args.bs),
        "--max-running-requests",
        str(args.bs),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--context-length",
        "67584",
        "--page-size",
        "128",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--quantization",
        "modelslim",
        "--speculative-draft-model-quantization",
        "modelslim",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--random-seed",
        "42",
        "--enable-metrics",
        "--enable-cache-report",
        "--disaggregation-mode",
        role,
        "--disaggregation-transfer-backend",
        "ascend",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.prefill_port if role == "prefill" else args.decode_port),
        "--disaggregation-bootstrap-port",
        str(args.bootstrap_port),
    ]
    if role == "decode" and functional:
        command += ["--disaggregation-decode-retraction-backup", "cpu_tensor"]
    if role == "decode":
        command += ["--disaggregation-decode-extra-slots", "0"]
        if performance and not args.disable_decode_prefix_cache:
            command += ["--disaggregation-decode-enable-radix-cache"]
    return command


def dump(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def record_server_env(out, phase, env):
    names = (
        "ASCEND_RT_VISIBLE_DEVICES",
        "SGLANG_HOST_IP",
        "ASCEND_MF_STORE_URL",
        "ASCEND_MF_TRANSFER_PROTOCOL",
        "HCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "HCCL_BUFFSIZE",
        "HCCL_OP_EXPANSION_MODE",
        "STREAMS_PER_DEVICE",
        "SGLANG_SET_CPU_AFFINITY",
        "MF_HYBM_USE_VMM_SEGMENT",
        "MF_PYTHON_PATH",
        "MEMFABRIC_HYBRID_EXTEND_LIB_PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "ASCEND_HOME_PATH",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS",
        "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ",
        "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE",
        "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT",
        "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD",
        "SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA",
        "SGLANG_NPU_SPARSE_KV_INDEXER",
        "SGLANG_NPU_SPARSE_KV_CACHE_SIZE",
        "SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB",
        "SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE",
        "SGLANG_NPU_MTP_SEPARATE_KV_TAIL",
        "SGLANG_NPU_FUSED_LIM_PRE_METADATA",
        "SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA",
        "SGLANG_ENABLE_SPEC_V2",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM",
        "SGLANG_TEST_RETRACT",
        "SGLANG_TEST_RETRACT_INTERVAL",
        "KERNEL_PYTHON_PATH",
    )
    dump(out / f"env-{phase}.json", {name: env.get(name) for name in names})


def check_split_abi():
    try:
        import sgl_kernel_npu
        import torch_npu  # noqa: F401

        from sglang.kernels.ops.attention.npu_copy_sfa import load_copy_sfa_operator

        operator = load_copy_sfa_operator()
    except (ImportError, AttributeError, RuntimeError, OSError) as error:
        raise RuntimeError(
            "The split-K/V kernel wheel is missing or incompatible. Build and install the matching kernel with setup_kernel.sh, set KERNEL_PYTHON_PATH, and retry before loading the model."
        ) from error
    print(
        f"split Copy-SFA ABI: {operator._schema}; package={sgl_kernel_npu.__file__}",
        flush=True,
    )
    so = Path(sgl_kernel_npu.__file__).resolve().parent / "lib" / "libsgl_kernel_npu.so"
    if not so.is_file():
        raise RuntimeError(f"Could not find the loaded kernel shared library: {so}")
    return {
        "package": str(Path(sgl_kernel_npu.__file__).resolve()),
        "so": str(so),
        "so_sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
        "schema": str(operator._schema),
    }


def runtime_info(model):
    import memfabric_hybrid
    import torch
    import torch_npu

    import sglang

    try:
        mf_version = importlib.metadata.version("memfabric-hybrid")
    except importlib.metadata.PackageNotFoundError:
        mf_version = None
    framework_root = HERE.parents[2]
    version = subprocess.run(
        ["git", "-C", str(framework_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    source = (
        framework_root / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
    )
    model_config_hashes = {}
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "quant_model_description.json",
        "quantization_config.json",
    ):
        path = Path(model) / name
        if path.is_file():
            model_config_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "framework": str(Path(sglang.__file__).resolve()),
        "framework_head": version.stdout.strip() if version.returncode == 0 else None,
        "memory_pool_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_config_sha256": model_config_hashes,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_file": str(Path(torch_npu.__file__).resolve()),
        "visible_npu_count": torch.npu.device_count(),
        "memfabric_file": str(Path(memfabric_hybrid.__file__).resolve()),
        "memfabric_version": mf_version,
        "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
    }


def fixture(args, out):
    if args.token_ids_json:
        path = Path(args.token_ids_json)
        data = json.loads(
            gzip.decompress(path.read_bytes())
            if path.suffix == ".gz"
            else path.read_text(encoding="utf-8")
        )
        ids = data if isinstance(data, list) else data["input_ids"]
        origin = (
            f"external fixture {path.resolve()}; historical equivalence not established"
        )
    else:
        path = HERE / "fixtures" / "prefix100_64000_token_ids.json.gz"
        manifest = json.loads(
            (HERE / "fixtures" / "manifest.json").read_text(encoding="utf-8")
        )
        compressed = path.read_bytes()
        if (
            hashlib.sha256(compressed).hexdigest()
            != "5e7a03d2aec1dfc78ea326670f4336b2af8aa7b764759dae8274a967ea43047e"
        ):
            raise ValueError("Bundled historical fixture gzip SHA256 mismatch")
        raw = gzip.decompress(compressed)
        if (
            hashlib.sha256(raw).hexdigest()
            != "62ddc5e8448554d7dbc1173eb6492ebadf50dc104669d0fe0d827c72b6b11889"
        ):
            raise ValueError("Bundled historical token IDs SHA256 mismatch")
        ids = json.loads(raw)
        if (
            manifest["file"] != path.name
            or manifest["token_count"] != 64000
            or manifest["gzip_sha256"] != hashlib.sha256(compressed).hexdigest()
            or manifest["token_ids_sha256"] != hashlib.sha256(raw).hexdigest()
            or min(ids) != manifest["min_token_id"]
            or max(ids) != manifest["max_token_id"]
        ):
            raise ValueError("Bundled historical fixture manifest mismatch")
        origin = "bundled historical 203 TPS prefix100 token IDs"
    if len(ids) != 64000 or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError(
            "Fixture must contain exactly 64000 nonnegative integer token IDs"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if max(ids) >= len(tokenizer):
        raise ValueError(
            f"Fixture token ID {max(ids)} exceeds tokenizer vocabulary {len(tokenizer)}"
        )
    data = {"input_ids": ids, "origin": origin}
    dump(out / "fixture.json", data)
    sha = hashlib.sha256((out / "fixture.json").read_bytes()).hexdigest()
    print(f"fixture={out / 'fixture.json'} sha256={sha} origin={origin}", flush=True)
    return ids


def precision_fixtures(args, out):
    fixture_dir = HERE / "fixtures" / "precision"
    manifest_path = fixture_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("version") != 1:
        raise ValueError("Unsupported precision fixture manifest version")
    if manifest.get("output_tokens") != 64 or manifest.get("answer") != "731954":
        raise ValueError(
            "Precision fixture manifest must require answer='731954' and "
            "output_tokens=64"
        )
    expected_tokenizer_files = {"tokenizer.json", "tokenizer_config.json"}
    tokenizer_hashes = manifest.get("tokenizer_files_sha256")
    if not isinstance(tokenizer_hashes, dict) or set(tokenizer_hashes) != (
        expected_tokenizer_files
    ):
        raise ValueError(
            "Precision fixture manifest must hash tokenizer.json and "
            "tokenizer_config.json"
        )
    expected_cases = {
        "short": (512, 4),
        "long": (4096, 4),
        "target64k": (64000, 2),
    }
    if set(manifest.get("cases", {})) != set(expected_cases):
        raise ValueError(
            f"Precision fixture manifest must define exactly {sorted(expected_cases)}"
        )
    for case, (token_count, request_count) in expected_cases.items():
        details = manifest["cases"][case]
        if (
            details.get("token_count") != token_count
            or details.get("requests") != request_count
        ):
            raise ValueError(
                f"Precision fixture {case} must declare token_count={token_count} "
                f"and requests={request_count}"
            )

    for name, expected_sha in manifest["tokenizer_files_sha256"].items():
        path = Path(args.model) / name
        if not path.is_file():
            raise ValueError(f"Required tokenizer file is missing: {path}")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Tokenizer file SHA256 mismatch for {name}: "
                f"expected={expected_sha} actual={actual_sha}"
            )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = len(tokenizer)
    output_dir = out / "precision-fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_bytes(manifest_bytes)

    loaded_cases = {}
    for case, details in manifest["cases"].items():
        path = fixture_dir / details["file"]
        compressed = path.read_bytes()
        gzip_sha = hashlib.sha256(compressed).hexdigest()
        if gzip_sha != details["gzip_sha256"]:
            raise ValueError(
                f"Precision fixture {case} gzip SHA256 mismatch: "
                f"expected={details['gzip_sha256']} actual={gzip_sha}"
            )
        raw = gzip.decompress(compressed)
        raw_sha = hashlib.sha256(raw).hexdigest()
        if raw_sha != details["raw_sha256"]:
            raise ValueError(
                f"Precision fixture {case} raw SHA256 mismatch: "
                f"expected={details['raw_sha256']} actual={raw_sha}"
            )
        payload = json.loads(raw)
        ids = payload.get("input_ids") if isinstance(payload, dict) else None
        if not isinstance(ids, list) or len(ids) != details["token_count"]:
            raise ValueError(
                f"Precision fixture {case} must contain exactly "
                f"{details['token_count']} input token IDs"
            )
        if any(type(token_id) is not int or token_id < 0 for token_id in ids):
            raise ValueError(f"Precision fixture {case} has invalid token IDs")
        if ids and max(ids) >= vocab_size:
            raise ValueError(
                f"Precision fixture {case} token ID {max(ids)} exceeds "
                f"tokenizer vocabulary {vocab_size}"
            )
        (output_dir / details["file"]).write_bytes(compressed)
        loaded_cases[case] = {
            "input_ids": ids,
            "requests": details["requests"],
            "token_count": details["token_count"],
            "gzip_sha256": gzip_sha,
            "raw_sha256": raw_sha,
        }
    dump(
        output_dir / "loaded.json",
        {
            "manifest": manifest,
            "tokenizer_vocab_size": vocab_size,
            "cases": {
                name: {key: value for key, value in case.items() if key != "input_ids"}
                for name, case in loaded_cases.items()
            },
        },
    )
    print(f"precision_fixtures={output_dir} tokenizer_vocab={vocab_size}", flush=True)
    return {"manifest": manifest, "cases": loaded_cases}


def validate_retrieval_answer(results, answer):
    line = answer + "\n"
    for result in results:
        text = result.get("response", {}).get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"Request {result.get('request_id')}: missing retrieval response text"
            )
        expected_prefix = line * (len(text) // len(line) + 1)
        if not text or not expected_prefix.startswith(text) or line not in text:
            raise ValueError(
                f"Request {result.get('request_id')}: response text is not a "
                f"valid repeated-answer prefix: {text!r}"
            )


def generate(router, ids, output, rid):
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output,
            "ignore_eos": True,
        },
        "rid": rid,
        "stream": True,
    }
    start_time = time.monotonic()
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            router + "/generate",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        final = None
        first = None
        last_token = None
        done = False
        previous = []
        event_path = EVENT_DIR / f"{rid}.jsonl.gz" if EVENT_DIR else None
        with (
            OPENER.open(req, timeout=7200) as response,
            gzip.open(event_path, "wt", encoding="utf-8")
            if event_path
            else open(os.devnull, "w", encoding="utf-8") as events,
        ):
            for line in response:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                elapsed = time.monotonic() - start_time
                events.write(
                    json.dumps({"elapsed_seconds": elapsed, "data": data}) + "\n"
                )
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise ValueError(f"SSE error: {event['error']}")
                current = event.get("output_ids", [])
                if current[: len(previous)] != previous:
                    raise ValueError(
                        "Non-cumulative SSE output; incremental streaming is unsupported"
                    )
                if current and first is None:
                    first = elapsed
                if len(current) > len(previous):
                    last_token = elapsed
                previous = current
                final = event
        elapsed = time.monotonic() - start_time
        if not done or final is None or first is None:
            raise ValueError("SSE stream ended without [DONE] and a token response")
        return {
            "request_id": rid,
            "response": final,
            "elapsed_seconds": elapsed,
            "ttft_seconds": first,
            "tpot_seconds": (last_token - first) / (len(previous) - 1)
            if len(previous) > 1
            else None,
            "events_path": str(event_path) if event_path else None,
        }
    except urllib.error.HTTPError as error:
        return {
            "request_id": rid,
            "error": f"HTTP {error.code}",
            "body": error.read().decode(errors="replace"),
            "elapsed_seconds": time.monotonic() - start_time,
        }
    except (OSError, ValueError) as error:
        return {
            "request_id": rid,
            "error": repr(error),
            "elapsed_seconds": time.monotonic() - start_time,
        }


def validate(results, input_length, output_length):
    for result in results:
        if "error" in result:
            raise RuntimeError(f"Request failed: {result}")
        response = result.get("response", {})
        meta = response.get("meta_info", {})
        if (
            meta.get("prompt_tokens") != input_length
            or meta.get("completion_tokens") != output_length
            or len(response.get("output_ids", [])) != output_length
            or (meta.get("finish_reason") or {}).get("type") != "length"
        ):
            raise ValueError(f"Request {result['request_id']}: length mismatch: {meta}")


def validate_cached(results, expected):
    cached = [x["response"]["meta_info"].get("cached_tokens") for x in results]
    if any(value != expected for value in cached):
        raise ValueError(f"expected cached_tokens={expected}, got {cached}")


def observed_decode_batch(log_path, expected):
    values = [
        int(value)
        for value in re.findall(
            r"Decode batch(?: \[\d+\])?,\s*#running-req:\s*(\d+)",
            log_path.read_text(encoding="utf-8", errors="replace"),
        )
    ]
    if not values or max(values) < expected:
        raise ValueError(
            f"Decode active batch did not reach BS={expected}; observed max={max(values) if values else 'missing'}; check {log_path}"
        )
    return {
        "max_running_requests": max(values),
        "samples": len(values),
        "log": str(log_path),
    }


def concurrent_generate(router, ids, output, count, label):
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(generate, router, ids, output, f"{label}-{i}")
            for i in range(count)
        ]
        results = [future.result() for future in futures]
    duration = time.monotonic() - started
    return results, duration


def mean_present(values):
    present = [value for value in values if isinstance(value, (int, float))]
    return statistics.mean(present) if present else None


def distribution(values):
    values = sorted(value for value in values if isinstance(value, (int, float)))
    if not values:
        return None

    def percentile(percent):
        position = (len(values) - 1) * percent / 100
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "stddev_population": statistics.pstdev(values),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def round_report(label, results, duration):
    return {
        "round": label,
        "requests": len(results),
        "successful_requests": len(results),
        "failed_requests": 0,
        "input_tokens": sum(
            x["response"]["meta_info"]["prompt_tokens"] for x in results
        ),
        "output_tokens": sum(
            x["response"]["meta_info"]["completion_tokens"] for x in results
        ),
        "duration_seconds": duration,
        "output_tokens_per_second": sum(
            x["response"]["meta_info"]["completion_tokens"] for x in results
        )
        / duration,
        "requests_per_second": len(results) / duration,
        "mean_latency_seconds": mean_present(
            [x.get("elapsed_seconds") for x in results]
        ),
        "mean_ttft_seconds": mean_present([x.get("ttft_seconds") for x in results]),
        "mean_tpot_seconds": mean_present([x.get("tpot_seconds") for x in results]),
        "mean_spec_accept_length": mean_present(
            [x["response"]["meta_info"].get("spec_accept_length") for x in results]
        ),
        "mean_spec_accept_rate": mean_present(
            [x["response"]["meta_info"].get("spec_accept_rate") for x in results]
        ),
        "latency_seconds": distribution([x.get("elapsed_seconds") for x in results]),
        "ttft_seconds": distribution([x.get("ttft_seconds") for x in results]),
        "tpot_seconds": distribution([x.get("tpot_seconds") for x in results]),
    }


def flush(prefill, decode):
    for url in (prefill, decode):
        text = request(url + "/flush_cache?timeout=60", {}, timeout=90)
        if "Cache flushed" not in text:
            raise RuntimeError(f"Unexpected flush response from {url}: {text}")
        print(f"flush {url}: {text.strip()}", flush=True)


def main():
    global EVENT_DIR, RUN_OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("prefill", "decode"))
    parser.add_argument("--model", required=True, help="Local model and tokenizer path")
    parser.add_argument(
        "--prefill-host",
        required=True,
        help="Full IPv4 address reachable from both containers",
    )
    parser.add_argument(
        "--decode-host",
        required=True,
        help="Full IPv4 address reachable from its container",
    )
    parser.add_argument(
        "--devices", required=True, help="Verified container-visible NPU IDs"
    )
    parser.add_argument(
        "--interface", required=True, help="Local HCCL and Gloo interface"
    )
    parser.add_argument("--model-name", default="GLM-5.2-w4a8")
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=163840)
    parser.add_argument(
        "--disable-decode-prefix-cache",
        action="store_true",
        help="Diagnostic only: disable decode radix cache and require full per-request token capacity",
    )
    parser.add_argument("--mem-fraction", type=float, default=0.92)
    parser.add_argument("--prefill-port", type=int, default=41000)
    parser.add_argument("--decode-port", type=int, default=30001)
    parser.add_argument("--router-port", type=int, default=8000)
    parser.add_argument("--store-port", type=int, default=24669)
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--transfer-protocol", default="device_sdma")
    parser.add_argument(
        "--token-ids-json", help="JSON list or object with input_ids; exactly 64000 IDs"
    )
    parser.add_argument(
        "--client",
        choices=("frozen", "stdlib"),
        default="frozen",
        help="Historical bench_serving client (default) or diagnostic stdlib SSE client",
    )
    parser.add_argument("--output-dir", default="split-kv-runs")
    parser.add_argument("--ready-timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.bs < 2 or args.tp < 1:
        parser.error("bs >= 2 and tp >= 1 are required")
    try:
        validate_host_addresses(args.prefill_host, args.decode_host, args.role)
    except ValueError as error:
        parser.error(str(error))
    devices = args.devices.split(",")
    if (
        len(devices) != args.tp
        or len(set(devices)) != len(devices)
        or any(not item.isdecimal() for item in devices)
    ):
        parser.error(
            "devices must contain exactly tp distinct comma-separated numeric container-visible IDs"
        )
    if (
        args.role == "decode"
        and args.disable_decode_prefix_cache
        and args.max_total_tokens < args.bs * (64000 + 1024 + 128)
    ):
        parser.error(
            "Without decode prefix cache, max-total-tokens must be at least bs * (64000 + 1024 + 128); actual capacity and admission must still be verified"
        )
    if not Path(args.model).is_dir():
        parser.error("model must be an existing local directory")
    out = Path(args.output_dir).resolve() / (
        args.role + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    out.mkdir(parents=True, exist_ok=False)
    RUN_OUT = out

    def terminate_handler(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_handler)
    EVENT_DIR = out / "events"
    EVENT_DIR.mkdir()
    controller_log = open(out / "controller.log", "w", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = (
        Tee(sys.stdout, controller_log),
        Tee(sys.stderr, controller_log),
    )
    dump(out / "config.json", vars(args))
    prefill = f"http://{args.prefill_host}:{args.prefill_port}"
    decode = f"http://{args.decode_host}:{args.decode_port}"
    router = f"http://127.0.0.1:{args.router_port}"
    proc = router_proc = None
    try:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
        info = runtime_info(args.model)
        if info["visible_npu_count"] != args.tp:
            raise RuntimeError(
                f"Visible NPU count {info['visible_npu_count']} does not match TP={args.tp}"
            )
        if args.role == "decode":
            info["split_kernel"] = check_split_abi()
        dump(out / "runtime.json", info)
        print(json.dumps(info, indent=2), flush=True)
        if args.role == "prefill":
            for port in (args.prefill_port, args.bootstrap_port, args.store_port):
                port_free(port)
            env = server_env(args)
            record_server_env(out, "prefill", env)
            proc = start(server_command(args, "prefill"), env, out / "prefill.log")
            wait_http(prefill + "/health", proc, args.ready_timeout)
            print(
                "Prefill ready. Keep this process running while decode controller runs.",
                flush=True,
            )
            status = proc.wait()
            if status != 0:
                raise RuntimeError(f"Prefill exited: {status}")
            return
        ids = fixture(args, out)
        precision = precision_fixtures(args, out)
        wait_http(prefill + "/health", timeout=args.ready_timeout)
        round_reports = []
        references = {}
        for phase in ("reference", "functional", "performance"):
            port_free(args.decode_port)
            port_free(args.router_port)
            functional = phase == "functional"
            env = server_env(args, functional, reference=phase == "reference")
            record_server_env(out, phase, env)
            proc = start(
                server_command(args, "decode", functional, phase == "performance"),
                env,
                out / f"decode-{phase}.log",
            )
            wait_http(decode + "/health", proc, args.ready_timeout)
            command = [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--prefill",
                prefill,
                "--decode",
                decode,
                "--host",
                "127.0.0.1",
                "--port",
                str(args.router_port),
            ]
            router_env = os.environ.copy()
            for name in INHERITED_ENV_TO_CLEAR:
                router_env.pop(name, None)
            router_proc = start(command, router_env, out / f"router-{phase}.log")
            wait_http(router + "/health", router_proc, args.ready_timeout)
            wait_router_workers(
                router, prefill, decode, router_proc, args.ready_timeout
            )
            if phase in ("reference", "functional"):
                for case, case_fixture in precision["cases"].items():
                    flush(prefill, decode)
                    results, duration = concurrent_generate(
                        router,
                        case_fixture["input_ids"],
                        precision["manifest"]["output_tokens"],
                        case_fixture["requests"],
                        f"{phase}-{case}",
                    )
                    dump(
                        out / f"{phase}-{case}.json",
                        {
                            "fixture": {
                                "case": case,
                                "token_count": case_fixture["token_count"],
                                "gzip_sha256": case_fixture["gzip_sha256"],
                                "raw_sha256": case_fixture["raw_sha256"],
                            },
                            "answer_validation": {
                                "expected": precision["manifest"]["answer"],
                                "passed": None,
                            },
                            "results": results,
                            "duration_seconds": duration,
                        },
                    )
                    validate(
                        results,
                        case_fixture["token_count"],
                        precision["manifest"]["output_tokens"],
                    )
                    validate_retrieval_answer(results, precision["manifest"]["answer"])
                    report_path = out / f"{phase}-{case}.json"
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    report["answer_validation"]["passed"] = True
                    dump(report_path, report)
                    tokens = [x["response"]["output_ids"] for x in results]
                    if phase == "reference":
                        reference_tokens = tokens[0]
                        if any(value != reference_tokens for value in tokens):
                            raise ValueError(
                                f"Native reference {case} greedy outputs differ"
                            )
                        references[case] = reference_tokens
                    else:
                        if any(value != references[case] for value in tokens):
                            raise ValueError(
                                f"Functional {case} greedy outputs differ from native reference"
                            )
                        count = sum(
                            x["response"]["meta_info"].get("num_retractions", 0)
                            for x in results
                        )
                        if count < 1:
                            raise ValueError(
                                f"Functional {case} requests did not report any retraction"
                            )
                        print(
                            f"functional {case} greedy exact match; retractions={count}",
                            flush=True,
                        )
            else:
                for index in range(7):
                    label = f"warm-{index + 1}" if index < 2 else f"formal-{index - 1}"
                    flush(prefill, decode)
                    if args.client == "frozen":
                        from frozen_bench import run_round

                        report = run_round(router, args.model, out, ids, args.bs, label)
                        tps = report.get("output_tokens_per_second")
                        if (
                            report.get("round") != label
                            or not isinstance(tps, (int, float))
                            or not math.isfinite(tps)
                            or tps <= 0
                        ):
                            raise ValueError(
                                f"{label}: frozen client returned an invalid report"
                            )
                        round_reports.append(report)
                        dump(out / "rounds-partial.json", round_reports)
                        print(
                            json.dumps(report, ensure_ascii=False, indent=2), flush=True
                        )
                        continue
                    seed = generate(router, ids, 1, label + "-seed")
                    dump(out / f"{label}-seed.json", seed)
                    validate([seed], 64000, 1)
                    results, duration = concurrent_generate(
                        router, ids, 1024, args.bs, label
                    )
                    dump(
                        out / f"{label}.json",
                        {
                            "seed": seed,
                            "results": results,
                            "duration_seconds": duration,
                            "output_tokens_per_second": args.bs * 1024 / duration,
                        },
                    )
                    validate(results, 64000, 1024)
                    validate_cached(results, 63872)
                    report = round_report(label, results, duration)
                    round_reports.append(report)
                    dump(out / "rounds-partial.json", round_reports)
                    print(
                        f"{label}: requests={len(results)} input=64000 output=1024 cached=63872 wall={duration:.3f}s output_tps={args.bs * 1024 / duration:.3f}",
                        flush=True,
                    )
            if phase == "performance":
                active_batch = observed_decode_batch(
                    out / "decode-performance.log", args.bs
                )
                dump(out / "observed-decode-batch.json", active_batch)
            stop(router_proc)
            router_proc = None
            stop(proc)
            proc = None
        formal_tps = [
            r["output_tokens_per_second"]
            for r in round_reports
            if r["round"].startswith("formal-")
        ]
        summary = {
            "greedy_accuracy_validation": {
                "cases": [
                    {
                        "input_tokens": case["token_count"],
                        "output_tokens": precision["manifest"]["output_tokens"],
                        "requests": case["requests"],
                    }
                    for case in precision["cases"].values()
                ],
                "answer": precision["manifest"]["answer"],
                "scope": precision["manifest"]["scope"],
            },
            "rounds": round_reports,
            "observed_decode_batch": active_batch,
            "formal_output_tps": {
                "median": statistics.median(formal_tps),
                "min": min(formal_tps),
                "max": max(formal_tps),
                "stddev_population": statistics.pstdev(formal_tps),
            },
            "note": "Client wall throughput is not scheduler decode-only throughput or the paper ITL-derived metric.",
        }
        dump(out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        original_error = sys.exc_info()[1]
        cleanup_errors = []
        for owned in (router_proc, proc):
            try:
                stop(owned)
            except Exception as error:
                cleanup_errors.append(repr(error))
        if cleanup_errors:
            message = f"Owned process cleanup failed: {cleanup_errors}"
            print(message, file=sys.stderr, flush=True)
            if original_error is None:
                raise RuntimeError(message)
    if args.role == "decode":
        print(
            f"PASS: reference, functional and seven benchmark rounds; cleanup complete; output={out}",
            flush=True,
        )


if __name__ == "__main__":
    exit_code = 1
    try:
        main()
        exit_code = 0
    except KeyboardInterrupt:
        traceback.print_exc()
        exit_code = 130
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
        if exit_code:
            traceback.print_exc()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if RUN_OUT is not None:
            (RUN_OUT / "exitcode.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    sys.exit(exit_code)
