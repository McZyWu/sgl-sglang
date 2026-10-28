"""Launch matched, bounded prefill/decode interval cases in a private container.

Run this only in a container owned by this experiment. The launcher preserves
the externally assigned device list. Cleanup targets only this script's newly
created process session; it never stops another container or existing server.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from check_stream_gaps import request_json


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stat(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "state": fields[0],
            "group": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": int(fields[19]),
        }
    except (OSError, ValueError, IndexError):
        return None


def members(owner):
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        info = stat(int(path.name))
        if info and info["group"] == owner["group"] and info["state"] != "Z":
            if (
                info["session"] != owner["session"]
                or info["start_ticks"] < owner["start_ticks"]
            ):
                raise RuntimeError(
                    "process-session ownership check failed; refusing cleanup"
                )
            result.append(info)
    return result


def stop_owned(server, owner):
    initial = members(owner)
    if initial:
        os.killpg(owner["group"], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while members(owner) and time.monotonic() < deadline:
        time.sleep(0.5)
    escalated = members(owner)
    if escalated:
        os.killpg(owner["group"], signal.SIGKILL)
    server.wait(timeout=30)
    deadline = time.monotonic() + 10
    while members(owner) and time.monotonic() < deadline:
        time.sleep(0.5)
    remaining = members(owner)
    return {
        "initial_members": initial,
        "kill_escalation_members": escalated,
        "remaining": remaining,
        "server_exitcode": server.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--intervals", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--input-lengths", nargs="+", type=int, default=[32768, 64000])
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=1200)
    parser.add_argument("--client-timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a fresh run")
    if not args.intervals or min(args.intervals) < 0:
        parser.error("intervals must be nonnegative")
    if (
        min(
            args.input_lengths
            + [
                args.num_requests,
                args.concurrency,
                args.output_tokens,
                args.warmup_output_tokens,
            ]
        )
        < 1
    ):
        parser.error("request counts and token lengths must be positive")
    if args.num_requests < args.concurrency:
        parser.error("num-requests must be at least concurrency")
    args.output_dir.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    port = int(os.environ.get("PORT", "18888"))
    base_url = f"http://127.0.0.1:{port}"
    state = {"phase": "prepared", "started_unix": time.time(), "cases": []}
    write_json(args.output_dir / "status.json", state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"controlled stop signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for interval in args.intervals:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(
                        "port already in use; refusing to touch an existing service"
                    )
            case_dir = args.output_dir / f"interval-{interval}"
            case_dir.mkdir()
            env = os.environ.copy()
            env["PREFILL_DECODE_INTERVAL"] = str(interval)
            case = {"interval": interval, "phase": "loading", "results": []}
            state["cases"].append(case)
            state["phase"] = "loading"
            write_json(args.output_dir / "status.json", state)
            with (case_dir / "server.log").open("wb") as log:
                server = subprocess.Popen(
                    ["bash", str(here / "launch_server.sh")],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            owner = stat(server.pid)
            if (
                not owner
                or owner["group"] != server.pid
                or owner["session"] != server.pid
            ):
                raise RuntimeError("new server session was not established")
            case["server_owner"] = owner
            write_json(args.output_dir / "status.json", state)
            try:
                deadline = time.monotonic() + args.ready_timeout
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(
                            f"server exited before readiness: {server.returncode}"
                        )
                    try:
                        info = request_json(base_url + "/server_info", timeout=3)
                        config = info.get("server_args", info)
                        if config.get("prefill_decode_interval") != interval:
                            raise RuntimeError(
                                "resolved scheduling interval differs from this case"
                            )
                        write_json(case_dir / "server-info.json", info)
                        break
                    except (OSError, TimeoutError, json.JSONDecodeError):
                        if time.monotonic() > deadline:
                            raise RuntimeError("server readiness deadline exceeded")
                        time.sleep(2)
                # Exercise every DP worker before measuring. Warmup is retained
                # separately and never included in the reported A/B statistics.
                replays = [
                    ("warmup", min(args.input_lengths), args.warmup_output_tokens)
                ]
                replays += [
                    (f"L{length}", length, args.output_tokens)
                    for length in args.input_lengths
                ]
                for name, length, output_tokens in replays:
                    case["phase"] = state["phase"] = (
                        "warming" if name == "warmup" else "testing"
                    )
                    write_json(args.output_dir / "status.json", state)
                    output = case_dir / f"{name}.json"
                    command = [
                        sys.executable,
                        str(here / "check_stream_gaps.py"),
                        "--base-url",
                        base_url,
                        "--requests",
                        str(args.requests),
                        "--output",
                        str(output),
                        "--expected-interval",
                        str(interval),
                        "--input-tokens",
                        str(length),
                        "--num-requests",
                        str(args.num_requests),
                        "--concurrency",
                        str(args.concurrency),
                        "--output-tokens",
                        str(output_tokens),
                        "--timeout",
                        str(args.client_timeout),
                    ]
                    with (case_dir / f"{name}.client.log").open("wb") as log:
                        result = subprocess.run(
                            command,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.client_timeout + 60,
                        )
                    if result.returncode:
                        raise RuntimeError(
                            f"client failed for length {length}: {result.returncode}"
                        )
                    summary = json.loads(output.read_text())["summary"]
                    if name == "warmup":
                        case["warmup"] = summary
                    else:
                        case["results"].append(summary)
                    write_json(args.output_dir / "status.json", state)
                    print(
                        json.dumps(
                            {
                                "interval": interval,
                                "length": length,
                                "phase": "case_complete",
                            }
                        ),
                        flush=True,
                    )
                case["phase"] = "complete"
            finally:
                case["cleanup"] = stop_owned(server, owner)
                write_json(args.output_dir / "status.json", state)
                if case["cleanup"]["remaining"]:
                    raise RuntimeError("owned server still has live processes")
        state["phase"] = "complete"
    except BaseException as exc:
        state["phase"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state["finished_unix"] = time.time()
        state["scripts_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in here.glob("*.py")
        }
        state["scripts_sha256"]["launch_server.sh"] = hashlib.sha256(
            (here / "launch_server.sh").read_bytes()
        ).hexdigest()
        write_json(args.output_dir / "status.json", state)


if __name__ == "__main__":
    main()
