"""Measure C8 A/B serving performance with the repository benchmark client.

Run against one local server at a time. The launch manifest binds the mode to
the live process; compare refuses incomplete or differently configured runs.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import urllib.request
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[3]
MODE_ENV = (
    "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD",
    "SGLANG_NPU_ENABLE_COPY_SFA",
    "SGLANG_NPU_SPARSE_KV_INDEXER",
)
ENV_KEYS = MODE_ENV + (
    "SGLANG_NPU_USE_MLAPO",
    "SGLANG_NPU_SPARSE_KV_CACHE_SIZE",
    "DEEPEP_HYBRID_DEPLOYMENT",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
    "HCCL_BUFFSIZE",
    "DEEPEP_HCCL_BUFFSIZE",
    "DEEP_USE_MODE",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM",
    "SGLANG_DEBUG_NPU_C8_OFFLOAD",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
)
SERVER_KEYS = (
    "model_path",
    "served_model_name",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "attention_backend",
    "tp_size",
    "ep_size",
    "dp_size",
    "attn_dp_size",
    "pp_size",
    "context_length",
    "page_size",
    "max_total_tokens",
    "max_running_requests",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "mem_fraction_static",
    "cuda_graph_bs_decode",
    "disable_cuda_graph",
    "disable_radix_cache",
    "disable_overlap_schedule",
    "enable_prefill_delayer",
    "prefill_delayer_max_delay_passes",
    "moe_a2a_backend",
    "deepep_mode",
    "speculative_algorithm",
    "random_seed",
    "stream_interval",
    "version",
)
METRICS = (
    "output_throughput",
    "request_throughput",
    "mean_e2e_latency_ms",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "mean_tpot_ms",
    "p95_tpot_ms",
)

# A5 Copy-SFA limits the length of each request's block table, not the
# aggregate indexer pool. req_to_token adds four columns before page alignment.
MAX_C8_CONTEXT = 262144 - 4
TUNING_KEYS = {"max_total_tokens", "max_running_requests", "cuda_graph_bs_decode"}
DECODE_METRICS = (
    "decode_output_throughput",
    "steady_state_window_s",
    "steady_batch_step_ms",
    "mean_tpot_ms",
    "p95_tpot_ms",
)


def metric_keys(record):
    return (
        DECODE_METRICS if record["protocol"].get("measurement") == "decode" else METRICS
    )


def decode_metrics(result, batch, length, output):
    """Do not turn a clamped output or an empty full-batch window into a win."""
    if (
        result["batch_size"] != batch
        or result["input_len"] != length
        or result["output_len"] != output
        or result["total_output_tokens"] != batch * output
        or result["num_early_stopped"] != 0
    ):
        raise ValueError(
            "Decode benchmark did not complete the requested batch/lengths"
        )
    values = {
        "decode_output_throughput": result["steady_state_output_throughput"],
        "steady_state_window_s": result["steady_state_window_s"],
        "mean_tpot_ms": result["mean_tpot_ms"],
        "p95_tpot_ms": result["p95_tpot_ms"],
    }
    if any(
        values[key] is None or not math.isfinite(values[key]) or values[key] <= 0
        for key in ("mean_tpot_ms", "p95_tpot_ms")
    ):
        raise ValueError("Invalid per-request TPOT")
    if (
        values["decode_output_throughput"] is None
        and values["steady_state_window_s"] is None
    ):
        return values | {"steady_batch_step_ms": None, "status": "no_full_batch_window"}
    if any(
        value is None or not math.isfinite(value) or value <= 0
        for value in values.values()
    ):
        raise ValueError(
            "No valid full-batch decode window/TPOT; inspect the client log"
        )
    values["steady_batch_step_ms"] = 1000 * batch / values["decode_output_throughput"]
    return values


def decode_command(args, config, dataset, raw_path, length, batch):
    return [
        sys.executable,
        "-m",
        "sglang.benchmark.one_batch_server",
        "--model-path",
        "None",
        "--base-url",
        args.url,
        "--local-tokenizer-path",
        config["model_path"],
        "--dataset-name",
        "random",
        "--dataset-path",
        str(dataset),
        "--batch-size",
        str(batch),
        "--input-len",
        str(length),
        "--output-len",
        str(args.new_tokens),
        "--temperature",
        "0",
        "--seed",
        "42",
        "--client-stream-interval",
        "1",
        "--result-filename",
        str(raw_path),
        "--no-append-to-github-summary",
    ]


def decode_table(record):
    text = "| Input | Batch requests | Round | Decode output tok/s | Mean TPOT ms | P95 TPOT ms | Full-batch window s | Status |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for row in record["cases"]:
        cells = [row["input_tokens"], row["concurrency"], row["repeat"]]
        cells += [
            f"{row[key]:.2f}" if row.get(key) is not None else "N/A"
            for key in (
                "decode_output_throughput",
                "mean_tpot_ms",
                "p95_tpot_ms",
                "steady_state_window_s",
            )
        ]
        cells.append(row.get("status", "pass"))
        text += "| " + " | ".join(map(str, cells)) + " |\n"
    return text


def validate_capacity(config, lengths, output):
    required = max(lengths) + output + 2
    # Worker max_req_len = min(context, logical pool) - 1; scheduler reserves
    # another token. Equality with input+output silently clips two output tokens.
    if (
        not isinstance(config["context_length"], int)
        or config["context_length"] < required
    ):
        raise ValueError(
            f"Need context_length >= {required} (input + output + 2); regenerate perf-configs and restart"
        )
    capacity = config.get("max_total_num_tokens") or config.get("max_total_tokens")
    if capacity and capacity < required:
        raise ValueError(
            f"Actual token pool {capacity} < required {required}; one complete request cannot fit"
        )
    return capacity


def batch_observations(log, offset, measured=False):
    """Read existing scheduler decode logs; never infer batch from client load."""
    with log.open("rb") as file:
        file.seek(offset)
        data = file.read().decode("utf-8", errors="replace")
    values = [
        int(value)
        for value in re.findall(r"Decode batch[^\n]*?#running-req:\s*(\d+)", data)
    ]
    return {
        "scope": f"logged decode batches, {'after client warmup marker' if measured else 'including warmup'}; per scheduler/DP, not padded graph size",
        "samples": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def run_client(command, log_path, timeout, server_log):
    """Copy client output and mark the server-log boundary after warmup."""
    boundary = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def copy():
            for line in process.stdout:
                log.write(line)
                log.flush()
                if (
                    "Starting main benchmark run" in line
                    and server_log.exists()
                    and not boundary
                ):
                    boundary.append(server_log.stat().st_size)

        reader = threading.Thread(target=copy, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        finally:
            reader.join(timeout=10)
            process.stdout.close()
        if code:
            raise subprocess.CalledProcessError(code, command)
    return boundary[0] if boundary else None


def memory_snapshot(path):
    with path.open("w", encoding="utf-8") as file:
        subprocess.run(
            ["npu-smi", "info"],
            stdout=file,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=30,
        )
    cards, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        card = re.match(r"\|\s*(\d+)\s*\|\s*Ascend", line)
        if card:
            current = card.group(1)
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
        if current is not None and len(pairs) >= 2 and int(pairs[-1][1]) > 0:
            used, total = map(int, pairs[-1])
            cards[current] = {
                "used_mb": used,
                "total_mb": total,
                "free_mb": total - used,
            }
            current = None
    return {
        "raw": path.name,
        "hbm_by_npu": cards,
        "scope": "device-wide snapshot, not per-process allocation or a continuous peak",
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def git_revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def process_start(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def capture_launch(args):
    write_json(
        args.output,
        {
            "mode": args.mode,
            "pid": args.pid,
            "process_start": process_start(args.pid),
            "port": int(os.environ.get("PORT", "8810")),
            "sglang_revision": git_revision(REPO),
            "kernel_revision": git_revision(os.environ["KERNEL_SOURCE_DIR"]),
            "packages": {
                name: version(name) for name in ("torch", "torch-npu", "sgl-kernel-npu")
            },
            "environment": {key: os.environ.get(key) for key in ENV_KEYS},
        },
    )


def live_manifest(path, port):
    item = json.loads(path.read_text(encoding="utf-8"))
    if item["port"] != port or process_start(item["pid"]) != item["process_start"]:
        raise ValueError("stale server process")
    command = Path(f"/proc/{item['pid']}/cmdline").read_bytes().split(b"\0")
    if b"sglang.launch_server" not in command:
        raise ValueError("launch process has not entered sglang.launch_server")
    return item


def find_launch(args):
    endpoint = urlsplit(args.url)
    if endpoint.scheme != "http" or endpoint.hostname not in (
        "127.0.0.1",
        "localhost",
        "::1",
    ):
        raise ValueError(
            "Performance manifest verification requires a local HTTP server"
        )
    if (
        endpoint.path not in ("", "/")
        or endpoint.query
        or endpoint.fragment
        or endpoint.username
    ):
        raise ValueError("Use the server root URL")
    paths = (
        [args.launch_manifest]
        if args.launch_manifest
        else args.results_root.glob("serve-*/launch.json")
    )
    matches = []
    for path in paths:
        try:
            matches.append((path, live_manifest(path, endpoint.port or 80)))
        except (OSError, ValueError, KeyError):
            continue
    if len(matches) != 1:
        raise ValueError(
            "Need exactly one live launch.json; restart this server with the updated container.sh serve"
        )
    path, launch = matches[0]
    expected = ["0", "0", "legacy"] if args.mode == "baseline" else ["1", "1", "lim"]
    if (
        launch["mode"] != args.mode
        or [launch["environment"][key] for key in MODE_ENV] != expected
    ):
        raise ValueError(
            f"Requested {args.mode}, but live server manifest is {launch['mode']}"
        )
    if launch["sglang_revision"] != git_revision(REPO):
        raise ValueError(
            "Server and client revisions differ; restart the server after pulling code"
        )
    return path, launch


def server_info(url):
    with urllib.request.urlopen(
        url.rstrip("/") + "/server_info", timeout=30
    ) as response:
        return json.load(response)


def dataset_bytes(count):
    # Natural-language synthetic prompts, not random vocabulary IDs. The
    # repository random loader repeats/truncates them to exact token lengths.
    rows = []
    for index in range(count):
        prompt = (
            f"Document {index}. Read the following inventory notes.\n"
            + "".join(
                f"Record {line}: warehouse {(index * 19 + line * 7) % 97}, "
                f"quantity {(index * 31 + line * 73) % 997}, checked in October 2026.\n"
                for line in range(128)
            )
            + "Summarize the records and explain the inventory trends."
        )
        rows.append(
            {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": "Summary."},
                ]
            }
        )
    return json.dumps(rows, ensure_ascii=False).encode()


def prepare_configs(args):
    if (
        not 132098 <= args.context_length <= MAX_C8_CONTEXT
        or (
            args.max_total_tokens is not None
            and args.max_total_tokens < args.context_length
        )
        or args.max_running_requests < max(args.concurrencies)
        or min(args.concurrencies) <= 0
    ):
        raise ValueError(
            "Require 132098 <= context <= 262140, explicit pool >= context, and positive concurrencies <= max-running-requests"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    original = args.config.read_text(encoding="utf-8")
    graph_batches = " ".join(
        map(str, sorted({1, *args.concurrencies, args.max_running_requests}))
    )
    for cache in (2048, 4096):
        for mode in ("baseline", "lim_copy_sfa"):
            target = args.output / f"c{cache}-{mode}.env"
            if target.exists():
                raise ValueError(f"Config already exists: {target}")
            footer = (
                "\n# Whole-model throughput search under target mean TPOT 50 ms/token; allow the 50s (<60).\n"
                f"LIM_EVAL_MODE={mode}\nCACHE_TOKENS={cache}\n"
                f"CONTEXT_LENGTH={args.context_length}\nMAX_TOTAL_TOKENS={args.max_total_tokens or ''}\n"
                f"MAX_RUNNING_REQUESTS={args.max_running_requests}\n"
                f'GRAPH_BATCH_SIZES="{graph_batches}"\n'
                'C8_PERF_INPUT_LENGTHS="131072 65536 16384"\nC8_PERF_NEW_TOKENS=1024\n'
                f'C8_PERF_CONCURRENCIES="{" ".join(map(str, args.concurrencies))}"\n'
                "C8_PERF_ROUNDS=1\nC8_PERF_WAVES=2\nC8_PERF_TPOT_LIMIT_MS=60\nC8_PERF_CASE_TIMEOUT=14400\n"
                "SGLANG_DEBUG_NPU_C8_OFFLOAD=0\n"
            )
            target.write_text(original + footer, encoding="utf-8", newline="\n")
            target.chmod(0o600)
            print(target, flush=True)
    print(
        "MAX_TOTAL_TOKENS empty = size each mode's actual pool from the same mem_fraction_static. Check startup max_total_num_tokens and host DRAM before increasing max-running-requests.",
        flush=True,
    )


def validate_result(result, count, length, output):
    if (
        result["completed"] != count
        or len(result["errors"]) != count
        or any(result["errors"])
    ):
        raise ValueError(
            "A measured request failed; this is not a valid performance result"
        )
    for key, expected in (("input_lens", length), ("output_lens", output)):
        if len(result[key]) != count or any(value != expected for value in result[key]):
            raise ValueError(
                f"Unexpected {key}: require {count} requests of {expected} tokens; "
                f"actual={result[key]}; finish_reasons={result.get('finish_reasons')}; errors={result.get('errors')}"
            )
    if result["total_output_tokens"] != count * output:
        raise ValueError("Unexpected total generated token count")
    if len(result["ttfts"]) != count or any(
        not math.isfinite(value) or value <= 0 for value in result["ttfts"]
    ):
        raise ValueError("A request has no valid first-token timing")
    for key in METRICS:
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"Invalid metric {key}: {result[key]}")


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "performance.json"
    if summary_path.exists():
        raise ValueError("Use a fresh output directory")
    record = {"status": "running", "mode": args.mode, "cases": []}
    decode = getattr(args, "measurement", "serving") == "decode"
    try:
        if (
            min(
                args.input_lengths
                + args.concurrencies
                + [args.new_tokens, args.rounds, args.waves, args.case_timeout]
            )
            <= 0
            or args.new_tokens < 2
        ):
            raise ValueError(
                "Lengths/concurrencies/rounds/waves/timeouts must be positive; new_tokens >= 2"
            )
        if len(set(args.input_lengths)) != len(args.input_lengths) or len(
            set(args.concurrencies)
        ) != len(args.concurrencies):
            raise ValueError(
                "Input lengths and concurrencies must not contain duplicates"
            )
        path, launch = find_launch(args)
        info = server_info(args.url)
        config = {key: info.get(key) for key in SERVER_KEYS}
        if config["kv_cache_dtype"] != "fp8_e4m3" or not config["disable_radix_cache"]:
            raise ValueError("Require C8 KV and disabled radix cache for both modes")
        capacity = validate_capacity(
            config | {"max_total_num_tokens": info.get("max_total_num_tokens")},
            args.input_lengths,
            args.new_tokens,
        )
        if launch["environment"].get("SGLANG_DEBUG_NPU_C8_OFFLOAD") not in (
            None,
            "0",
            "false",
            "False",
        ):
            raise ValueError(
                "Disable SGLANG_DEBUG_NPU_C8_OFFLOAD and restart before timing performance"
            )
        if not math.isfinite(args.tpot_limit_ms) or args.tpot_limit_ms <= 0:
            raise ValueError("TPOT limit must be positive and finite")
        if (
            config["stream_interval"] != 1
            or config["speculative_algorithm"] is not None
        ):
            raise ValueError(
                "Require stream_interval=1 and speculative decoding disabled"
            )
        # Stable input source even when the modes scan different concurrencies.
        data = dataset_bytes(max(1024, max(args.concurrencies) * args.waves))
        dataset = args.output / "prompts.json"
        dataset.write_bytes(data)
        record.update(
            launch_manifest=str(path),
            launch=launch,
            server_configuration=config,
            actual_token_capacity=capacity,
            objective={
                "metric": "mean_tpot_ms",
                "strict_upper_bound_ms": args.tpot_limit_ms,
                "gate": "every measured repetition must satisfy the bound",
            },
            protocol={
                "measurement": "decode" if decode else "serving",
                "requests_equal_concurrency": getattr(
                    args, "requests_equal_concurrency", False
                ),
                "input_lengths": args.input_lengths,
                "concurrencies": args.concurrencies,
                "new_tokens": args.new_tokens,
                "rounds": args.rounds,
                "waves": 1
                if decode or getattr(args, "requests_equal_concurrency", False)
                else args.waves,
                "seed": 42,
                "dataset_sha256": hashlib.sha256(data).hexdigest(),
            },
        )
        if decode:
            record["objective"]["maximize"] = "decode_output_throughput"
            states = info.get("internal_states") or [{}]
            per_dp = (
                states[0].get("effective_max_running_requests_per_dp")
                or config["max_running_requests"]
            )
            dp_ranks = (config["dp_size"] or 1) * (config["attn_dp_size"] or 1)
            if not per_dp or max(args.concurrencies) > per_dp * dp_ranks:
                raise ValueError(
                    "Decode batch exceeds the server's effective running-request cap; restart with a larger cap"
                )
            print(
                "Decode metric: last request first token -> first request finish; one batch, no waves",
                flush=True,
            )
        write_json(args.output / "server_info.json", info)
        record["memory_before"] = memory_snapshot(args.output / "npu-smi-before.txt")
        server_log = path.parent / "server.log"
        # Interleave cases across repetitions to expose time-dependent drift.
        for repeat in range(args.rounds):
            for length in args.input_lengths:
                for concurrency in args.concurrencies:
                    live_manifest(path, launch["port"])
                    count = (
                        concurrency
                        if decode or getattr(args, "requests_equal_concurrency", False)
                        else max(4, concurrency * args.waves)
                    )
                    name = f"i{length}-o{args.new_tokens}-c{concurrency}-r{repeat + 1}"
                    raw_path = args.output / f"{name}.jsonl"
                    record["active_case"] = {
                        "name": name,
                        "raw_result": str(raw_path),
                        "client_log": str(args.output / f"{name}.log"),
                    }
                    write_json(summary_path, record)
                    if (
                        decode
                        and capacity
                        and concurrency * (length + args.new_tokens) > capacity
                    ):
                        record["cases"].append(
                            {
                                "input_tokens": length,
                                "concurrency": concurrency,
                                "repeat": repeat + 1,
                                "requests": count,
                                "status": "capacity_limited",
                                "meets_tpot_slo": False,
                                "required_tokens": concurrency
                                * (length + args.new_tokens),
                                **dict.fromkeys(DECODE_METRICS),
                            }
                        )
                        write_json(summary_path, record)
                        print(
                            f"CAPACITY LIMITED {name}: full batch cannot fit token pool={capacity}",
                            flush=True,
                        )
                        continue
                    log_offset = (
                        server_log.stat().st_size if server_log.exists() else None
                    )
                    command = [
                        sys.executable,
                        "-m",
                        "sglang.benchmark.serving",
                        "--backend",
                        "sglang",
                        "--base-url",
                        args.url,
                        "--model",
                        config["model_path"],
                        "--tokenizer",
                        config["model_path"],
                        "--dataset-name",
                        "random",
                        "--dataset-path",
                        str(dataset),
                        "--tokenize-prompt",
                        "--random-input-len",
                        str(length),
                        "--random-output-len",
                        str(args.new_tokens),
                        "--random-range-ratio",
                        "1",
                        "--seed",
                        "42",
                        "--temperature",
                        "0",
                        "--num-prompts",
                        str(count),
                        "--max-concurrency",
                        str(concurrency),
                        "--request-rate",
                        "inf",
                        "--warmup-requests",
                        str(concurrency),
                        "--output-details",
                        "--disable-tqdm",
                        "--output-file",
                        str(raw_path),
                    ]
                    if decode:
                        command = decode_command(
                            args, config, dataset, raw_path, length, concurrency
                        )
                    write_json(args.output / f"{name}.command.json", command)
                    print(
                        f"START {name}: {count} requests; warmup={concurrency}",
                        flush=True,
                    )
                    if capacity and concurrency * (length + args.new_tokens) > capacity:
                        print(
                            f"NOTE: {concurrency} clients demand {concurrency * (length + args.new_tokens)} tokens, actual pool={capacity}; active decode batch may be lower",
                            flush=True,
                        )
                    measured_offset = run_client(
                        command,
                        args.output / f"{name}.log",
                        args.case_timeout,
                        server_log,
                    )
                    lines = raw_path.read_text(encoding="utf-8").splitlines()
                    if len(lines) != 1:
                        raise ValueError(f"Expected one benchmark record: {raw_path}")
                    result = json.loads(lines[0])
                    if decode:
                        metrics = decode_metrics(
                            result, concurrency, length, args.new_tokens
                        )
                        result_info = server_info(args.url)
                    else:
                        validate_result(result, count, length, args.new_tokens)
                        metrics = {key: result[key] for key in METRICS}
                        result_info = result["server_info"]
                    if {key: result_info.get(key) for key in SERVER_KEYS} != config:
                        raise ValueError(
                            "Server configuration changed during benchmarking"
                        )
                    case = {
                        "input_tokens": length,
                        "concurrency": concurrency,
                        "repeat": repeat + 1,
                        "requests": count,
                        "raw_result": raw_path.name,
                        **metrics,
                        "meets_tpot_slo": metrics.get("status", "pass") == "pass"
                        and result["mean_tpot_ms"] < args.tpot_limit_ms,
                        "decode_batch": batch_observations(server_log, log_offset)
                        if log_offset is not None
                        else {
                            "samples": 0,
                            "max": None,
                            "scope": "server.log unavailable",
                        },
                        "measured_decode_batch": batch_observations(
                            server_log, measured_offset, measured=True
                        )
                        if measured_offset is not None
                        else {"samples": 0, "max": None},
                    }
                    record["cases"].append(case)
                    write_json(summary_path, record)
                    if decode:
                        (args.output / "decode-summary.md").write_text(
                            decode_table(record), encoding="utf-8"
                        )
                        if metrics.get("status", "pass") == "pass":
                            print(
                                f"PASS {name}: decode output={metrics['decode_output_throughput']:.3f} tok/s; mean TPOT={metrics['mean_tpot_ms']:.2f} ms; full-batch window={metrics['steady_state_window_s']:.2f} s",
                                flush=True,
                            )
                        else:
                            print(
                                f"NO FULL-BATCH WINDOW {name}: no decode throughput eligible for comparison",
                                flush=True,
                            )
                    else:
                        print(
                            f"PASS {name}: output={result['output_throughput']:.2f} tok/s; TTFT={result['mean_ttft_ms']:.2f} ms; TPOT={result['mean_tpot_ms']:.2f} ms",
                            flush=True,
                        )
        live_manifest(path, launch["port"])
        record["memory_after"] = memory_snapshot(args.output / "npu-smi-after.txt")
        record["status"] = "pass"
        record.pop("active_case", None)
        write_json(args.output / "slo-summary.json", slo_summary(record))
        print(json.dumps(slo_summary(record)["best"], indent=2), flush=True)
        if decode:
            text = decode_table(record)
            (args.output / "decode-summary.md").write_text(text, encoding="utf-8")
            print(text, flush=True)
        print(f"Completed {args.mode}: {summary_path}", flush=True)
    except Exception as error:
        record.update(status="fail", error=str(error))
        raise
    finally:
        write_json(summary_path, record)


def grouped(record):
    groups = {}
    for case in record["cases"]:
        groups.setdefault((case["input_tokens"], case["concurrency"]), []).append(case)
    expected = {
        (length, concurrency)
        for length in record["protocol"]["input_lengths"]
        for concurrency in record["protocol"]["concurrencies"]
    }
    if set(groups) != expected or any(
        sorted(row["repeat"] for row in rows)
        != list(range(1, record["protocol"]["rounds"] + 1))
        for rows in groups.values()
    ):
        raise ValueError("Incomplete/repeated performance cases")
    return groups


def slo_summary(record):
    """Keep workloads separate and gate *all* rounds, not just a fast median."""
    limit = record["objective"]["strict_upper_bound_ms"]
    keys = metric_keys(record)
    points = []
    for (length, concurrency), samples in sorted(grouped(record).items()):
        batches = [
            item["decode_batch"]["max"]
            for item in samples
            if item.get("decode_batch", {}).get("max") is not None
        ]
        points.append(
            {
                "mode": record["mode"],
                "cache_tokens": int(
                    record["launch"]["environment"]["SGLANG_NPU_SPARSE_KV_CACHE_SIZE"]
                ),
                "input_tokens": length,
                "output_tokens": record["protocol"]["new_tokens"],
                "client_concurrency": concurrency,
                "status": ";".join(
                    sorted({item.get("status", "pass") for item in samples})
                ),
                **{
                    key: statistics.median(item[key] for item in samples)
                    if all(item.get("status", "pass") == "pass" for item in samples)
                    else None
                    for key in keys
                },
                "worst_round_mean_tpot_ms": max(
                    item["mean_tpot_ms"] for item in samples
                )
                if all(item["mean_tpot_ms"] is not None for item in samples)
                else None,
                "below_50ms_target": all(
                    item["mean_tpot_ms"] is not None and item["mean_tpot_ms"] < 50
                    for item in samples
                ),
                "meets_slo": all(
                    item.get("status", "pass") == "pass"
                    and item["mean_tpot_ms"] is not None
                    and item["mean_tpot_ms"] < limit
                    for item in samples
                ),
                "observed_decode_batch_max": max(batches) if batches else None,
            }
        )
    best = []
    for length in record["protocol"]["input_lengths"]:
        candidates = [
            point
            for point in points
            if point["input_tokens"] == length and point["meets_slo"]
        ]
        winner = (
            max(
                candidates,
                key=lambda point: (
                    point[keys[0]],
                    -point["client_concurrency"],
                ),
            )
            if candidates
            else None
        )
        best.append(
            {
                "input_tokens": length,
                "point": winner,
                "scope": "best measured point; extend scan if optimum is at an edge",
            }
        )
    return {
        "objective": record["objective"],
        "aggregation": "median per metric across rounds; SLO must pass every round",
        "points": points,
        "best": best,
    }


def compare(args):
    baselines, optimized = args.baseline, args.optimized
    if len(baselines) != len(optimized):
        raise ValueError("Pass matching baseline/optimized directories for each C")
    rows, points, configurations, seen = [], [], [], set()
    reference = None
    for baseline, opt in zip(baselines, optimized):
        base, tuned = [
            json.loads((directory / "performance.json").read_text(encoding="utf-8"))
            for directory in (baseline, opt)
        ]
        if (
            base["status"] != "pass"
            or tuned["status"] != "pass"
            or base["mode"] != "baseline"
            or tuned["mode"] != "lim_copy_sfa"
        ):
            raise ValueError(
                "Require complete baseline and lim_copy_sfa runs; failed requests cannot become speedups"
            )
        cache = int(base["launch"]["environment"]["SGLANG_NPU_SPARSE_KV_CACHE_SIZE"])
        if (
            cache in seen
            or str(cache)
            != tuned["launch"]["environment"]["SGLANG_NPU_SPARSE_KV_CACHE_SIZE"]
        ):
            raise ValueError("Require one matching A/B pair per cache budget")
        seen.add(cache)
        if reference is None:
            reference = base
        for candidate in (base, tuned):
            if set(candidate["protocol"]) != set(reference["protocol"]):
                raise ValueError(
                    "Different protocol keys; do not mix serving and decode runs"
                )
            for key in ("objective",):
                if candidate[key] != reference[key]:
                    raise ValueError(f"Different {key}")
            for key in reference["protocol"]:
                if (
                    key != "concurrencies"
                    and candidate["protocol"][key] != reference["protocol"][key]
                ):
                    raise ValueError(f"Different protocol: {key}")
            for key in SERVER_KEYS:
                if (
                    key not in TUNING_KEYS
                    and candidate["server_configuration"][key]
                    != reference["server_configuration"][key]
                ):
                    raise ValueError(f"Different server setting: {key}")
            for key in ("sglang_revision", "kernel_revision", "packages"):
                if candidate["launch"][key] != reference["launch"][key]:
                    raise ValueError(f"Different source/package: {key}")
            for key in ENV_KEYS:
                if key not in MODE_ENV + (
                    "SGLANG_NPU_SPARSE_KV_CACHE_SIZE",
                ) and candidate["launch"]["environment"].get(key) != reference[
                    "launch"
                ]["environment"].get(key):
                    raise ValueError(f"Different environment: {key}")
        summaries = [slo_summary(candidate) for candidate in (base, tuned)]
        points.extend(summaries[0]["points"] + summaries[1]["points"])
        configurations.append(
            {
                "cache_tokens": cache,
                "baseline": str(baseline),
                "optimized": str(opt),
                "baseline_configuration": base["server_configuration"],
                "optimized_configuration": tuned["server_configuration"],
                "baseline_actual_token_capacity": base.get("actual_token_capacity"),
                "optimized_actual_token_capacity": tuned.get("actual_token_capacity"),
                "baseline_memory": base.get("memory_after"),
                "optimized_memory": tuned.get("memory_after"),
            }
        )
        for left, right in zip(summaries[0]["best"], summaries[1]["best"]):
            b, o = left["point"], right["point"]
            row = {
                "cache_tokens": cache,
                "input_tokens": left["input_tokens"],
                "output_tokens": base["protocol"]["new_tokens"],
                "baseline_eligible": b is not None,
                "optimized_eligible": o is not None,
            }
            for prefix, point in (("baseline", b), ("optimized", o)):
                for key in (
                    "client_concurrency",
                    "observed_decode_batch_max",
                    *metric_keys(base),
                    "worst_round_mean_tpot_ms",
                ):
                    row[f"{prefix}_{key}"] = point[key] if point else None
            row["throughput_speedup"] = (
                o[metric_keys(base)[0]] / b[metric_keys(base)[0]] if b and o else None
            )
            rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "comparison.json",
        {
            "objective": reference["objective"],
            "scope": f"highest measured {metric_keys(reference)[0]} satisfying mean TPOT in every round; independent concurrency optima, same model/topology/memory fraction",
            "rows": rows,
            "points": points,
            "configurations": configurations,
        },
    )
    for name, data in (("comparison", rows), ("all-points", points)):
        with (args.output / f"{name}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    def cell(value):
        return "N/A" if value is None else f"{value:.2f}"

    metric = metric_keys(reference)[0]
    label = "Decode output" if metric == "decode_output_throughput" else "E2E output"
    text = f"| C | Input | Baseline clients | Baseline {label} tok/s | Baseline mean TPOT ms | HiSparse clients | HiSparse {label} tok/s | HiSparse mean TPOT ms | Throughput ratio |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for row in rows:
        text += f"| {row['cache_tokens']} | {row['input_tokens']} | {cell(row['baseline_client_concurrency'])} | {cell(row['baseline_' + metric])} | {cell(row['baseline_mean_tpot_ms'])} | {cell(row['optimized_client_concurrency'])} | {cell(row['optimized_' + metric])} | {cell(row['optimized_mean_tpot_ms'])} | {cell(row['throughput_speedup'])} |\n"
    text += "\nN/A = no valid measured point satisfies the bound (including capacity-limited batches). Best scanned point, not a proof of the global maximum. all-points.csv includes all scanned points; comparison.json records actual capacities/configurations and device-wide memory snapshots. Batch observations include warmup.\n"
    if label == "Decode output":
        text += "Decode output = tokens delivered between the last request's first token and the first request's finish, divided by that window. Mean TPOT is the arithmetic mean of each request's (finish - first token) / (output tokens - 1); steady_batch_step_ms = 1000 * batch / decode tok/s is a separate quantity.\n"
    else:
        text += "E2E output throughput includes prefill and queueing.\n"
    (args.output / "comparison.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    launch = subparsers.add_parser("capture-launch")
    launch.add_argument("--mode", choices=("baseline", "lim_copy_sfa"), required=True)
    launch.add_argument("--pid", type=int, required=True)
    launch.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-configs")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--context-length", type=int, default=135168)
    prepare.add_argument("--max-total-tokens", type=int)
    prepare.add_argument("--max-running-requests", type=int, default=16)
    prepare.add_argument("--concurrencies", type=int, nargs="+", default=[2, 4, 8, 16])
    bench = subparsers.add_parser("run")
    bench.add_argument("--mode", choices=("baseline", "lim_copy_sfa"), required=True)
    bench.add_argument(
        "--measurement", choices=("serving", "decode"), default="serving"
    )
    bench.add_argument("--url", default="http://127.0.0.1:8810")
    bench.add_argument("--results-root", type=Path, required=True)
    bench.add_argument("--launch-manifest", type=Path)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument(
        "--input-lengths", type=int, nargs="+", default=[131072, 65536, 16384]
    )
    bench.add_argument("--concurrencies", type=int, nargs="+", default=[2, 4, 8, 16])
    bench.add_argument("--new-tokens", type=int, default=1024)
    bench.add_argument("--rounds", type=int, default=1)
    bench.add_argument("--waves", type=int, default=2)
    bench.add_argument(
        "--requests-equal-concurrency",
        action="store_true",
        help="Set num_prompts=concurrency, including c1/c2; ignore waves/minimum four",
    )
    bench.add_argument("--case-timeout", type=int, default=14400)
    bench.add_argument("--tpot-limit-ms", type=float, default=60)
    report = subparsers.add_parser("compare")
    report.add_argument("--baseline", type=Path, nargs="+", required=True)
    report.add_argument("--optimized", type=Path, nargs="+", required=True)
    report.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args()
    {
        "capture-launch": capture_launch,
        "prepare-configs": prepare_configs,
        "run": run,
        "compare": compare,
    }[parsed.action](parsed)
