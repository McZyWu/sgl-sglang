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
import statistics
import subprocess
import sys
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
        args.context_length < 132096
        or args.max_total_tokens < args.context_length
        or args.max_total_tokens > 262144
    ):
        raise ValueError(
            "Require context >= 132096 and context <= max_total_tokens <= 262144 for the A5 C8 source pool"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    original = args.config.read_text(encoding="utf-8")
    for cache in (2048, 4096):
        for mode in ("baseline", "lim_copy_sfa"):
            target = args.output / f"c{cache}-{mode}.env"
            if target.exists():
                raise ValueError(f"Config already exists: {target}")
            footer = (
                "\n# Fixed long-context performance settings; override the base above.\n"
                f"LIM_EVAL_MODE={mode}\nCACHE_TOKENS={cache}\n"
                f"CONTEXT_LENGTH={args.context_length}\nMAX_TOTAL_TOKENS={args.max_total_tokens}\n"
                'C8_PERF_INPUT_LENGTHS="131072 65536 16384"\nC8_PERF_NEW_TOKENS=1024\n'
                'C8_PERF_CONCURRENCIES="1"\nC8_PERF_ROUNDS=3\nC8_PERF_WAVES=4\n'
            )
            target.write_text(original + footer, encoding="utf-8", newline="\n")
            target.chmod(0o600)
            print(target, flush=True)


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
                f"Unexpected {key}: require {count} requests of {expected} tokens"
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
        if (
            not isinstance(config["context_length"], int)
            or max(args.input_lengths) + args.new_tokens > config["context_length"]
        ):
            raise ValueError("Input + output exceeds the server context length")
        if (
            config["stream_interval"] != 1
            or config["speculative_algorithm"] is not None
        ):
            raise ValueError(
                "Require stream_interval=1 and speculative decoding disabled"
            )
        if (
            config["max_total_tokens"]
            and max(args.input_lengths) + args.new_tokens > config["max_total_tokens"]
        ):
            raise ValueError(
                "A single request exceeds max_total_tokens; generate the long-context perf configs first"
            )
        data = dataset_bytes(max(4, max(args.concurrencies) * args.waves))
        dataset = args.output / "prompts.json"
        dataset.write_bytes(data)
        record.update(
            launch_manifest=str(path),
            launch=launch,
            server_configuration=config,
            protocol={
                "input_lengths": args.input_lengths,
                "concurrencies": args.concurrencies,
                "new_tokens": args.new_tokens,
                "rounds": args.rounds,
                "waves": args.waves,
                "seed": 42,
                "dataset_sha256": hashlib.sha256(data).hexdigest(),
            },
        )
        write_json(args.output / "server_info.json", info)
        with (args.output / "npu-smi-before.txt").open("w", encoding="utf-8") as file:
            subprocess.run(
                ["npu-smi", "info"],
                stdout=file,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=30,
            )
        # Interleave cases across repetitions to expose time-dependent drift.
        for repeat in range(args.rounds):
            for length in args.input_lengths:
                for concurrency in args.concurrencies:
                    live_manifest(path, launch["port"])
                    count = max(4, concurrency * args.waves)
                    name = f"i{length}-o{args.new_tokens}-c{concurrency}-r{repeat + 1}"
                    raw_path = args.output / f"{name}.jsonl"
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
                    write_json(args.output / f"{name}.command.json", command)
                    print(
                        f"START {name}: {count} requests; warmup={concurrency}",
                        flush=True,
                    )
                    if (
                        config["max_total_tokens"]
                        and concurrency * (length + args.new_tokens)
                        > config["max_total_tokens"]
                    ):
                        print(
                            "NOTE: concurrent token demand exceeds max_total_tokens; server scheduling can reduce active batch size",
                            flush=True,
                        )
                    with (args.output / f"{name}.log").open(
                        "w", encoding="utf-8"
                    ) as log:
                        subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=args.case_timeout,
                        )
                    lines = raw_path.read_text(encoding="utf-8").splitlines()
                    if len(lines) != 1:
                        raise ValueError(f"Expected one benchmark record: {raw_path}")
                    result = json.loads(lines[0])
                    validate_result(result, count, length, args.new_tokens)
                    if {
                        key: result["server_info"].get(key) for key in SERVER_KEYS
                    } != config:
                        raise ValueError(
                            "Server configuration changed during benchmarking"
                        )
                    case = {
                        "input_tokens": length,
                        "concurrency": concurrency,
                        "repeat": repeat + 1,
                        "requests": count,
                        "raw_result": raw_path.name,
                        **{key: result[key] for key in METRICS},
                    }
                    record["cases"].append(case)
                    write_json(summary_path, record)
                    print(
                        f"PASS {name}: output={result['output_throughput']:.2f} tok/s; TTFT={result['mean_ttft_ms']:.2f} ms; TPOT={result['mean_tpot_ms']:.2f} ms",
                        flush=True,
                    )
        live_manifest(path, launch["port"])
        with (args.output / "npu-smi-after.txt").open("w", encoding="utf-8") as file:
            subprocess.run(
                ["npu-smi", "info"],
                stdout=file,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=30,
            )
        record["status"] = "pass"
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


def compare_pair(baseline, optimized):
    base = json.loads((baseline / "performance.json").read_text(encoding="utf-8"))
    opt = json.loads((optimized / "performance.json").read_text(encoding="utf-8"))
    if (
        base["status"] != "pass"
        or opt["status"] != "pass"
        or base["mode"] != "baseline"
        or opt["mode"] != "lim_copy_sfa"
    ):
        raise ValueError("Require a complete baseline and a complete lim_copy_sfa run")
    for key in ("protocol", "server_configuration"):
        if base[key] != opt[key]:
            raise ValueError(f"A/B mismatch in {key}; use identical settings")
    for key in ("sglang_revision", "kernel_revision", "packages"):
        if base["launch"][key] != opt["launch"][key]:
            raise ValueError(f"A/B source mismatch: {key}")
    for key in ENV_KEYS:
        if (
            key not in MODE_ENV
            and base["launch"]["environment"][key] != opt["launch"]["environment"][key]
        ):
            raise ValueError(f"A/B environment mismatch: {key}")
    left, right = grouped(base), grouped(opt)
    cache = int(base["launch"]["environment"]["SGLANG_NPU_SPARSE_KV_CACHE_SIZE"])
    rows = []
    for length, concurrency in sorted(left):
        b, o = left[length, concurrency], right[length, concurrency]
        row = {
            "cache_tokens": cache,
            "input_tokens": length,
            "output_tokens": base["protocol"]["new_tokens"],
            "concurrency": concurrency,
        }
        for key in METRICS:
            bv, ov = [item[key] for item in b], [item[key] for item in o]
            bm, om = statistics.median(bv), statistics.median(ov)
            row.update({f"baseline_{key}": bm, f"optimized_{key}": om})
            if key == "output_throughput":
                row.update(
                    throughput_speedup=om / bm,
                    throughput_gain_pct=(om / bm - 1) * 100,
                    baseline_throughput_min=min(bv),
                    baseline_throughput_max=max(bv),
                    optimized_throughput_min=min(ov),
                    optimized_throughput_max=max(ov),
                )
            elif key.endswith("_ms"):
                row[f"{key}_reduction_pct"] = (1 - om / bm) * 100
        rows.append(row)
    return rows, base


def compare(args):
    baselines = [args.baseline] if isinstance(args.baseline, Path) else args.baseline
    optimized = [args.optimized] if isinstance(args.optimized, Path) else args.optimized
    if len(baselines) != len(optimized):
        raise ValueError("Pass matching baseline/optimized directories for each C")
    rows, reference, seen = [], None, set()
    for baseline, opt in zip(baselines, optimized):
        pair, base = compare_pair(baseline, opt)
        cache = pair[0]["cache_tokens"]
        if cache in seen:
            raise ValueError(f"Repeated C={cache}; pass one A/B pair per cache budget")
        seen.add(cache)
        if reference is not None:
            for key in ("protocol", "server_configuration"):
                if base[key] != reference[key]:
                    raise ValueError(f"Different C groups use different {key}")
            for key in ("sglang_revision", "kernel_revision", "packages"):
                if base["launch"][key] != reference["launch"][key]:
                    raise ValueError(f"Different C groups use different {key}")
            for key in ENV_KEYS:
                if (
                    key not in MODE_ENV + ("SGLANG_NPU_SPARSE_KV_CACHE_SIZE",)
                    and base["launch"]["environment"][key]
                    != reference["launch"]["environment"][key]
                ):
                    raise ValueError(f"Different C groups use different {key}")
        reference = base
        rows.extend(pair)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "comparison.json",
        {
            "baseline": [str(path) for path in baselines],
            "optimized": [str(path) for path in optimized],
            "aggregation": "median across repetitions; no pooling across workloads",
            "rows": rows,
        },
    )
    with (args.output / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = "| C | Input | Output | Concurrency | Baseline tok/s | LIM+Copy tok/s | Speedup | TPOT reduction | TTFT reduction |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for row in rows:
        text += f"| {row['cache_tokens']} | {row['input_tokens']} | {row['output_tokens']} | {row['concurrency']} | {row['baseline_output_throughput']:.2f} | {row['optimized_output_throughput']:.2f} | {row['throughput_speedup']:.3f}x | {row['mean_tpot_ms_reduction_pct']:+.2f}% | {row['mean_ttft_ms_reduction_pct']:+.2f}% |\n"
    text += "\nPositive reduction = lower latency; negative = regression. Throughput includes prefill, scheduling and decode. P95 values in CSV/JSON are medians of per-run P95 values.\n"
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
    prepare.add_argument("--context-length", type=int, default=132096)
    prepare.add_argument("--max-total-tokens", type=int, default=147456)
    bench = subparsers.add_parser("run")
    bench.add_argument("--mode", choices=("baseline", "lim_copy_sfa"), required=True)
    bench.add_argument("--url", default="http://127.0.0.1:8810")
    bench.add_argument("--results-root", type=Path, required=True)
    bench.add_argument("--launch-manifest", type=Path)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument(
        "--input-lengths", type=int, nargs="+", default=[131072, 65536, 16384]
    )
    bench.add_argument("--concurrencies", type=int, nargs="+", default=[1])
    bench.add_argument("--new-tokens", type=int, default=1024)
    bench.add_argument("--rounds", type=int, default=3)
    bench.add_argument("--waves", type=int, default=4)
    bench.add_argument("--case-timeout", type=int, default=3600)
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
