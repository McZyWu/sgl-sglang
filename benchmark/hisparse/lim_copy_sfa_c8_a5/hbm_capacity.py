"""Find per-mode admitted batches under a common sampled per-card HBM limit.

Runs inside the A5 container. Owns only the servers it launches. Uses the
standard bench_serving output throughput and request TPOT, not pure decode TPS.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib.metadata import version
from pathlib import Path

import performance as perf
import psutil
from monitor_hbm import Sampler, read_hbm

SCRIPT = Path(__file__).resolve().parent
RESOURCE_FAILURES = {"hbm_limit", "out_of_memory", "token_pool_limited"}
BASE_KEYS = (
    "MODEL_PATH",
    "TP_SIZE",
    "EP_SIZE",
    "ATTN_DP_SIZE",
    "PORT",
    "HOST",
    "MOE_A2A_BACKEND",
    "DEEPEP_MODE",
    "DEEPEP_HYBRID_DEPLOYMENT",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
    "HCCL_BUFFSIZE",
    "DEEPEP_HCCL_BUFFSIZE",
    "DEEP_USE_MODE",
    "WEIGHT_QUANTIZATION",
    "CHUNKED_PREFILL_SIZE",
    "MAX_PREFILL_TOKENS",
    "ENABLE_PREFILL_DELAYER",
    "PREFILL_DELAYER_MAX_DELAY_PASSES",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM",
    "KERNEL_SOURCE_DIR",
)


def base_settings(path):
    code = (
        "import json,os; print(json.dumps({k:os.environ.get(k) for k in "
        + repr(BASE_KEYS)
        + "}))"
    )
    output = subprocess.check_output(
        ["bash", "-c", 'set -a; source "$1"; python3 -c "$2"', "bash", str(path), code],
        text=True,
    )
    return json.loads(output)


def remember_children(process):
    if process is None or process.poll() is not None:
        return
    children = getattr(process, "owned_children", {})
    try:
        for child in psutil.Process(process.pid).children(recursive=True):
            children[(child.pid, child.create_time())] = child
    except psutil.NoSuchProcess:
        pass
    process.owned_children = children


def stop_owned(process):
    if process is None:
        return
    remember_children(process)
    # psutil.Process retains the creation time, so a reused PID cannot become
    # an unrelated termination target. Remembered workers may have their own SID.
    children = list(getattr(process, "owned_children", {}).values())
    if process.poll() is None:
        children.append(psutil.Process(process.pid))
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=10)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=10)


def card_peak(summary):
    return max(
        (group["max_card_peak_mib"] for group in summary["cases"].values()),
        default=None,
    )


def host_kv_estimate(model_path, batch, length, output, ranks):
    config = json.loads((Path(model_path) / "config.json").read_text())
    config = config.get("text_config") or config
    layers = config.get("num_hidden_layers") or config.get("num_layers")
    if not isinstance(layers, int) or layers <= 0:
        raise ValueError("Cannot estimate host KV: model config has no layer count")
    context = ((length + output + 129) // 128) * 128
    # Request row zero and the extra req_to_token columns are also reserved.
    stride = ((context + 4 + 127) // 128) * 128
    return (batch + 1) * stride * (656 * layers + 4) * ranks


def classify_point(info, measured, batch, required, peak, limit, errors):
    if errors or peak is None:
        return "hbm_monitor_error"
    if peak > limit:
        return "hbm_limit"
    capacity = info.get("max_total_num_tokens")
    if capacity is None or capacity < required:
        return "token_pool_limited"
    if not measured or any(row.get("max") != batch for row in measured):
        return "batch_not_observed"
    return "pass"


def config_text(original, output, mode, cache, length, batch, args):
    context = ((length + args.new_tokens + 2 + 127) // 128) * 128
    graph = "1" if batch == 1 else f"1 {batch}"
    # Explicitly allocate only the tested batch's full-context token pool. The
    # common sampled HBM ceiling, rather than an automatic 0.8 pool, is binding.
    return (
        original
        + (
            f"\n# Controlled HBM experiment: {mode}, C{cache}, L{length}, B{batch}\n"
            f"LIM_EVAL_MODE={mode}\nCACHE_TOKENS={cache}\n"
            f"CONTEXT_LENGTH={context}\nMAX_RUNNING_REQUESTS={batch}\n"
            f"MAX_TOTAL_TOKENS={context * batch}\nMEM_FRACTION_STATIC={args.mem_fraction}\n"
            f'GRAPH_BATCH_SIZES="{graph}"\n'
        )
        + (
            f"C8_RESULTS_DIR={str(output / 'server')!r}\n"
            "C8_TRACE_OFFLOAD=0\nSGLANG_DEBUG_NPU_C8_OFFLOAD=0\n"
        )
    )


def probe(args, original, settings, mode, cache, length, batch):
    point = args.output / f"c{cache}-{mode}" / f"i{length}-o{args.new_tokens}-b{batch}"
    point.mkdir(parents=True, exist_ok=False)
    config = point / "config.env"
    config.write_text(config_text(original, point, mode, cache, length, batch, args))
    config.chmod(0o600)
    port = int(settings.get("PORT") or 8810)
    host = settings.get("HOST") or "127.0.0.1"
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Capacity harness requires HOST=127.0.0.1 or localhost")
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(
                f"Port {port} is occupied; stop your existing test server first"
            )
    label = f"c{cache}-{mode}-i{length}-b{batch}"
    sampler = Sampler(point / "hbm", args.devices, args.interval, label=label).start()
    server = client = None
    info = {}
    result = dict(
        mode=mode,
        cache_tokens=cache,
        input_tokens=length,
        output_tokens=args.new_tokens,
        batch=batch,
        directory=str(point),
        status="starting",
    )
    url = f"http://127.0.0.1:{port}"
    env = os.environ | {"C8_CONFIG": str(config), "PYTHONUNBUFFERED": "1"}
    try:
        if mode == "lim_copy_sfa":
            try:
                estimate = host_kv_estimate(
                    settings["MODEL_PATH"],
                    batch,
                    length,
                    args.new_tokens,
                    int(settings["TP_SIZE"] or 1),
                )
            except (OSError, ValueError) as error:
                result.update(status="host_config_error", error=str(error))
                return result
            result["host_kv_estimate_bytes"] = estimate
            available = psutil.virtual_memory().available
            shm_free = psutil.disk_usage("/dev/shm").free
            result.update(host_available_bytes=available, shm_free_bytes=shm_free)
            if estimate * 1.1 + 64 * 1024**3 > available or estimate > shm_free:
                result["status"] = "host_memory_limited"
                return result
        with (point / "launch.log").open("w") as log:
            server = subprocess.Popen(
                ["bash", str(SCRIPT / "container.sh"), "serve"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + args.startup_timeout
            while True:
                remember_children(server)
                snapshot = sampler.summary()
                peak = card_peak(snapshot)
                if snapshot["errors"]:
                    result["status"] = "hbm_monitor_error"
                    return result
                if peak is not None and peak > args.limit_mib:
                    result["status"] = "hbm_limit"
                    return result
                if server.poll() is not None:
                    text = (point / "launch.log").read_text(errors="replace")
                    result["status"] = (
                        "out_of_memory"
                        if re.search(r"out.of.memory|OutOfMemoryError", text, re.I)
                        else "startup_failed"
                    )
                    return result
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    result["status"] = "startup_timeout"
                    return result
                time.sleep(0.25)
            info = perf.server_info(url)
            if (info.get("dp_size") or 1) != 1 or (info.get("attn_dp_size") or 1) != 1:
                raise ValueError(
                    "This admitted-batch experiment currently requires attention DP1"
                )
            required = ((length + args.new_tokens + 2 + 127) // 128) * 128 * batch
            result["actual_token_capacity"] = info.get("max_total_num_tokens")
            if (
                not result["actual_token_capacity"]
                or result["actual_token_capacity"] < required
            ):
                result["status"] = "token_pool_limited"
                return result
            command = [
                sys.executable,
                str(SCRIPT / "performance.py"),
                "run",
                "--mode",
                mode,
                "--url",
                url,
                "--results-root",
                str(point / "server"),
                "--output",
                str(point / "perf"),
                "--input-lengths",
                str(length),
                "--concurrencies",
                str(batch),
                "--new-tokens",
                str(args.new_tokens),
                "--rounds",
                str(args.rounds),
                "--requests-equal-concurrency",
                "--waves",
                "1",
                "--case-timeout",
                str(args.case_timeout),
                "--tpot-limit-ms",
                str(args.tpot_limit_ms),
            ]
            perf.write_json(point / "command.json", command)
            with (point / "client.log").open("w") as client_log:
                client = subprocess.Popen(
                    command,
                    env=env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                deadline = time.monotonic() + args.case_timeout * args.rounds + 60
                while client.poll() is None:
                    remember_children(server)
                    remember_children(client)
                    snapshot = sampler.summary()
                    if (
                        snapshot["errors"]
                        or (card_peak(snapshot) or 0) > args.limit_mib
                    ):
                        result["status"] = (
                            "hbm_monitor_error" if snapshot["errors"] else "hbm_limit"
                        )
                        return result
                    if server.poll() is not None or time.monotonic() > deadline:
                        result["status"] = "runtime_failed"
                        return result
                    time.sleep(0.25)
            if client.returncode:
                logs = (point / "client.log").read_text(errors="replace") + (
                    point / "launch.log"
                ).read_text(errors="replace")
                result["status"] = (
                    "out_of_memory"
                    if re.search(r"out.of.memory|OutOfMemoryError", logs, re.I)
                    else "benchmark_failed"
                )
                return result
            record = json.loads((point / "perf/performance.json").read_text())
            samples = record["cases"]
            result.update(
                output_throughput=statistics.median(
                    row["output_throughput"] for row in samples
                ),
                mean_tpot_ms=statistics.median(row["mean_tpot_ms"] for row in samples),
                p95_tpot_ms=statistics.median(row["p95_tpot_ms"] for row in samples),
                worst_round_mean_tpot_ms=max(row["mean_tpot_ms"] for row in samples),
                measured_decode_batches=[
                    row["measured_decode_batch"] for row in samples
                ],
                launch=record["launch"],
                server_configuration=record["server_configuration"],
            )
            result["status"] = classify_point(
                info,
                result["measured_decode_batches"],
                batch,
                required,
                card_peak(sampler.summary()),
                args.limit_mib,
                sampler.summary()["errors"],
            )
            return result
    finally:
        stop_owned(client)
        stop_owned(server)
        telemetry = sampler.stop()
        result["peak_hbm_mib"] = card_peak(telemetry)
        if result["status"] == "pass" and (
            telemetry["errors"] or result["peak_hbm_mib"] > args.limit_mib
        ):
            result["status"] = (
                "hbm_monitor_error" if telemetry["errors"] else "hbm_limit"
            )
        result["hbm_by_npu"] = telemetry["cases"].get(label, {}).get("by_npu", {})
        perf.write_json(point / "point.json", result)
        print(
            f"{label}: {result['status']}; peak={result['peak_hbm_mib']} MiB; output={result.get('output_throughput')} tok/s; TPOT={result.get('mean_tpot_ms')} ms",
            flush=True,
        )


def search(batches, evaluate, refine=True):
    """Return the tested bracket; never interpret an unobserved batch as HBM OOM."""
    points, low, high, reason = [], 0, None, "scan_edge"
    for batch in batches:
        point = evaluate(batch)
        points.append(point)
        if point["status"] == "pass":
            low = batch
        else:
            reason = point["status"]
            high = batch
            break
    if refine and high is not None and reason in RESOURCE_FAILURES:
        while high - low > 1:
            middle = (low + high) // 2
            point = evaluate(middle)
            points.append(point)
            if point["status"] == "pass":
                low = middle
            elif point["status"] in RESOURCE_FAILURES:
                high, reason = middle, point["status"]
            else:
                reason = point["status"]
                break
    return dict(
        points=points,
        max_admitted_batch=low or None,
        next_rejected_batch=high,
        boundary_reason=reason,
        scope="tested local bracket; resource monotonicity assumed during refinement; scan_edge is not an HBM limit",
    )


def report(record, output):
    lines = [
        "| C | Input | Mode | Max admitted B | Output tok/s at max B | Mean TPOT ms | Peak GiB/card | Boundary | Best output tok/s (<TPOT limit) | B |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group in record["groups"]:
        valid = [point for point in group["points"] if point["status"] == "pass"]
        maximum = max(valid, key=lambda point: point["batch"]) if valid else None
        candidates = [
            point
            for point in valid
            if point["worst_round_mean_tpot_ms"] < record["settings"]["tpot_limit_ms"]
        ]
        best = (
            max(candidates, key=lambda point: point["output_throughput"])
            if candidates
            else None
        )
        fastest = (
            max(valid, key=lambda point: point["output_throughput"]) if valid else None
        )
        group.update(
            max_batch_point=maximum, best_tpot_point=best, fastest_point=fastest
        )

        def cell(point, key, scale=1):
            return "N/A" if not point else f"{point[key] / scale:.2f}"

        lines.append(
            f"| {group['cache_tokens']} | {group['input_tokens']} | {group['mode']} | {cell(maximum, 'batch')} | {cell(maximum, 'output_throughput')} | {cell(maximum, 'mean_tpot_ms')} | {cell(maximum, 'peak_hbm_mib', 1024)} | {group.get('boundary_reason', 'in_progress')} | {cell(best, 'output_throughput')} | {cell(best, 'batch')} |"
        )
    pairs = {}
    for group in record["groups"]:
        pairs.setdefault((group["cache_tokens"], group["input_tokens"]), {})[
            group["mode"]
        ] = group
    comparisons = []
    lines += [
        "",
        "| C | Input | Selection | Baseline B | HiSparse B | Baseline output tok/s | HiSparse output tok/s | Ratio | Baseline TPOT ms | HiSparse TPOT ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (cache, length), modes in sorted(pairs.items()):
        if set(modes) != {"baseline", "lim_copy_sfa"}:
            continue
        for criterion in ("max_batch_point", "fastest_point", "best_tpot_point"):
            base, opt = (
                modes[mode][criterion] for mode in ("baseline", "lim_copy_sfa")
            )
            row = dict(cache_tokens=cache, input_tokens=length, selection=criterion)
            for name, point in (("baseline", base), ("optimized", opt)):
                for key in (
                    "batch",
                    "output_throughput",
                    "mean_tpot_ms",
                    "peak_hbm_mib",
                ):
                    row[f"{name}_{key}"] = point[key] if point else None
            row["output_throughput_ratio"] = (
                opt["output_throughput"] / base["output_throughput"]
                if base and opt
                else None
            )
            comparisons.append(row)
            ratio = f"{row['output_throughput_ratio']:.3f}" if base and opt else "N/A"
            lines.append(
                f"| {cache} | {length} | {criterion} | {cell(base, 'batch')} | {cell(opt, 'batch')} | {cell(base, 'output_throughput')} | {cell(opt, 'output_throughput')} | {ratio} | {cell(base, 'mean_tpot_ms')} | {cell(opt, 'mean_tpot_ms')} |"
            )
    record["comparisons"] = comparisons
    output.mkdir(parents=True, exist_ok=True)
    perf.write_json(output / "capacity.json", record)
    text = (
        "\n".join(lines)
        + "\n\nOutput throughput uses standard bench_serving total wall time. HBM is a sampled device-wide peak, including startup/graph capture/warmup/prefill/decode. Max admitted B requires exact request completion, pool capacity and a measured decode log sample at B; it does not claim B persisted throughout decode. Non-memory failures do not prove an HBM limit.\n"
    )
    (output / "capacity.md").write_text(text)
    rows = [point for group in record["groups"] for point in group["points"]]
    with (output / "all-points.csv").open("w", newline="") as file:
        fields = [
            "mode",
            "cache_tokens",
            "input_tokens",
            "batch",
            "status",
            "output_throughput",
            "mean_tpot_ms",
            "p95_tpot_ms",
            "peak_hbm_mib",
            "actual_token_capacity",
            "directory",
        ]
        writer = csv.DictWriter(file, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(text, flush=True)


def compare_runs(args):
    record = None
    seen = set()
    for directory in args.compare:
        item = json.loads((directory / "capacity.json").read_text())
        if item["status"] != "complete":
            raise ValueError("Capacity comparison requires completed scans")
        if record is None:
            record = dict(status="complete", settings=item["settings"], groups=[])
        if item["settings"] != record["settings"]:
            raise ValueError(
                "Different source/model/topology/HBM budget/protocol; do not mix capacity runs"
            )
        for group in item["groups"]:
            key = (group["cache_tokens"], group["input_tokens"], group["mode"])
            if key in seen:
                raise ValueError(f"Repeated capacity group: {key}")
            seen.add(key)
            record["groups"].append(group)
    report(record, args.output)


def main(args):
    if (args.output / "capacity.json").exists():
        raise ValueError("Use a fresh output directory")
    settings = base_settings(args.base_config)
    if settings["MOE_A2A_BACKEND"] == "deepep" and max(args.batches) > int(
        settings["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] or 32
    ):
        raise ValueError(
            "Batch scan exceeds configured DeepEP dispatch capacity; prepare/validate communication capacity before extending the scan"
        )
    args.limit_mib = args.hbm_limit_gib * 1024
    if (
        not 0 < args.mem_fraction < 1
        or not math.isfinite(args.hbm_limit_gib)
        or args.hbm_limit_gib <= 0
        or min(args.batches + args.input_lengths + [args.rounds, args.new_tokens]) <= 0
    ):
        raise ValueError("Invalid memory fraction, limit, batch or workload")
    aligned_context = ((max(args.input_lengths) + args.new_tokens + 129) // 128) * 128
    if args.new_tokens < 2 or aligned_context > perf.MAX_C8_CONTEXT:
        raise ValueError(
            "Output must be >=2 and aligned context must fit native C8 block tables"
        )
    cards = read_hbm()
    if any(
        str(device) not in cards or args.limit_mib > cards[str(device)]["total_mib"]
        for device in args.devices
    ):
        raise ValueError(
            "HBM limit exceeds a selected device's capacity or a device is missing"
        )
    args.batches = sorted(set(args.batches))
    args.output = args.output.resolve()
    record = dict(
        status="running",
        settings=dict(
            base=settings,
            hbm_limit_gib=args.hbm_limit_gib,
            mem_fraction=args.mem_fraction,
            devices=args.devices,
            interval=args.interval,
            new_tokens=args.new_tokens,
            rounds=args.rounds,
            tpot_limit_ms=args.tpot_limit_ms,
            sglang_revision=perf.git_revision(perf.REPO),
            kernel_revision=perf.git_revision(settings["KERNEL_SOURCE_DIR"]),
            packages={
                name: version(name) for name in ("torch", "torch-npu", "sgl-kernel-npu")
            },
            dataset_sha256=hashlib.sha256(perf.dataset_bytes(1024)).hexdigest(),
        ),
        groups=[],
    )
    original = args.base_config.read_text()
    try:
        for cache in args.caches:
            for length in args.input_lengths:
                for mode in args.modes:
                    group = dict(
                        cache_tokens=cache, input_tokens=length, mode=mode, points=[]
                    )
                    record["groups"].append(group)

                    def evaluate(batch):
                        point = probe(
                            args, original, settings, mode, cache, length, batch
                        )
                        group["points"].append(point)
                        report(record, args.output)
                        return point

                    found = search(args.batches, evaluate, not args.no_refine)
                    group.update(found)
                    report(record, args.output)
                    if group["boundary_reason"] not in RESOURCE_FAILURES | {
                        "scan_edge",
                        "batch_not_observed",
                    }:
                        raise RuntimeError(
                            f"Non-capacity failure: {group['boundary_reason']}; inspect point logs"
                        )
        record["status"] = "complete"
    finally:
        report(record, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--compare", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        choices=("baseline", "lim_copy_sfa"),
        nargs="+",
        default=["baseline", "lim_copy_sfa"],
    )
    parser.add_argument("--caches", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument(
        "--input-lengths", type=int, nargs="+", default=[131072, 65536, 16384]
    )
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32]
    )
    parser.add_argument("--hbm-limit-gib", type=float, default=90)
    parser.add_argument("--mem-fraction", type=float, default=0.98)
    parser.add_argument("--devices", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--new-tokens", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--tpot-limit-ms", type=float, default=60)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--case-timeout", type=int, default=14400)
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare_runs(args)
    elif args.base_config:
        main(args)
    else:
        parser.error("Pass --base-config to run or --compare to combine scans")
