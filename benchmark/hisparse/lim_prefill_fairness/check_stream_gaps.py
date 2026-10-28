"""Replay frozen text-tokenized requests and retain every streaming gap.

The input is JSONL, optionally gzip, with one ``token_ids`` array per request.
Use the same input file for interval=0 and interval=1. No model patch or
profiling is involved. Client gaps are service latency, not isolated NPU time.
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


def request_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def run_one(base_url, token_ids, output_tokens, timeout, index):
    request_id = f"lim-fair-{time.time_ns()}-{index}"
    body = {
        "rid": request_id,
        "input_ids": token_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    req = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start_unix = time.time()
    start = time.perf_counter()
    events = []
    text = ""
    error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                meta = data.get("meta_info", {})
                count = int(meta.get("completion_tokens", 0))
                if count and (not events or count > events[-1]["tokens"]):
                    events.append(
                        {
                            "seconds": time.perf_counter() - start,
                            "unix_time": time.time(),
                            "tokens": count,
                        }
                    )
                text = data.get("text", text)
                finish = meta.get("finish_reason")
                if isinstance(finish, dict) and finish.get("type") in (
                    "abort",
                    "error",
                ):
                    raise RuntimeError(str(finish))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    duration = time.perf_counter() - start
    itls = []
    gaps = []
    for previous, current in zip(events, events[1:]):
        gap = current["seconds"] - previous["seconds"]
        count = current["tokens"] - previous["tokens"]
        itls.extend([gap / count] * count)
        gaps.append(
            {"after_token": previous["tokens"], "new_tokens": count, "seconds": gap}
        )
    completed = events[-1]["tokens"] if events else 0
    if error is None and completed != output_tokens:
        error = f"expected {output_tokens} output tokens, received {completed}"
    return {
        "request_index": index,
        "request_id": request_id,
        "start_unix": start_unix,
        "input_tokens": len(token_ids),
        "input_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "output_tokens": completed,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "ttft_seconds": events[0]["seconds"] if events else None,
        "e2e_seconds": duration,
        "tpot_seconds": (
            (duration - events[0]["seconds"]) / (completed - 1)
            if events and completed > 1
            else None
        ),
        "events": events,
        "itls_seconds": itls,
        "gaps": gaps,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18888")
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-interval", required=True, type=int)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=32768)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if (
        min(args.num_requests, args.concurrency, args.input_tokens, args.output_tokens)
        < 1
    ):
        parser.error("request counts and token lengths must be positive")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite prior results")
    opener = gzip.open if args.requests.suffix == ".gz" else open
    inputs = []
    with opener(args.requests, "rt", encoding="utf-8") as stream:
        for line in stream:
            if len(inputs) == args.num_requests:
                break
            row = json.loads(line)
            tokens = row["token_ids"][: args.input_tokens]
            if len(tokens) != args.input_tokens:
                parser.error("frozen request is shorter than --input-tokens")
            inputs.append(tokens)
    if len(inputs) != args.num_requests:
        parser.error("not enough frozen requests")
    server = request_json(args.base_url + "/server_info")
    config = server.get("server_args", server)
    if config.get("prefill_decode_interval") != args.expected_interval:
        raise RuntimeError(
            "server prefill_decode_interval differs from the requested case"
        )
    if config.get("enable_mixed_chunk"):
        raise RuntimeError(
            "this comparison requires ordinary decode, with mixed chunk off"
        )
    if config.get("speculative_algorithm") is not None:
        raise RuntimeError("this comparison requires MTP off")
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_one, args.base_url, tokens, args.output_tokens, args.timeout, index
            )
            for index, tokens in enumerate(inputs)
        ]
        results = [future.result() for future in futures]
    duration = time.perf_counter() - start
    itls = [value for row in results for value in row["itls_seconds"]]
    gaps = [gap for row in results for gap in row["gaps"]]
    steady_itls = [
        gap["seconds"] / gap["new_tokens"]
        for gap in gaps
        if gap["after_token"] >= 3
        for _ in range(gap["new_tokens"])
    ]
    tpot = [row["tpot_seconds"] for row in results if row["tpot_seconds"] is not None]
    ttft = [row["ttft_seconds"] for row in results if row["ttft_seconds"] is not None]
    summary = {
        "prefill_decode_interval": args.expected_interval,
        "requests": len(results),
        "completed": sum(row["error"] is None for row in results),
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "concurrency": args.concurrency,
        "duration_seconds": duration,
        "output_throughput": sum(row["output_tokens"] for row in results) / duration,
        "mean_tpot_ms": statistics.mean(tpot) * 1000 if tpot else None,
        "mean_ttft_ms": statistics.mean(ttft) * 1000 if ttft else None,
        "mean_e2e_ms": statistics.mean(row["e2e_seconds"] for row in results) * 1000,
        "median_itl_ms": statistics.median(itls) * 1000 if itls else None,
        "median_itl_after_token_3_ms": (
            statistics.median(steady_itls) * 1000 if steady_itls else None
        ),
        "mean_itl_after_token_3_ms": (
            statistics.mean(steady_itls) * 1000 if steady_itls else None
        ),
        "max_chunk_gap_seconds": max((gap["seconds"] for gap in gaps), default=0),
        "max_first_two_chunk_gap_seconds": max(
            (gap["seconds"] for gap in gaps if gap["after_token"] <= 2), default=0
        ),
        "chunk_gaps_gt_1s": sum(gap["seconds"] > 1 for gap in gaps),
        "chunk_gaps_gt_10s": sum(gap["seconds"] > 10 for gap in gaps),
        "note": (
            "All gaps retained. Chunk gap and adjusted per-token ITL are separate metrics. "
            "After-token-3 statistics still include any later prefill interference."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"summary": summary, "server_info": server, "results": results}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["completed"] != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
