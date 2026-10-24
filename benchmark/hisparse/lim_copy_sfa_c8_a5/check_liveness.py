"""Alternate exact-length short/long requests against an already running server.

This checks completion and path transitions, not model accuracy or throughput.
Uses only the standard library; request timeouts are errors and never retried.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def request_json(url, path, payload, timeout):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"HTTP {error.code} {path}: {error.read().decode(errors='replace')[:2000]}"
        ) from error


def run(args):
    for name in (
        "rounds",
        "concurrency",
        "long_tokens",
        "cache_tokens",
        "new_tokens",
        "timeout",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.long_tokens <= args.cache_tokens:
        raise ValueError("long_tokens must exceed cache_tokens to exercise offload")
    if args.cache_tokens < 2048:
        raise ValueError("C8 validation requires cache_tokens >= 2048")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "running",
        "configuration": vars(args) | {"output": str(args.output)},
        "requests": [],
    }

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        info = request_json(args.url, "/server_info", None, args.timeout)
        record["server_info"] = {
            key: info.get(key)
            for key in (
                "context_length",
                "tp_size",
                "ep_size",
                "attn_dp_size",
                "kv_cache_dtype",
                "attention_backend",
                "moe_a2a_backend",
                "deepep_mode",
                "disable_overlap_schedule",
                "enable_prefill_delayer",
                "version",
                "frontend",
            )
        }
        tokens = request_json(
            args.url,
            "/tokenize",
            {
                "model": args.model,
                "prompt": "Read these notes and summarize the key facts. ",
                "add_special_tokens": False,
            },
            args.timeout,
        )
        seed = tokens["tokens"]
        if not seed or not all(isinstance(token, int) for token in seed):
            raise ValueError("/tokenize returned no flat token ID list")
        # Tokenizer model_max_length may be -1 or much larger than the server's
        # configured context. Prefer the resolved launch setting when present.
        limits = [
            value
            for value in (info.get("context_length"), tokens.get("max_model_len"))
            if isinstance(value, int) and value > 0
        ]
        if limits and args.long_tokens + args.new_tokens > min(limits):
            raise ValueError(
                "long_tokens + new_tokens exceeds the server context length"
            )
        prefix = "c8-manual-" + uuid.uuid4().hex[:12]

        def generate(item):
            rid, length = item
            ids = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
            start = time.monotonic()
            try:
                result = request_json(
                    args.url,
                    "/generate",
                    {
                        "rid": rid,
                        "input_ids": ids,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": args.new_tokens,
                            "ignore_eos": True,
                        },
                        "stream": False,
                    },
                    args.timeout,
                )
                meta = result["meta_info"]
                if (
                    meta.get("prompt_tokens") != length
                    or meta.get("completion_tokens") != args.new_tokens
                ):
                    raise ValueError(f"Unexpected token counts: {meta}")
                if meta.get("finish_reason", {}).get("type") != "length":
                    raise ValueError(f"Request did not finish by length: {meta}")
                return {
                    "rid": rid,
                    "input_tokens": length,
                    "output_tokens": args.new_tokens,
                    "seconds": time.monotonic() - start,
                    "status": "pass",
                    "text_sample": result.get("text", "")[:160],
                }
            except Exception as error:
                return {
                    "rid": rid,
                    "input_tokens": length,
                    "status": "fail",
                    "error": str(error),
                    "seconds": time.monotonic() - start,
                }

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for round_id in range(args.rounds):
                # Finish each wave before switching paths: dense -> sparse ->
                # dense on the next round. Different short lengths mix padding.
                for kind in ("short", "long"):
                    items = [
                        (
                            f"{prefix}-{round_id}-{kind}-{i}",
                            (142, 257, 384, 465)[i % 4]
                            if kind == "short"
                            else args.long_tokens,
                        )
                        for i in range(args.concurrency)
                    ]
                    results = list(pool.map(generate, items))
                    record["requests"].extend(results)
                    save()
                    failures = [
                        result for result in results if result["status"] != "pass"
                    ]
                    print(
                        f"round={round_id + 1} kind={kind} completed={len(results) - len(failures)}/{len(results)}",
                        flush=True,
                    )
                    if failures:
                        raise RuntimeError(f"Request failure: {failures[0]}")
        record["status"] = "pass"
        record["completed_requests"] = len(record["requests"])
        record["generated_tokens"] = sum(
            item["output_tokens"] for item in record["requests"]
        )
        record["scope"] = (
            "HTTP liveness only; inspect server logs for eager/graph and overlap configuration"
        )
        print(
            f"PASS: {record['completed_requests']} requests, {record['generated_tokens']} generated tokens; {args.output}"
        )
    except Exception as error:
        record.update(status="fail", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8810")
    parser.add_argument("--model", default="c8-a5")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--long-tokens", type=int, default=4096)
    parser.add_argument("--cache-tokens", type=int, default=2048)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
