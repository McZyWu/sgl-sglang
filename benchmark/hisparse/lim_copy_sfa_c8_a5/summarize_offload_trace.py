"""Summarize C8_OFFLOAD_TRACE counts, never average cumulative percentages.

latest: last snapshot per rank, covering the service lifetime.
window: last minus first snapshot per rank after --offset; unsampled edges
are excluded. Use one service log and no concurrent workloads during GSM8K.
"""

import argparse
import json
import re
from pathlib import Path

COUNTERS = (
    "prefix_selected_tokens",
    "selected_miss_tokens",
    "invalid_miss_plan_rows",
    "tail_selected_tokens",
    "cache_refill_tokens",
    "sparse_request_layer_rows",
    "decode_layer_calls",
    "decode_request_layer_rows",
    "decode_d2h_tokens",
    "prefill_d2h_tokens",
    "prefill_h2d_tokens",
)


def parse_trace(text):
    ranks = {}
    for number, line in enumerate(text.splitlines(), 1):
        before, found, payload = line.partition("C8_OFFLOAD_TRACE ")
        if not found or not payload.lstrip().startswith("{"):
            continue
        sample = json.loads(payload)
        for key in (*COUNTERS, "forwards", "max_decode_seq_len"):
            if type(sample.get(key)) is not int or sample[key] < 0:
                raise ValueError(
                    f"Line {number}: missing/invalid {key}; require current C8 trace format"
                )
        rank = " ".join(re.findall(r"\b(?:TP|EP|DP|PP)\d+\b", before)) or "unranked"
        series = ranks.setdefault(rank, [])
        if series:
            previous = series[-1]
            keys = (*COUNTERS, "forwards", "max_decode_seq_len")
            if any(sample[key] < previous[key] for key in keys):
                raise ValueError(
                    f"Line {number}: counters reset for {rank}; use a single service launch log"
                )
            if sample["forwards"] == previous["forwards"]:
                if any(sample[key] != previous[key] for key in keys):
                    raise ValueError(
                        f"Line {number}: conflicting duplicate snapshot for {rank}"
                    )
                continue
        series.append(sample)
    return ranks


def rate(counts):
    selected, misses = counts["prefix_selected_tokens"], counts["selected_miss_tokens"]
    if counts["invalid_miss_plan_rows"] or misses > selected:
        return {
            "status": "invalid_plan",
            "prefix_cache_hits": None,
            "prefix_cache_hit_rate_pct": None,
        }
    if selected == 0:
        return {
            "status": "no_cache_management_samples",
            "prefix_cache_hits": None,
            "prefix_cache_hit_rate_pct": None,
        }
    return {
        "status": "ok",
        "prefix_cache_hits": selected - misses,
        "prefix_cache_hit_rate_pct": 100 * (selected - misses) / selected,
    }


def summarize(ranks, scope):
    rows, total, excluded = [], dict.fromkeys(COUNTERS, 0), []
    for rank, series in sorted(ranks.items()):
        first, last = series[0], series[-1]
        if scope == "window" and len(series) < 2:
            excluded.append(rank)
            rows.append(
                {"rank": rank, "snapshots": 1, "status": "insufficient_snapshots"}
            )
            continue
        counts = {
            key: last[key] - (first[key] if scope == "window" else 0)
            for key in COUNTERS
        }
        for key, value in counts.items():
            total[key] += value
        rows.append(
            {
                "rank": rank,
                "snapshots": len(series),
                "first_forward": first["forwards"],
                "last_forward": last["forwards"],
                "service_max_decode_seq_len": last["max_decode_seq_len"],
                **counts,
                **rate(counts),
            }
        )
    aggregate = {**total, **rate(total)}
    if not rows or len(excluded) == len(rows):
        aggregate["status"] = "no_snapshot_window" if rows else "no_trace_records"
    return {
        "scope": scope,
        "coverage": "last cumulative snapshot per rank; includes earlier requests/service warmups"
        if scope == "latest"
        else "first-to-last logged snapshot per rank in the selected file range; excludes unsampled leading/trailing intervals",
        "aggregation": "sum(prefix hits) / sum(prefix selected); rank x layer x token counts, not unique request tokens; tail excluded",
        "excluded_ranks": excluded,
        "aggregate": aggregate,
        "per_rank": rows,
    }


def prediction_lengths(path, cache_tokens=None):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = [(row.get("prompt_tokens"), row.get("completion_tokens")) for row in rows]
    pairs = [
        (prompt, output)
        for prompt, output in pairs
        if type(prompt) is int and type(output) is int and prompt >= 0 and output >= 0
    ]
    result = {"questions": len(rows), "questions_with_length_metadata": len(pairs)}
    if pairs:
        result.update(
            max_prompt_tokens=max(prompt for prompt, _ in pairs),
            max_completion_tokens=max(output for _, output in pairs),
            max_prompt_plus_completion_tokens=max(
                prompt + output for prompt, output in pairs
            ),
        )
    if cache_tokens is not None:
        if cache_tokens <= 0 or cache_tokens % 128:
            raise ValueError("cache_tokens must be a positive multiple of 128")
        # Q1's last decode sees L=prompt+completion-1; its prefix is floor((L-1)/128)*128.
        result.update(
            cache_tokens_for_estimate=cache_tokens,
            questions_with_possible_cache_management_decode=sum(
                output >= 2 and (prompt + output - 2) // 128 * 128 >= cache_tokens
                for prompt, output in pairs
            ),
            estimate_scope="derived from completed Q1 request lengths; actual native-path coverage uses trace counters",
        )
    return result


def report(log, scope="latest", offset=0, predictions=None, cache_tokens=None):
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    with log.open("rb") as file:
        file.seek(0, 2)
        end = file.tell()
        if offset > end:
            raise ValueError("Log was truncated/rotated or offset exceeds file size")
        file.seek(offset)
        raw = file.read()
    result = summarize(parse_trace(raw.decode("utf-8", errors="replace")), scope)
    result.update(log=str(log.resolve()), byte_range=[offset, end])
    if predictions:
        result["gsm8k_lengths"] = prediction_lengths(predictions, cache_tokens)
    return result


def print_summary(result):
    print("C8 GSM8K / cache-management hit-rate summary", flush=True)
    print(f"Scope: {result['coverage']}", flush=True)
    for row in [*result["per_rank"], {"rank": "ALL (weighted)", **result["aggregate"]}]:
        value = row.get("prefix_cache_hit_rate_pct")
        display = "N/A" if value is None else f"{value:.4f}%"
        print(
            f"{row['rank']}: hit_rate={display}; hits={row.get('prefix_cache_hits')}; selected={row.get('prefix_selected_tokens')}; misses={row.get('selected_miss_tokens')}; cache_management_rows={row.get('sparse_request_layer_rows')}; copy_sfa_decode_calls={row.get('decode_layer_calls')}; status={row['status']}",
            flush=True,
        )
    if "gsm8k_lengths" in result:
        print("GSM8K lengths: " + json.dumps(result["gsm8k_lengths"]), flush=True)
    if result["excluded_ranks"]:
        print(
            "Excluded ranks (need >=2 snapshots): "
            + ", ".join(result["excluded_ranks"]),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--scope", choices=("latest", "window"), default="latest")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--cache-tokens", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(
        args.log, args.scope, args.offset, args.predictions, args.cache_tokens
    )
    print_summary(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
