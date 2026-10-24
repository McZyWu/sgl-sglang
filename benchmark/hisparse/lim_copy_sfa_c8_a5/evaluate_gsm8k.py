"""Full 1319-question GSM8K using the repository's few_shot_gsm8k.run_eval."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import requests

from sglang.test.few_shot_gsm8k import INVALID, get_answer_value, run_eval

FULL_TEST_SIZE = 1319


def read_data(path):
    path = path.expanduser().resolve()
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows):
        if not isinstance(row.get("question"), str) or not isinstance(
            row.get("answer"), str
        ):
            raise ValueError(f"{path}:{index + 1}: expected question/answer strings")
        if "####" not in row["answer"] or get_answer_value(row["answer"]) == INVALID:
            raise ValueError(f"{path}:{index + 1}: invalid GSM8K gold answer")
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(args):
    if args.parallel <= 0 or args.max_new_tokens <= 0 or args.num_shots < 0:
        raise ValueError(
            "parallel/max_new_tokens must be positive; num_shots must be nonnegative"
        )
    endpoint = urlsplit(args.url)
    if (
        endpoint.scheme != "http"
        or not endpoint.hostname
        or endpoint.path not in ("", "/")
    ):
        raise ValueError("--url must be the server root, e.g. http://127.0.0.1:8810")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("--url must not contain credentials, query or fragment")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "gsm8k.result.json"
    if result_path.exists() or (output / "tmp_output_gsm8k.txt").exists():
        raise ValueError(f"Choose a fresh output directory: {output}")
    record = {
        "status": "running",
        "evaluator": "sglang.test.few_shot_gsm8k.run_eval",
        "protocol": "few-shot raw completion; repository last-number scorer",
        "url": args.url,
        "num_questions": FULL_TEST_SIZE,
        "num_shots": args.num_shots,
        "parallel": args.parallel,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "result_directory": str(output),
    }

    def save():
        result_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    save()
    previous_directory = Path.cwd()
    try:
        data = args.data_path.expanduser().resolve()
        rows, digest = read_data(data)
        if (
            len(rows) != FULL_TEST_SIZE
            or len({row["question"] for row in rows}) != FULL_TEST_SIZE
        ):
            raise ValueError(
                f"Full test requires {FULL_TEST_SIZE} unique questions; found {len(rows)} rows"
            )
        shot_data = args.train_path.expanduser().resolve() if args.train_path else None
        shots, shot_digest = read_data(shot_data) if shot_data else (rows, digest)
        if args.num_shots > len(shots):
            raise ValueError("Not enough few-shot examples")
        if shot_data and args.num_shots:
            questions = {row["question"] for row in rows}
            if any(row["question"] in questions for row in shots[: args.num_shots]):
                raise ValueError(
                    "Training few-shot examples overlap the test questions"
                )
        record.update(
            data_path=str(data),
            data_sha256=digest,
            few_shot_data_path=str(shot_data or data),
            few_shot_data_sha256=shot_digest,
            few_shot_source="train" if shot_data else "test_prefix",
            test_examples_reused_as_shots=args.num_shots if not shot_data else 0,
        )
        if args.num_shots and not shot_data:
            print(
                f"Protocol: test's first {args.num_shots} examples are also few-shot demonstrations, matching run_eval's default.",
                flush=True,
            )
        response = requests.get(args.url.rstrip("/") + "/model_info", timeout=30)
        response.raise_for_status()
        model_info = response.json()
        record["model_info"] = {
            key: model_info.get(key)
            for key in (
                "model_path",
                "served_model_name",
                "is_generation",
                "weight_version",
            )
        }
        save()
        print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
        # run_eval writes its full state dump to the working directory.
        os.chdir(output)
        metrics = run_eval(
            SimpleNamespace(
                num_shots=args.num_shots,
                data_path=str(data),
                few_shot_data_path=str(shot_data) if shot_data else None,
                output_jsonl=str(output / "predictions.jsonl"),
                num_questions=FULL_TEST_SIZE,
                max_new_tokens=args.max_new_tokens,
                parallel=args.parallel,
                host=endpoint.hostname,
                port=endpoint.port or 80,
                temperature=0.0,
            )
        )
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError(f"Non-finite evaluation metrics: {metrics}")
        predictions = [
            json.loads(line)
            for line in (output / "predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if [row["index"] for row in predictions] != list(range(FULL_TEST_SIZE)):
            raise ValueError("Incomplete or duplicated GSM8K predictions")
        finishes = [row["finish_reason"] for row in predictions]
        types = [
            value.get("type") if isinstance(value, dict) else value
            for value in finishes
        ]
        errors = sum(value not in ("stop", "length") for value in types)
        record.update(
            completed_questions=len(predictions),
            truncated=sum(value == "length" for value in types),
            request_errors=errors,
        )
        if errors:
            raise ValueError(f"{errors} requests did not end normally")
        record.update(
            status="pass", metrics={key: float(value) for key, value in metrics.items()}
        )
        print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
    except BaseException as error:
        record.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        os.chdir(previous_directory)
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8810")
    parser.add_argument("--data-path", type=Path, default=Path("/data/wzy/test.jsonl"))
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())
