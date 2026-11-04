"""Run the byte-frozen 100-cache historical SGLang benchmark client."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parent
_ASSETS = _ROOT / "frozen_client"
_CACHE_TOKENS = 63872
_INPUT_TOKENS = 64000
_OUTPUT_TOKENS = 1024


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_part_asset(name: str, spec: dict) -> bytes:
    if "parts" in spec:
        part_dir = _ASSETS / (name + ".parts")
        chunks = []
        for part in spec["parts"]:
            data = (part_dir / part["path"]).read_bytes()
            if len(data) != part["bytes"] or _sha(data) != part["sha256"]:
                raise ValueError(f"frozen asset part mismatch: {name}/{part['path']}")
            chunks.append(data)
        data = b"".join(chunks)
    else:
        data = (_ASSETS / name).read_bytes()
    if len(data) != spec["bytes"] or _sha(data) != spec["sha256"]:
        raise ValueError(f"frozen asset mismatch: {name}")
    return data


def _verify_assets(runtime: Path) -> tuple[Path, Path, Path]:
    manifest = json.loads((_ASSETS / "asset_manifest.json").read_text(encoding="utf-8"))
    source_manifest = (_ASSETS / "source_manifest.json").read_bytes()
    if _sha(source_manifest) != manifest["source_manifest_sha256"]:
        raise ValueError("source_manifest.json hash mismatch")
    assets = manifest["assets"]
    archive = _read_part_asset("bench_client.tar.gz", assets["bench_client.tar.gz"])
    archive_path = runtime / "bench_client.tar.gz"
    archive_path.write_bytes(archive)
    entry_gz = _read_part_asset(
        "bench_serving_with_acceptance.py.gz",
        assets["bench_serving_with_acceptance.py.gz"],
    )
    wrapper_gz = _read_part_asset("run_bench.py.gz", assets["run_bench.py.gz"])
    fixture_gz = _read_part_asset(
        "fixtures/prefix100_warm.pkl.gz", assets["fixtures/prefix100_warm.pkl.gz"]
    )
    formal_gz = _read_part_asset(
        "fixtures/prefix100_formal16.pkl.gz",
        assets["fixtures/prefix100_formal16.pkl.gz"],
    )
    files = (
        (
            entry_gz,
            "bench_serving_with_acceptance.py.gz",
            "bench_serving_with_acceptance.py",
        ),
        (wrapper_gz, "run_bench.py.gz", "run_bench.py"),
        (fixture_gz, "fixtures/prefix100_warm.pkl.gz", "prefix100_warm.pkl"),
        (formal_gz, "fixtures/prefix100_formal16.pkl.gz", "prefix100_formal16.pkl"),
    )
    out = {}
    for compressed, asset_name, basename in files:
        raw = gzip.decompress(compressed) if asset_name.endswith(".gz") else compressed
        spec = assets[asset_name]
        if len(raw) != spec["source_bytes"] or _sha(raw) != spec["source_sha256"]:
            raise ValueError(f"decompressed source hash mismatch: {asset_name}")
        target = runtime / basename
        target.write_bytes(raw)
        out[basename] = target
    return archive_path, out["bench_serving_with_acceptance.py"], out["run_bench.py"]


def _extract_archive(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe path in frozen client archive: {member.name}")
        try:
            tf.extractall(destination, filter="data")
        except TypeError as exc:
            raise RuntimeError(
                "Python tarfile data filter is required for safe extraction"
            ) from exc
    python_dir = destination / "python"
    if not python_dir.is_dir():
        raise ValueError("frozen client archive has no python/ directory")
    return python_dir


def _normalize_ids(fixture_ids) -> list[int]:
    if isinstance(fixture_ids, (str, Path)):
        path = Path(fixture_ids)
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
        fixture_ids = json.loads(raw)
    ids = [int(item) for item in fixture_ids]
    if len(ids) != _INPUT_TOKENS or any(item < 0 for item in ids):
        raise ValueError("fixture_ids must contain exactly 64000 nonnegative token IDs")
    return ids


def _make_cache_fixture(ids: list[int], bs: int, cache_dir: Path, work: Path) -> dict:
    """Preserve historical fixtures or extend their DatasetRow-compatible fields."""
    helper = _ASSETS / "fixture_helper.py"
    ids_path = work / "fixture_ids.json"
    ids_path.write_text(json.dumps(ids, separators=(",", ":")), encoding="utf-8")
    warm_path = (
        cache_dir / "gen_shared_prefix_42_1_1_64000_0_1_ids_TokenizersBackend.pkl"
    )
    formal_path = (
        cache_dir
        / f"gen_shared_prefix_42_1_{bs}_64000_0_1024_ids_TokenizersBackend.pkl"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(helper),
            str(ids_path),
            str(warm_path),
            str(formal_path),
            str(bs),
            str(work),
        ],
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
    )
    (work / "fixture-helper.stdout.log").write_text(result.stdout, encoding="utf-8")
    (work / "fixture-helper.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"fixture preparation failed ({result.returncode}); see fixture-helper logs"
        )
    metadata = json.loads((work / "fixture-helper.json").read_text(encoding="utf-8"))
    return metadata


def _client_command(
    wrapper: Path,
    entry: Path,
    host: str,
    port: int,
    model: str,
    prompts: int,
    output_len: int,
    output_file: Path,
) -> list[str]:
    return [
        sys.executable,
        str(wrapper),
        str(entry),
        "--backend",
        "sglang",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--dataset-name",
        "generated-shared-prefix",
        "--gsp-num-groups",
        "1",
        "--gsp-system-prompt-len",
        "64000",
        "--gsp-question-len",
        "0",
        "--gsp-prompts-per-group",
        str(prompts),
        "--gsp-output-len",
        str(output_len),
        "--max-concurrency",
        str(prompts),
        "--seed",
        "42",
        "--tokenize-prompt",
        "--random-range-ratio",
        "1.0",
        "--request-rate",
        "inf",
        "--warmup-requests",
        "0",
        "--cache-report",
        "--output-file",
        str(output_file),
        "--output-details",
    ]


def _assert_cache_hit(stdout_text: str, expected_cache: Path) -> None:
    marker = f"Loading cached generated input data from {expected_cache}"
    if marker not in stdout_text or "Caching generated input data to" in stdout_text:
        raise RuntimeError(
            f"client did not load the required fixture cache: {expected_cache}"
        )


def _run_client(
    label_dir: Path,
    round_name: str,
    command: list[str],
    old_python: Path,
    home: Path,
    expected_cache: Path,
) -> dict:
    run_dir = label_dir / round_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            env.pop(key, None)
    env.pop("SGLANG_IS_IN_CI", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(old_python), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["HOME"] = str(home)
    env["TMPDIR"] = env["TMP"] = env["TEMP"] = str(run_dir / "tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    started = time.monotonic()
    with (
        (run_dir / "stdout.log").open("wb") as stdout,
        (run_dir / "stderr.log").open("wb") as stderr,
    ):
        proc = subprocess.run(
            command, cwd=run_dir, env=env, stdout=stdout, stderr=stderr, check=False
        )
    elapsed = time.monotonic() - started
    (run_dir / "exitcode.txt").write_text(f"{proc.returncode}\n", encoding="ascii")
    stdout_text = (run_dir / "stdout.log").read_text(encoding="utf-8", errors="replace")
    result_file = next(iter(sorted(run_dir.glob("*.jsonl"))), None)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{round_name} client exit {proc.returncode}; artifacts: {run_dir}"
        )
    if result_file is None:
        raise RuntimeError(
            f"{round_name} produced no result JSONL; artifacts: {run_dir}"
        )
    try:
        _assert_cache_hit(stdout_text, expected_cache)
    except RuntimeError as exc:
        raise RuntimeError(f"{round_name}: {exc}; artifacts: {run_dir}") from exc
    rows = [
        json.loads(line)
        for line in result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one result row, found {len(rows)} in {result_file}")
    return {
        "row": rows[0],
        "wall_seconds": elapsed,
        "result_jsonl": str(result_file),
        "directory": str(run_dir),
    }


def _validate_row(
    row: dict,
    requests: int,
    expected_cache: int | None,
    expected_output: int = _OUTPUT_TOKENS,
) -> dict:
    completed = int(row.get("completed", -1))
    failed = int(row.get("failed", requests - completed))
    total_input = int(row.get("total_input_tokens", -1))
    total_output = int(row.get("total_output_tokens", -1))
    duration = float(row.get("duration", 0))
    input_lens = row.get("input_lens")
    output_lens = row.get("output_lens")
    errors = row.get("errors")
    cached = row.get("cached_tokens")
    problems = []
    if completed != requests or failed != 0:
        problems.append(f"completion {completed}/{requests}, failed={failed}")
    if (
        total_input != requests * _INPUT_TOKENS
        or total_output != requests * expected_output
    ):
        problems.append(f"token totals input={total_input} output={total_output}")
    if not isinstance(input_lens, list) or input_lens != [_INPUT_TOKENS] * requests:
        problems.append("input_lens mismatch")
    if not isinstance(output_lens, list) or output_lens != [expected_output] * requests:
        problems.append("output_lens mismatch")
    if not isinstance(errors, list) or len(errors) != requests or any(errors):
        problems.append("errors missing or nonempty")
    if expected_cache is not None and (
        not isinstance(cached, list) or cached != [expected_cache] * requests
    ):
        problems.append(f"cached_tokens mismatch: {cached}")
    if not math.isfinite(duration) or duration <= 0:
        problems.append(f"invalid benchmark duration {duration}")
    throughput = total_output / duration if duration > 0 else 0.0
    reported = float(row.get("output_throughput", throughput))
    if not math.isclose(reported, throughput, rel_tol=0.015, abs_tol=0.1):
        problems.append(
            f"throughput mismatch reported={reported} recomputed={throughput}"
        )
    if problems:
        raise ValueError("benchmark report validation failed: " + "; ".join(problems))
    return {
        "requests": requests,
        "successful_requests": completed,
        "failed_requests": failed,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "duration_seconds": duration,
        "output_tokens_per_second": throughput,
        "mean_latency_seconds": float(row.get("mean_e2e_latency_ms", 0)) / 1000,
        "mean_ttft_seconds": float(row.get("mean_ttft_ms", 0)) / 1000,
        "mean_tpot_seconds": float(row.get("mean_tpot_ms", 0)) / 1000,
        "mean_spec_accept_length": row.get("accept_length"),
        "cached_tokens": cached,
    }


def run_round(router_url, model, output_dir, fixture_ids, bs, label):
    """Run one seed request and one measured batch with the frozen historical client."""
    if not isinstance(bs, int) or bs < 1:
        raise ValueError("bs must be a positive integer")
    ids = _normalize_ids(fixture_ids)
    parsed = urlsplit(router_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or not parsed.port
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "router_url must be an explicit http(s) origin with host and port"
        )
    label_dir = Path(output_dir) / str(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    runtime = label_dir / "runtime"
    runtime.mkdir(exist_ok=True)
    archive, entry, wrapper = _verify_assets(runtime)
    old_python = _extract_archive(archive, runtime / "old_client")
    home = label_dir / "client_home"
    cache_dir = home / ".cache" / "sglang" / "benchmark"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fixture_meta = _make_cache_fixture(ids, bs, cache_dir, label_dir)
    fixture_sha = _sha(json.dumps(ids, separators=(",", ":")).encode())
    (label_dir / "fixture-provenance.json").write_text(
        json.dumps(
            {"ids_sha256": fixture_sha, "token_count": len(ids), **fixture_meta},
            indent=2,
        ),
        encoding="utf-8",
    )
    warm_output = label_dir / "seed" / "result.jsonl"
    measure_output = label_dir / "measure" / "result.jsonl"
    warm_cache = (
        cache_dir / "gen_shared_prefix_42_1_1_64000_0_1_ids_TokenizersBackend.pkl"
    )
    formal_cache = (
        cache_dir
        / f"gen_shared_prefix_42_1_{bs}_64000_0_1024_ids_TokenizersBackend.pkl"
    )
    seed = _run_client(
        label_dir,
        "seed",
        _client_command(
            wrapper, entry, parsed.hostname, parsed.port, model, 1, 1, warm_output
        ),
        old_python,
        home,
        warm_cache,
    )
    _validate_row(seed["row"], 1, None, expected_output=1)
    measure = _run_client(
        label_dir,
        "measure",
        _client_command(
            wrapper,
            entry,
            parsed.hostname,
            parsed.port,
            model,
            bs,
            _OUTPUT_TOKENS,
            measure_output,
        ),
        old_python,
        home,
        formal_cache,
    )
    report = _validate_row(measure["row"], bs, _CACHE_TOKENS)
    report.update(
        {
            "round": label,
            "fixture_ids_sha256": fixture_sha,
            "fixture_count": len(ids),
            "fixture_expansion": fixture_meta,
            "seed": {"requests": 1, "directory": seed["directory"]},
            "formal_directory": measure["directory"],
            "frozen_finish_reason": False,
        }
    )
    (label_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
