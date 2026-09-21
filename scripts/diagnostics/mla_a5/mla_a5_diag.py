#!/usr/bin/env python3
"""Temporary MLA diagnostics for SGLang PR 40131 (1344e1e2a5 / 3ba07ddc43).

Install/restore use only the Python standard library. Analysis uses CPU torch.
Only the explicit run-case command launches the existing SGLang test/server.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


BACKEND = "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
PREPARE = "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
MODULE = "python/sglang/srt/hardware_backend/npu/attention/_mla_a5_diag.py"
BACKUP = ".mla-a5-diag-backup"
_counts = {}
_banner_done = False


def _mode():
    value = os.getenv("SGLANG_MLA_DIAG", "off").lower()
    if value not in ("off", "observe", "replace_merge"):
        raise ValueError("SGLANG_MLA_DIAG must be off, observe, or replace_merge")
    return value


def _torch():
    import torch
    return torch


def _rank():
    torch = _torch()
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _root():
    result = Path(os.getenv("SGLANG_MLA_DIAG_DIR", "/tmp/mla-a5-diag"))
    result.mkdir(parents=True, exist_ok=True)
    return result


def _emit(event, **fields):
    record = {"event": event, "rank": _rank(), "pid": os.getpid(), **fields}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with (_root() / f"rank{_rank()}-pid{os.getpid()}.jsonl").open("a") as stream:
        stream.write(line + "\n")
    print("[MLA_DIAG] " + line, flush=True)


def _error(where):
    message = traceback.format_exc()
    try:
        _emit("diagnostic_error", where=where, traceback=message)
    except Exception:
        print(f"[MLA_DIAG] diagnostic_error {where}: {message}", flush=True)


def _banner():
    global _banner_done
    if _banner_done:
        return
    _banner_done = True
    import torch_npu
    torch = _torch()
    _emit(
        "config", mode=_mode(), torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__), module=__file__,
        env={key: os.getenv(key) for key in (
            "ASCEND_USE_FIA", "SGLANG_USE_FIA_NZ", "SGLANG_NPU_USE_MLAPO",
            "SGLANG_MLA_DIAG", "SGLANG_MLA_DIAG_LAYERS",
            "SGLANG_MLA_DIAG_STEPS", "SGLANG_MLA_DIAG_DIR",
        )},
    )


def _take(kind, layer):
    if _mode() == "off":
        return False
    _banner()
    layers = os.getenv("SGLANG_MLA_DIAG_LAYERS", "0,13,26")
    if layers != "all" and int(layer) not in {int(x) for x in layers.split(",")}:
        return False
    key = (kind, int(layer))
    step = _counts.get(key, 0)
    if step >= int(os.getenv("SGLANG_MLA_DIAG_STEPS", "2")):
        return False
    _counts[key] = step + 1
    return True


def _positions(length, count):
    if length <= 0:
        return []
    count = min(length, max(1, count))
    if count == 1:
        return [length - 1]
    return sorted({i * (length - 1) // (count - 1) for i in range(count)})


def _cpu(tensor):
    return tensor.detach().to(device="cpu").contiguous().clone()


def _describe(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device)}


def _pick_heads(tensor, heads):
    torch = _torch()
    idx = torch.tensor(heads, dtype=torch.long, device=tensor.device)
    return tensor.detach().index_select(1, idx)


def _pick_rows_heads(tensor, rows, heads):
    torch = _torch()
    idx = torch.tensor(rows, dtype=torch.long, device=tensor.device)
    return _cpu(_pick_heads(tensor.detach().index_select(0, idx), heads))


def _stats(actual, expected):
    torch = _torch()
    original_a, original_b = actual.detach().cpu(), expected.detach().cpu()
    a, b = original_a.double(), original_b.double()
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    finite = torch.isfinite(a) & torch.isfinite(b)
    matching_inf = torch.isinf(a) & (a == b)
    result = {
        "numel": a.numel(), "exact": bool(torch.equal(original_a, original_b)),
        "unequal_count": int((original_a != original_b).sum()),
        "nonfinite_mismatch": int((~finite & ~matching_inf).sum()),
        "actual_nan": int(torch.isnan(a).sum()),
        "expected_nan": int(torch.isnan(b).sum()),
    }
    if finite.any():
        delta = a[finite] - b[finite]
        ref = b[finite]
        result.update(
            max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()),
            rms_rel=float(delta.square().mean().sqrt() /
                          ref.square().mean().sqrt().clamp_min(1e-12)),
            mean_signed=float(delta.mean()),
        )
    return result


def _save(kind, layer, packet):
    path = _root() / f"{kind}-L{layer}-r{_rank()}-p{os.getpid()}-{time.time_ns()}.pt"
    _torch().save(packet, path)
    _emit("capture", kind=kind, layer=int(layer), file=str(path))


def record_cache(layer, kv_a, k_pe, ckv_cache, rope_cache, locations, valid_tokens):
    """PA_BNSD only: compare returned KV with slots written by the SAME call."""
    try:
        if not _take("cache", layer):
            return
        if os.getenv("SGLANG_USE_FIA_NZ", "0").lower() in ("1", "true"):
            _emit("cache_skip", reason="NZ is outside this diagnostic", layer=int(layer))
            return
        torch = _torch()
        kv = kv_a.detach().reshape(-1, kv_a.shape[-1])
        rope = k_pe.detach().reshape(-1, k_pe.shape[-1])
        count = min(int(valid_tokens), locations.numel(), kv.shape[0], rope.shape[0])
        loc_cpu = locations[:count].detach().cpu().long().tolist()
        last_writer = {slot: i for i, slot in enumerate(loc_cpu) if slot >= 0}
        rows = sorted(last_writer.values())
        rows = [rows[i] for i in _positions(len(rows), 16)]
        if not rows:
            _emit("cache_skip", reason="no valid write slots", layer=int(layer))
            return
        row_idx = torch.tensor(rows, device=kv.device, dtype=torch.long)
        slots = locations.detach()[row_idx].long()
        actual_kv = _cpu(ckv_cache.detach().reshape(-1, kv.shape[-1]).index_select(0, slots))
        actual_rope = _cpu(rope_cache.detach().reshape(-1, rope.shape[-1]).index_select(0, slots))
        expected_kv = _cpu(kv.index_select(0, row_idx)).to(actual_kv.dtype)
        expected_rope = _cpu(rope.index_select(0, row_idx)).to(actual_rope.dtype)
        packet = {"kind": "cache", "layer": int(layer), "rank": _rank(),
                  "rows": rows, "slots": _cpu(slots),
                  "actual_kv": actual_kv, "expected_kv": expected_kv,
                  "actual_rope": actual_rope, "expected_rope": expected_rope}
        _emit("cache_roundtrip", layer=int(layer),
              kv=_stats(actual_kv, expected_kv), rope=_stats(actual_rope, expected_rope))
        _save("cache", layer, packet)
    except Exception:
        _error("record_cache")


def begin(layer, backend, batch, q, qr, k, kr, v, cu_query):
    try:
        if not _take("attention", layer.layer_id):
            return None
        torch = _torch()
        prefix = backend.forward_metadata.prefix_lens
        prefix = None if prefix is None else prefix.tolist()
        blocks = backend.forward_metadata.flatten_prefix_block_tables
        blocks_cpu = None if blocks is None else _cpu(blocks).long()
        meta = {
            "cu_query": list(cu_query), "prefix_lens": prefix,
            "block_ids": None if blocks_cpu is None else blocks_cpu.tolist(),
            "page_size": int(backend.page_size), "q": _describe(q),
            "k": _describe(k), "v": _describe(v), "qr": _describe(qr),
            "kr": _describe(kr), "global_valid_tokens": int(batch.global_num_token_non_padded_cpu),
        }
        _emit("prefix_enter", layer=int(layer.layer_id), metadata=meta)
        if prefix is None or blocks_cpu is None:
            _emit("missing_prefix_metadata", layer=int(layer.layer_id))
            return None
        if len(prefix) != len(cu_query) or (cu_query and cu_query[-1] != q.shape[0]):
            raise ValueError("TND query/prefix metadata does not match input")
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            raise ValueError("Diagnostic expects equal expanded MLA Q/K/V head counts")
        heads = _positions(q.shape[1], int(os.getenv("SGLANG_MLA_DIAG_HEADS", "2")))
        req_limit = int(os.getenv("SGLANG_MLA_DIAG_REQUESTS", "2"))
        candidates = [i for i, length in enumerate(prefix) if length > 0]
        empty = [i for i, length in enumerate(prefix) if length == 0]
        if empty and candidates:
            candidates = candidates[:1] + empty[:1] + candidates[1:] + empty[1:]
        else:
            candidates += empty
        packet = {"kind": "attention", "layer": int(layer.layer_id), "rank": _rank(),
                  "scale": float(layer.scaling), "metadata": meta, "samples": []}
        q_start, p_start = 0, 0
        max_keys = int(os.getenv("SGLANG_MLA_DIAG_MAX_KEYS", "16384"))
        selected = set(candidates[:req_limit])
        page_tokens = (blocks_cpu[:, None] * backend.page_size +
                       torch.arange(backend.page_size)[None, :]).flatten()
        for request, (q_end, prefix_len) in enumerate(zip(cu_query, prefix)):
            q_len = q_end - q_start
            if request in selected and q_len > 0 and q_len + prefix_len <= max_keys:
                rows = _positions(q_len, int(os.getenv("SGLANG_MLA_DIAG_ROWS", "8")))
                absolute = [q_start + row for row in rows]
                req_idx = int(batch.req_pool_indices[request].item())
                logical = _cpu(backend.req_to_token_pool.req_to_token[req_idx, :prefix_len]).long()
                sample = {
                    "request": request, "q_start": q_start, "q_end": q_end,
                    "p_start": p_start, "prefix_len": prefix_len,
                    "rows": rows, "absolute_rows": absolute, "heads": heads,
                    "q": _pick_rows_heads(q, absolute, heads),
                    "qr": _pick_rows_heads(qr, absolute, heads),
                    "current_k": _cpu(_pick_heads(k[q_start:q_end], heads)),
                    "current_kr": _cpu(_pick_heads(kr[q_start:q_end], heads)),
                    "current_v": _cpu(_pick_heads(v[q_start:q_end], heads)),
                    "logical_token_slots": logical,
                    "gathered_token_slots": page_tokens[p_start:p_start + prefix_len].clone(),
                }
                packet["samples"].append(sample)
            q_start, p_start = q_end, p_start + prefix_len
        if not packet["samples"]:
            _emit("attention_skip", layer=int(layer.layer_id), reason="no request within capture limits")
            return None
        return packet
    except Exception:
        _error("begin")
        return None


def current(packet, output, lse):
    if packet is None:
        return
    try:
        packet["current_output_meta"] = _describe(output)
        packet["current_lse_meta"] = _describe(lse)
        if tuple(lse.shape) != tuple(output.shape[:2]) + (1,):
            raise ValueError(f"Expected TND LSE [T,N,1], got {tuple(lse.shape)}")
        for sample in packet["samples"]:
            args = sample["absolute_rows"], sample["heads"]
            sample["current_output"] = _pick_rows_heads(output, *args)
            sample["current_lse"] = _pick_rows_heads(lse, *args).squeeze(-1)
    except Exception:
        packet["capture_error"] = traceback.format_exc()
        _error("current")


def prefix(packet, backend, layer, latent, rope_cache, k, kr, v, output, lse):
    if packet is None:
        return
    try:
        packet["prefix_output_meta"] = _describe(output)
        packet["prefix_lse_meta"] = _describe(lse)
        packet["projection"] = {
            "class": type(layer.kv_b_proj).__name__,
            "weight": _describe(layer.kv_b_proj.weight) if hasattr(layer.kv_b_proj, "weight") else None,
        }
        if tuple(lse.shape) != tuple(output.shape[:2]) + (1,):
            raise ValueError(f"Expected TND LSE [T,N,1], got {tuple(lse.shape)}")
        for sample in packet["samples"]:
            start, length, heads = sample["p_start"], sample["prefix_len"], sample["heads"]
            for name, value in (("prefix_k", k), ("prefix_kr", kr), ("prefix_v", v)):
                sample[name] = _cpu(_pick_heads(value[start:start + length], heads))
            args = sample["absolute_rows"], heads
            sample["prefix_output"] = _pick_rows_heads(output, *args)
            sample["prefix_lse"] = _pick_rows_heads(lse, *args).squeeze(-1)
            token_rows = _positions(length, 16)
            if token_rows:
                torch = _torch()
                local_idx = torch.tensor(token_rows, device=latent.device, dtype=torch.long)
                slots = sample["logical_token_slots"][token_rows].to(device=latent.device)
                key_buffer = backend.token_to_kv_pool.get_key_buffer(layer.layer_id)
                rope_buffer = backend.token_to_kv_pool.get_value_buffer(layer.layer_id)
                actual_kv = _cpu(latent.detach().reshape(-1, latent.shape[-1]).index_select(0, start + local_idx))
                actual_rope = _cpu(rope_cache.detach().reshape(-1, rope_cache.shape[-1]).index_select(0, start + local_idx))
                expected_kv = _cpu(key_buffer.detach().reshape(-1, latent.shape[-1]).index_select(0, slots))
                expected_rope = _cpu(rope_buffer.detach().reshape(-1, rope_cache.shape[-1]).index_select(0, slots))
                sample["gather_kv"] = _stats(actual_kv, expected_kv)
                sample["gather_rope"] = _stats(actual_rope, expected_rope)
            sample["slot_mapping"] = _stats(sample["gathered_token_slots"], sample["logical_token_slots"])
    except Exception:
        packet["capture_error"] = traceback.format_exc()
        _error("prefix")


def _explicit_merge(outputs, lses):
    torch = _torch()
    a, b = outputs[0].float(), outputs[1].float()
    la, lb = lses[0].float(), lses[1].float()
    total = torch.logaddexp(la, lb)
    return a * (la - total).exp().unsqueeze(-1) + b * (lb - total).exp().unsqueeze(-1)


def merge(packet, lses, outputs, update_type):
    import torch_npu
    result = torch_npu.npu_attention_update(lses, outputs, update_type)
    mode = _mode()
    replacement = None
    if mode == "replace_merge":
        # Apply to ALL prefix batches/layers/ranks, irrespective of capture limits.
        replacement = _explicit_merge(outputs, lses)
    if packet is not None:
        try:
            shape = packet["metadata"]["q"]["shape"]
            merged = result[0].reshape(shape[0], shape[1], -1)
            for sample in packet["samples"]:
                args = sample["absolute_rows"], sample["heads"]
                sample["merged_output"] = _pick_rows_heads(merged, *args)
                if replacement is not None:
                    sample["replacement_output"] = _pick_rows_heads(replacement.reshape_as(merged), *args)
            packet["mode"] = mode
            _save("attention", packet["layer"], packet)
        except Exception:
            _error("merge_capture")
    if replacement is not None:
        return replacement, result[1]
    return result


def _reference(q, qr, k, kr, v, scale, rows=None, prefix_len=0):
    torch = _torch()
    if k.shape[0] == 0:
        return (torch.zeros((*q.shape[:2], v.shape[-1]), dtype=torch.float32),
                torch.full(q.shape[:2], -float("inf"), dtype=torch.float32))
    logits = (torch.einsum("qhd,khd->hqk", q.float(), k.float()) +
              torch.einsum("qhd,khd->hqk", qr.float(), kr.float())) * scale
    if rows is not None:
        mask = torch.arange(k.shape[0])[None, :] > (torch.tensor(rows)[:, None] + prefix_len)
        logits.masked_fill_(mask.unsqueeze(0), -float("inf"))
    lse = torch.logsumexp(logits, dim=-1).transpose(0, 1).contiguous()
    output = torch.einsum("hqk,khd->qhd", logits.softmax(dim=-1), v.float())
    return output, lse


def analyze(directory):
    torch = _torch()
    torch.set_num_threads(max(1, int(os.getenv("MLA_DIAG_CPU_THREADS", "4"))))
    directory = Path(directory)
    reports = []
    for path in sorted(directory.glob("*.pt")):
        try:
            packet = torch.load(path, map_location="cpu", weights_only=True)
            result = {"file": path.name, "kind": packet["kind"],
                      "layer": packet["layer"], "rank": packet["rank"]}
            if packet["kind"] == "cache":
                result.update(kv=_stats(packet["actual_kv"], packet["expected_kv"]),
                              rope=_stats(packet["actual_rope"], packet["expected_rope"]))
            else:
                result["metadata"] = packet["metadata"]
                result["samples"] = []
                if packet.get("capture_error"):
                    result["capture_error"] = packet["capture_error"]
                for sample in packet["samples"]:
                    q, qr, scale = sample["q"], sample["qr"], packet["scale"]
                    co, cl = _reference(q, qr, sample["current_k"], sample["current_kr"],
                                        sample["current_v"], scale, sample["rows"])
                    po, pl = _reference(q, qr, sample["prefix_k"], sample["prefix_kr"], sample["prefix_v"], scale)
                    full, _ = _reference(
                        q, qr, torch.cat((sample["prefix_k"], sample["current_k"])),
                        torch.cat((sample["prefix_kr"], sample["current_kr"])),
                        torch.cat((sample["prefix_v"], sample["current_v"])),
                        scale, sample["rows"], sample["prefix_len"],
                    )
                    explicit = _explicit_merge(
                        (sample["current_output"], sample["prefix_output"]),
                        (sample["current_lse"], sample["prefix_lse"]),
                    )
                    ref_merge = _explicit_merge((co, po), (cl, pl))
                    item = {"request": sample["request"], "rows": sample["rows"], "heads": sample["heads"],
                            "prefix_len": sample["prefix_len"],
                            "slot_mapping": sample["slot_mapping"],
                            "gather_kv": sample.get("gather_kv"), "gather_rope": sample.get("gather_rope")}
                    comparisons = {
                        "current_output_vs_fp32": (sample["current_output"], co),
                        "current_output_vs_rounded_ref": (sample["current_output"], co.to(sample["current_output"].dtype)),
                        "current_lse_vs_fp32": (sample["current_lse"], cl),
                        "prefix_output_vs_fp32": (sample["prefix_output"], po),
                        "prefix_output_vs_rounded_ref": (sample["prefix_output"], po.to(sample["prefix_output"].dtype)),
                        "prefix_lse_vs_fp32": (sample["prefix_lse"], pl),
                        "merge_vs_explicit_same_fia_parts": (sample["merged_output"], explicit),
                        "explicit_same_fia_parts_vs_fp32_full": (explicit, full),
                        "merged_vs_fp32_full": (sample["merged_output"], full),
                        "reference_split_vs_full": (ref_merge, full),
                        "prefix_weight_vs_fp32": (
                            (sample["prefix_lse"].float() - torch.logaddexp(sample["current_lse"].float(), sample["prefix_lse"].float())).exp(),
                            (pl - torch.logaddexp(cl, pl)).exp(),
                        ),
                    }
                    for name, (actual, expected) in comparisons.items():
                        item[name] = _stats(actual, expected)
                    if "replacement_output" in sample:
                        item["replacement_vs_cpu_explicit"] = _stats(sample["replacement_output"], explicit)
                    result["samples"].append(item)
            reports.append(result)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        except Exception:
            reports.append({"file": path.name, "analysis_error": traceback.format_exc()})
            print(f"Analysis failed: {path.name}\n{traceback.format_exc()}", file=sys.stderr)
    output = directory / "analysis.json"
    output.write_text(json.dumps(reports, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Analyzed {len(reports)} capture files -> {output}", file=sys.stderr)
    if not reports:
        raise SystemExit("No captures: check [MLA_DIAG] config/prefix_enter/diagnostic_error logs and imported source path.")
    if any("analysis_error" in report or "capture_error" in report for report in reports):
        raise SystemExit(2)


def _replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one source anchor, found {count}: {old[:120]!r}. No files changed.")
    return text.replace(old, new, 1)


def _patched_sources(repo):
    backend = (repo / BACKEND).read_text(encoding="utf-8")
    prepare = (repo / PREPARE).read_text(encoding="utf-8")
    imp = "from sglang.srt.hardware_backend.npu.attention import _mla_a5_diag as _mla_diag\n"
    backend = _replace_once(backend, "import torch_npu\n", "import torch_npu\n" + imp)
    prepare = _replace_once(prepare, "import torch_npu\n", "import torch_npu\n" + imp)
    anchor = "        k_pe = k_pe.reshape(B, -1, m.qk_rope_head_dim)\n    else:"
    prepare = _replace_once(prepare, anchor,
        "        k_pe = k_pe.reshape(B, -1, m.qk_rope_head_dim)\n"
        "        _mla_diag.record_cache(\n"
        "            m.layer_id, kv_a, k_pe, ckv_cache, k_rope_cache,\n"
        "            forward_batch.out_cache_loc, forward_batch.global_num_token_non_padded_cpu,\n"
        "        )\n    else:")
    anchor = "                    q_nope, q_rope = q_nope.contiguous(), q_rope.contiguous()\n"
    backend = _replace_once(backend, anchor, anchor +
        "                    _mla_ctx = _mla_diag.begin(\n"
        "                        layer, self, forward_batch, q_nope, q_rope,\n"
        "                        k_nope, k_rope, v, cu_query_lens,\n"
        "                    )\n")
    anchor = ("                            return_softmax_lse=True,\n"
              "                        )\n                    )\n                else:\n"
              "                    num_tokens = q_nope.size(0)\n")
    backend = _replace_once(backend, anchor,
        "                            return_softmax_lse=True,\n"
        "                        )\n                    )\n"
        "                    _mla_diag.current(_mla_ctx, attn_output, attn_lse)\n"
        "                else:\n                    num_tokens = q_nope.size(0)\n")
    anchor = ("                    attn_output, _ = torch_npu.npu_attention_update(\n"
              "                        (attn_lse.reshape(-1), prefix_lse.reshape(-1)),\n")
    backend = _replace_once(backend, anchor,
        "                    _mla_diag.prefix(\n"
        "                        _mla_ctx, self, layer, kv_cached, k_rope_cached,\n"
        "                        k_nope, k_rope, v, prefix_output, prefix_lse,\n"
        "                    )\n"
        "                    attn_output, _ = _mla_diag.merge(\n"
        "                        _mla_ctx,\n"
        "                        (attn_lse.reshape(-1), prefix_lse.reshape(-1)),\n")
    compile(backend, BACKEND, "exec")
    compile(prepare, PREPARE, "exec")
    return {BACKEND: backend.encode("utf-8"), PREPARE: prepare.encode("utf-8")}


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def install(repo):
    repo = Path(repo).resolve()
    backup = repo / BACKUP
    if backup.exists() or (repo / MODULE).exists():
        raise SystemExit("Diagnostic/backup already exists. Restore before installing again.")
    changes = _patched_sources(repo)
    changes[MODULE] = Path(__file__).read_bytes()
    compile(changes[MODULE], MODULE, "exec")
    original = {name: (repo / name).read_bytes() for name in (BACKEND, PREPARE)}
    manifest = {name: {"original_sha256": _sha(original[name]) if name in original else None,
                       "patched_sha256": _sha(content)} for name, content in changes.items()}
    backup.mkdir()
    for name, content in original.items():
        destination = backup / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        for name, content in changes.items():
            (repo / name).write_bytes(content)
    except Exception:
        for name, content in original.items():
            (repo / name).write_bytes(content)
        (repo / MODULE).unlink(missing_ok=True)
        shutil.rmtree(backup)
        raise
    print(f"Installed diagnostics in {repo}\nBackup: {backup}\nMode defaults to off.")


def restore(repo):
    repo = Path(repo).resolve()
    backup = repo / BACKUP
    if backup.is_symlink() or backup.resolve().parent != repo:
        raise SystemExit("Backup path is outside the repository; refusing restore.")
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != {BACKEND, PREPARE, MODULE}:
        raise SystemExit("Unexpected backup manifest; refusing restore.")
    for name, entry in manifest.items():
        if _sha((repo / name).read_bytes()) != entry["patched_sha256"]:
            raise SystemExit(f"{name} changed after installation. Refusing to overwrite your edits.")
        if entry["original_sha256"] is not None:
            if _sha((backup / name).read_bytes()) != entry["original_sha256"]:
                raise SystemExit(f"Backup checksum mismatch for {name}")
    for name, entry in manifest.items():
        if entry["original_sha256"] is None:
            (repo / name).unlink()
        else:
            (repo / name).write_bytes((backup / name).read_bytes())
    shutil.rmtree(backup)
    print(f"Restored original bytes in {repo}")


def run_case(repo, model, seed):
    import runpy
    import unittest
    repo = Path(repo).resolve()
    source = str(repo / "python")
    sys.path.insert(0, source)
    inherited = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = source + (os.pathsep + inherited if inherited else "")
    os.environ.pop("GITHUB_EVENT_NAME", None)
    os.chdir(repo)
    namespace = runpy.run_path(str(repo / "test/registered/npu/basic_function/HiCache/test_npu_hicache_mla.py"))
    if model is not None:
        matrix = namespace["TEST_MODEL_MATRIX"]
        limits = dict(next(iter(matrix.values())))
        matrix.clear()
        matrix[model] = limits
    case = namespace["TestAscendMlaHicache"]
    original_setup = case.setUpClass

    @classmethod
    def setup(cls):
        original_setup()
        cls.common_args = list(cls.common_args)
        if "--random-seed" in cls.common_args:
            cls.common_args[cls.common_args.index("--random-seed") + 1] = str(seed)
        else:
            cls.common_args.extend(["--random-seed", str(seed)])
        print(f"Original test server arguments: {cls.common_args}", flush=True)

    case.setUpClass = setup
    print(f"Running original test_npu_hicache_mla: model={list(namespace['TEST_MODEL_MATRIX'])}, seed={seed}", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for command in ("install", "restore"):
        sub.add_parser(command).add_argument("--repo", required=True)
    sub.add_parser("analyze").add_argument("directory")
    run = sub.add_parser("run-case")
    run.add_argument("--repo", required=True)
    run.add_argument("--model", help="Override only the existing test's model path at runtime")
    run.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.action == "install":
        install(args.repo)
    elif args.action == "restore":
        restore(args.repo)
    elif args.action == "run-case":
        run_case(args.repo, args.model, args.seed)
    else:
        analyze(args.directory)


if __name__ == "__main__":
    main()
