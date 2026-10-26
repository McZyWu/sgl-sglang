"""Opt-in C8 transfer counters. Tensor updates also execute on graph replay."""

import json
import logging

import torch

from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

logger = logging.getLogger(__name__)


class C8OffloadTrace:
    # Values count layer-token rows, not unique requests or PCIe transactions.
    KEYS = (
        "prefill_d2h_tokens",
        "prefill_h2d_tokens",
        "decode_d2h_tokens",
        "decode_layer_calls",
        "decode_request_layer_rows",
        "sparse_request_layer_rows",
        "miss_request_layer_rows",
        "cache_refill_tokens",
        "selected_miss_tokens",
        "max_decode_seq_len",
        "prefix_selected_tokens",
        "tail_selected_tokens",
        "invalid_miss_plan_rows",
    )

    def __init__(self, device, interval):
        if interval <= 0:
            raise ValueError("SGLANG_DEBUG_NPU_C8_OFFLOAD_INTERVAL must be positive")
        self.counts = torch.zeros(len(self.KEYS), dtype=torch.int64, device=device)
        self.interval = interval
        self.forwards = 0
        self.last_values = None
        logger.warning(
            "C8_OFFLOAD_TRACE enabled: diagnostic tensor counters and periodic host synchronization; disable for performance measurements"
        )

    def reset(self):
        # Called once after all startup graph captures/warmups have finished.
        self.counts.zero_()
        self.forwards = 0
        self.last_values = None

    def record_d2h(self, mask, decode):
        self.counts[2 if decode else 0].add_(mask.sum(dtype=torch.int64))

    def record_prefill_h2d(self, lengths):
        self.counts[1].add_(sum(lengths))

    def record_decode(self, plan):
        valid = plan.valid_rows
        sparse = valid & plan.sparse_rows
        misses = torch.where(sparse, plan.miss_counts, 0)
        self.counts[3].add_(1)
        self.counts[4].add_(valid.sum(dtype=torch.int64))
        self.counts[5].add_(sparse.sum(dtype=torch.int64))
        self.counts[6].add_((misses > 0).sum(dtype=torch.int64))
        self.counts[7].add_(misses.sum(dtype=torch.int64))
        self.counts[8].add_(
            torch.where(sparse, plan.topk_misses, 0).sum(dtype=torch.int64)
        )
        self.counts[9].copy_(
            torch.maximum(self.counts[9], torch.where(valid, plan.seq_lens, 0).max())
        )
        selected = sparse[:, None] & (plan.source_ids >= 0)
        prefix = selected & (plan.source_ids < plan.prefix_lens[:, None])
        tail = (
            selected
            & (plan.source_ids >= plan.prefix_lens[:, None])
            & (plan.source_ids < plan.seq_lens[:, None])
        )
        prefix_counts = prefix.sum(dim=1, dtype=torch.int64)
        self.counts[10].add_(prefix_counts.sum())
        self.counts[11].add_(tail.sum(dtype=torch.int64))
        self.counts[12].add_(
            (
                sparse & ((plan.topk_misses < 0) | (plan.topk_misses > prefix_counts))
            ).sum(dtype=torch.int64)
        )

    @staticmethod
    def hit_rate(selected, misses, invalid):
        if selected <= 0 or invalid or not 0 <= misses <= selected:
            return None
        return 100.0 * (selected - misses) / selected

    def maybe_log(self, phase):
        if get_is_capture_mode():
            return
        self.forwards += 1
        if self.forwards != 1 and self.forwards % self.interval:
            return
        values = dict(zip(self.KEYS, self.counts.cpu().tolist()))
        previous = self.last_values or dict.fromkeys(self.KEYS, 0)
        self.last_values = values.copy()
        values["prefix_cache_hit_rate_pct"] = self.hit_rate(
            values["prefix_selected_tokens"],
            values["selected_miss_tokens"],
            values["invalid_miss_plan_rows"],
        )
        values["window_prefix_cache_hit_rate_pct"] = self.hit_rate(
            values["prefix_selected_tokens"] - previous["prefix_selected_tokens"],
            values["selected_miss_tokens"] - previous["selected_miss_tokens"],
            values["invalid_miss_plan_rows"] - previous["invalid_miss_plan_rows"],
        )
        values["prefix_cache_hits"] = (
            values["prefix_selected_tokens"] - values["selected_miss_tokens"]
            if values["prefix_cache_hit_rate_pct"] is not None
            else None
        )
        values.update(
            phase=phase,
            forwards=self.forwards,
            scope="cumulative per rank since startup capture reset; includes service warmups; cold and warm misses combined",
            prefill_d2h_bytes=values["prefill_d2h_tokens"] * 656,
            prefill_h2d_bytes=values["prefill_h2d_tokens"] * 656,
            decode_d2h_bytes=values["decode_d2h_tokens"] * 656,
            cache_refill_payload_bytes=values["cache_refill_tokens"] * 656,
        )
        logger.info("C8_OFFLOAD_TRACE %s", json.dumps(values))


def c8_trace_for_backend(backend):
    manager = getattr(backend, "sparse_kv_manager", None)
    return getattr(getattr(manager, "c8_cache", None), "trace", None)
