"""CPU dispatch checks; no NPU or serving dependencies are required."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def variants():
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/layers/attention/graph_variants.py"
    )
    spec = importlib.util.spec_from_file_location("sparse_graph_variants_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
        yield module


@pytest.mark.parametrize(
    "lengths,expected",
    [
        ([2048, 4096], "sparse_kv_long"),
        ([2048, 0], "sparse_kv_long"),
        ([2047, 4096], "sparse_kv_resident"),
        ([1], "sparse_kv_resident"),
        ([0, 0], "sparse_kv_long"),
        ([], "sparse_kv_long"),
        (None, "sparse_kv_resident"),
    ],
)
def test_graph_variant_matches_live_request_lengths(variants, lengths, expected):
    batch = SimpleNamespace(
        seq_lens_cpu=None if lengths is None else torch.tensor(lengths)
    )
    assert variants.SparseKVGraphVariants(2048).select(batch) == expected
    assert variants.skip_sparse_kv_residency(batch, 2048, is_capturing=False) == (
        expected == "sparse_kv_long"
    )


def test_unlabelled_capture_keeps_dynamic_short_path(variants):
    batch = SimpleNamespace(seq_lens_cpu=torch.tensor([4096, 8192]))
    assert not variants.skip_sparse_kv_residency(batch, 2048, is_capturing=True)


@pytest.mark.parametrize("fixed", [False, True])
def test_capture_variant_is_fixed_independently_of_dummy_lengths(variants, fixed):
    batch = SimpleNamespace(
        seq_lens_cpu=torch.tensor([1]), sparse_kv_skip_residency=fixed
    )
    assert variants.skip_sparse_kv_residency(batch, 2048, is_capturing=True) == fixed
    batch.seq_lens_cpu = torch.tensor([4096])
    assert variants.skip_sparse_kv_residency(batch, 2048, is_capturing=False) == fixed


def test_long_mixed_long_replay_selects_distinct_graphs(variants):
    selector = variants.SparseKVGraphVariants(2048)
    batch = SimpleNamespace(seq_lens_cpu=torch.tensor([4096, 8192]))
    assert selector.select(batch) == variants.SPARSE_KV_LONG
    batch.seq_lens_cpu[0] = 1024  # rollback or a newly admitted short request
    assert selector.select(batch) == variants.SPARSE_KV_RESIDENT
    batch.seq_lens_cpu[0] = 2048
    assert selector.select(batch) == variants.SPARSE_KV_LONG


def test_device_lengths_do_not_trigger_a_scalar_read(variants):
    batch = SimpleNamespace(seq_lens_cpu=torch.empty(2, device="meta"))
    assert (
        variants.SparseKVGraphVariants(2048).select(batch)
        == variants.SPARSE_KV_RESIDENT
    )
