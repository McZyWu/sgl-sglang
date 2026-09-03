from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.runner import base_cuda_graph_runner
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    get_batch_sizes_to_capture,
)
from sglang.srt.speculative.dspark_components import dspark_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def graph_runtime(monkeypatch):
    monkeypatch.setattr(
        base_cuda_graph_runner,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(bs=[1, 4, 32])
                ),
                torch_compile_max_bs=32,
            ),
            overlap=SimpleNamespace(enable_two_batch_overlap=False),
        ),
    )
    monkeypatch.setattr(
        base_cuda_graph_runner,
        "get_flags",
        lambda: SimpleNamespace(capture=SimpleNamespace(enable_torch_compile=False)),
    )
    monkeypatch.setattr(
        base_cuda_graph_runner,
        "get_cuda_graph_batch_size_alignment",
        lambda: 32,
    )
    monkeypatch.setattr(
        base_cuda_graph_runner,
        "get_cuda_graph_max_batch_size",
        lambda size: size,
    )
    monkeypatch.setattr(base_cuda_graph_runner, "is_npu", lambda: True)


def _runner(*, draft: bool, dspark: bool):
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(size=32),
        is_draft_worker=draft,
        spec_algorithm=SimpleNamespace(is_dspark=lambda: dspark),
    )


def test_generic_dspark_draft_captures_local_batch_sizes(graph_runtime, monkeypatch):
    monkeypatch.setattr(dspark_config, "draft_is_deepseek_v4", lambda: False)

    capture_bs, compile_bs = get_batch_sizes_to_capture(
        _runner(draft=True, dspark=True)
    )

    assert capture_bs == [1, 4, 32]
    assert compile_bs == []


def test_dspark_target_keeps_verify_width_alignment(graph_runtime):
    capture_bs, _ = get_batch_sizes_to_capture(
        _runner(draft=False, dspark=True), captured_req_width=8
    )

    # TP32 gathered target rows require B*8 to be divisible by 32. The draft
    # override above must not leak into the target runner.
    assert capture_bs == [4, 32]


@pytest.mark.parametrize(
    ("runner", "is_deepseek_v4"),
    [
        (_runner(draft=False, dspark=True), False),
        (_runner(draft=True, dspark=False), False),
        (_runner(draft=True, dspark=True), True),
    ],
)
def test_target_and_non_generic_drafts_keep_target_alignment(
    graph_runtime, monkeypatch, runner, is_deepseek_v4
):
    monkeypatch.setattr(dspark_config, "draft_is_deepseek_v4", lambda: is_deepseek_v4)

    capture_bs, _ = get_batch_sizes_to_capture(runner)

    assert capture_bs == [32]
