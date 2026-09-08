"""Exercise production shared-expert scheduling with CPU collective doubles.

The doubles check subgroup shapes, collective ownership, stream ordering and
hook cleanup. They do not substitute for NPU numerical or timing validation.
"""

import ast
import contextlib
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[5]
MODEL = REPO / "python/sglang/srt/models/kimi_k3.py"
SOURCE = MODEL.read_text(encoding="utf-8")


def run(
    source,
    *,
    legacy,
    fine,
    side,
    comm,
    rows,
    group_size,
    latent,
    front,
    prefix,
    fail=False
):
    events = []
    active = ["main"]

    def record(name):
        events.append((name, active[0]))

    class Tensor:
        def __init__(self, shape, value=1):
            self.shape = tuple(shape)
            self.value = value
            self.dtype = "bf16"
            self.device = "npu"

        def new_empty(self, shape):
            record("allocate")
            return Tensor(shape, 0)

        def record_stream(self, s):
            record("record_tensor:" + s.name)

        def __add__(self, other):
            assert self.shape == other.shape, (self.shape, other.shape)
            return Tensor(self.shape, self.value + other.value)

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, s):
            record("wait_stream:" + s.name)

        def wait_event(self, e):
            record("wait_event:" + e)

        def record_event(self):
            record("event")
            return self.name

    main, alt = Stream("main"), Stream("alt")

    @contextlib.contextmanager
    def stream(s):
        prev = active[0]
        active[0] = s.name
        try:
            yield
        finally:
            active[0] = prev

    class Group:
        world_size = group_size

        def all_gather_into_tensor(self, out, x):
            assert out.shape == (x.shape[0] * group_size, *x.shape[1:])
            out.value = x.value
            record("all_gather")

        def reduce_scatter_tensor(self, out, x):
            assert x.shape == (out.shape[0] * group_size, *out.shape[1:])
            out.value = x.value
            record("reduce_scatter")

    class Handle:
        def __init__(self, table, key):
            self.table = table
            self.key = key

        def remove(self):
            self.table.pop(self.key)
            record("remove_" + self.key)

    class Experts:
        def __init__(self):
            self.dispatcher = self
            self.hooks = {}

        def register_pre_dispatch_hook(self, fn):
            self.hooks["pre"] = fn
            return Handle(self.hooks, "pre")

        def register_post_dispatch_hook(self, fn):
            self.hooks["post"] = fn
            return Handle(self.hooks, "post")

        def __call__(self, x, topk):
            if "pre" in self.hooks:
                self.hooks["pre"](self, x, topk)
            record("dispatch")
            if fail:
                raise RuntimeError("injected dispatch failure")
            if "post" in self.hooks:
                self.hooks["post"](self, x)
            record("routed_gemm")
            record("combine")
            return Tensor(x.shape, 3)

    def empty_like(x):
        record("allocate_like")
        return Tensor(x.shape, 0)

    torch = NS(
        Tensor=Tensor,
        empty_like=empty_like,
        cuda=NS(current_stream=lambda: main, stream=stream),
    )

    def allreduce(x):
        record("allreduce")
        return x

    scope = {
        "torch": torch,
        "_is_npu": True,
        "TYPE_CHECKING": False,
        "use_symmetric_memory": lambda *a, **kw: contextlib.nullcontext(),
        "is_allocation_symmetric": lambda: False,
        "tensor_model_parallel_all_reduce": allreduce,
        "_add3": lambda a, b, c: a + b if c is None else a + b + c,
    }
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "KimiK3MoE"
    )
    wanted = {
        "_gather_shared_expert_inputs",
        "_reduce_scatter_shared_experts",
        "_prepare_shared_experts_input",
        "_finalize_shared_experts_output",
        "_forward_shared_experts",
        "_forward_unfused",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    for n in methods:
        n.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(module), "<actual-production-methods>", "exec"
        ),
        scope,
    )
    obj = type("ProductionMethods", (), {n.name: scope[n.name] for n in methods})()

    def shared(x):
        record("shared_gemm")
        return Tensor(x.shape, 2)

    def ep_front(x):
        record("front")
        return ("topk", x) if front else None

    def gate(x):
        record("gate")
        return x

    def down(x):
        record("down")
        return x, None

    def norm(x):
        record("norm")
        return x

    def up(x):
        record("up")
        return x, None

    obj.__dict__.update(
        _shared_experts_tp_group=Group(),
        _shared_experts_tp_comm=comm,
        _shared_experts_tp1=not comm,
        _sbo_shared_overlap=side,
        _npu_overlap_shared_rs=legacy,
        alt_stream=alt,
        shared_experts=shared,
        _can_overlap_shared_experts_npu=lambda x: bool(
            fine and side and comm and latent and rows
        ),
        _ep_front=ep_front,
        _ep_front_overlap=lambda x: None,
        experts=Experts(),
        _use_mega_moe=False,
        use_latent_moe=latent,
        gate=gate,
        topk=lambda x, g: "topk",
        routed_expert_down_proj=down,
        _reduce_latent=norm,
        routed_expert_up_proj=up,
        tp_size=32,
        _ep_a2a=True,
        moe_hidden_size=8,
    )
    x = Tensor((rows, 8))
    try:
        out = obj._forward_unfused(x, prefix_sum=Tensor(x.shape, 7) if prefix else None)
    except RuntimeError as exc:
        assert fail and str(exc) == "injected dispatch failure"
        assert not obj.experts.hooks, "dispatch hooks leaked"
        return None, events
    assert not fail
    assert out.shape == x.shape
    assert not obj.experts.hooks, "dispatch hooks leaked"
    assert sum(n == "all_gather" for n, _ in events) == int(bool(rows and comm)), events
    assert sum(n == "reduce_scatter" for n, _ in events) == int(
        bool(rows and comm)
    ), events
    return (out.shape, out.value), events


class TestSharedExpertSubgroupOverlap(unittest.TestCase):
    def test_collective_ownership_and_order(self):
        for fine, side, comm, rows, size, latent, front, prefix in itertools.product(
            (False, True),
            (False, True),
            (False, True),
            (0, 2),
            (1, 4, 32),
            (False, True),
            (False, True),
            (False, True),
        ):
            kw = dict(
                fine=fine,
                side=side,
                comm=comm,
                rows=rows,
                group_size=size,
                latent=latent,
                front=front,
                prefix=prefix,
            )
            with self.subTest(**kw):
                off, off_events = run(SOURCE, legacy=False, **kw)
                on, on_events = run(SOURCE, legacy=True, **kw)
                self.assertEqual(off, on)
                fine_eligible = fine and side and comm and latent and rows
                if fine_eligible or not (side and comm and rows):
                    # Fine-grained scheduling owns RS regardless of legacy flag.
                    self.assertEqual(off_events, on_events)
                else:
                    self.assertIn(("reduce_scatter", "main"), off_events)
                    self.assertIn(("reduce_scatter", "alt"), on_events)
                    self.assertLess(
                        on_events.index(("combine", "main")),
                        on_events.index(("reduce_scatter", "alt")),
                    )
                    if latent:
                        self.assertLess(
                            on_events.index(("reduce_scatter", "alt")),
                            on_events.index(("norm", "main")),
                        )

    def test_dispatch_failure_cleans_up_hooks(self):
        for legacy, size in itertools.product((False, True), (1, 4, 32)):
            with self.subTest(legacy=legacy, size=size):
                run(
                    SOURCE,
                    legacy=legacy,
                    fine=True,
                    side=True,
                    comm=True,
                    rows=2,
                    group_size=size,
                    latent=True,
                    front=True,
                    prefix=False,
                    fail=True,
                )


if __name__ == "__main__":
    unittest.main()
