"""CPU checks for NZ writes through the opt-in dispatch and its fallbacks."""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMLANZIndicesDispatch(unittest.TestCase):
    def test_writes_and_fallbacks(self):
        repo = Path(__file__).resolve().parents[5]
        path = repo / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
        tree = ast.parse(path.read_text())
        helper = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_mla_fia_nz_scatter_indices"
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "NPUMLATokenToKVPool"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_set_fia_nz_kv_buffer"
        )

        def scatter(dst, indices, src):
            dst.index_copy_(0, indices.flatten().long(), src)

        scope = {
            "torch": torch,
            "torch_npu": types.SimpleNamespace(npu_scatter_nd_update_=scatter),
        }
        exec(
            compile(
                ast.Module(body=[helper, method], type_ignores=[]), str(path), "exec"
            ),
            scope,
        )
        store = scope[method.name]
        module_name = "sgl_kernel_npu.mem_cache.mla_nz_indices"
        module = types.ModuleType(module_name)
        calls = []

        def indices(loc, k_dim, r_dim, page):
            calls.append(loc)
            return tuple(scope[helper.name](loc, dim, page) for dim in (k_dim, r_dim))

        module.build_mla_nz_scatter_indices = indices
        torch.manual_seed(3)
        with patch.dict(sys.modules, {module_name: module}):
            for rows in (32, 256, 33):
                for dtype in (torch.int32, torch.int64):
                    for page in (16, 128):
                        for k_dim, r_dim in ((512, 64), (32, 16)):
                            with self.subTest(
                                rows=rows, dtype=dtype, page=page, k_dim=k_dim
                            ):
                                pages = (rows + 1 + page - 1) // page + 1
                                source = torch.randn(rows, k_dim + r_dim + 16)
                                k, r = (
                                    source[:, :k_dim],
                                    source[:, k_dim : k_dim + r_dim],
                                )
                                loc = torch.arange(1, rows + 1, dtype=dtype)
                                expected = []
                                for dim, value in ((k_dim, k), (r_dim, r)):
                                    logical = torch.zeros(pages, page, 1, dim)
                                    logical.view(-1, dim)[loc.long()] = value
                                    expected.append(
                                        logical.reshape(pages, page, dim // 16, 16)
                                        .permute(0, 2, 1, 3)
                                        .contiguous()
                                        .reshape_as(logical)
                                    )
                                for enabled in (False, True):
                                    caches = [torch.zeros_like(x) for x in expected]
                                    obj = types.SimpleNamespace(
                                        start_layer=0,
                                        page_size=page,
                                        k_buffer=[caches[0]],
                                        v_buffer=[caches[1]],
                                        kv_lora_rank=k_dim,
                                        qk_rope_head_dim=r_dim,
                                        use_fused_mla_nz_indices=enabled,
                                    )
                                    calls.clear()
                                    store(obj, 0, loc, k, r)
                                    for actual, reference in zip(caches, expected):
                                        self.assertTrue(torch.equal(actual, reference))
                                    eligible = (
                                        enabled
                                        and rows in (32, 256)
                                        and dtype == torch.int32
                                        and page == 128
                                        and k_dim == 512
                                    )
                                    self.assertEqual(len(calls), int(eligible))


if __name__ == "__main__":
    unittest.main()
