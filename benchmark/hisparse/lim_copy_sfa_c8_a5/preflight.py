"""Check A5 C8 prerequisites before allocating a model or running kernels."""

import argparse
import json
import os
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path


def check(mode, kernel_source=None):
    if mode == "lim_copy_sfa" and kernel_source:
        source = Path(kernel_source).resolve()
        registration = source / "csrc/a5_c8_registration.cpp"
        if not registration.is_file():
            raise RuntimeError(
                f"Kernel checkout lacks A5 C8: {source}\n"
                f"Missing source: {registration}\n"
                "Use McZyWu/sgl-kernel-npu branch "
                "feat/hisparse-lim-copysfa-c8-a5 (442023d6 or a descendant), "
                "then run container.sh build-kernel before testing."
            )
    try:
        import acl
        import torch
        import torch_npu
    except ImportError as error:
        raise RuntimeError(
            "C8 preflight requires torch, torch_npu and the CANN Python runtime. "
            "Source /usr/local/Ascend/ascend-toolkit/set_env.sh first. "
            f"Missing dependency: {error}"
        ) from error

    architecture, status = acl.rt.get_device_info(torch.npu.current_device(), 601)
    record = {
        "mode": mode,
        "architecture": architecture,
        "acl_status": status,
        "device": torch.npu.get_device_name(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "kv_format": "C8/656 bytes",
    }
    if status or architecture != 3510:
        raise RuntimeError(
            f"C8 offload requires Ascend950/A5 (arch35); detected {record}"
        )
    if not hasattr(torch_npu, "npu_quant_lightning_indexer") or not hasattr(
        torch_npu, "npu_kv_quant_sparse_flash_attention"
    ):
        raise RuntimeError(
            "torch_npu lacks the A5 quant LI / quant sparse attention APIs."
        )
    if mode == "lim_copy_sfa":
        from sglang.kernels.ops.attention.npu_quant_sparse import (
            load_quant_copy_sfa_operator,
            load_quant_lim_operator,
        )

        spec = find_spec("sgl_kernel_npu")
        candidate = spec.origin if spec is not None else "not installed"
        try:
            kernel_version = version("sgl-kernel-npu")
        except PackageNotFoundError:
            kernel_version = "unknown"
        try:
            load_quant_lim_operator()
            load_quant_copy_sfa_operator()
        except (RuntimeError, OSError, ImportError) as error:
            details = []
            cause = error
            while cause is not None:
                details.append(f"{type(cause).__name__}: {cause}")
                cause = cause.__cause__
            raise RuntimeError(
                f"Installed kernel version: {kernel_version}\n"
                f"Kernel import path: {candidate}\n"
                + "\nCaused by: ".join(details)
                + "\nRebuild/install the matching Ascend950 C8 wheel with "
                "container.sh build-kernel, then use container.sh test meta."
            ) from error
        import sgl_kernel_npu

        import sglang

        record["sglang_source"] = sglang.__file__
        record["kernel_package"] = sgl_kernel_npu.__file__
        record["kernel_distribution_version"] = kernel_version
        for name in (
            "unidex_copy",
            "shm_allocator_create_and_register",
            "shm_allocator_free_all",
        ):
            if not hasattr(torch.ops.npu, name):
                raise RuntimeError(f"Matching kernel wheel lacks npu::{name}")
    record["status"] = "ready"
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("baseline", "lim_copy_sfa"), default="lim_copy_sfa"
    )
    parser.add_argument("--kernel-source", default=os.environ.get("KERNEL_SOURCE_DIR"))
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.mode, args.kernel_source), indent=2))
    except RuntimeError as error:
        parser.exit(2, str(error) + "\n")
