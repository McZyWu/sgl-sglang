"""Check A5 C8 prerequisites before allocating a model or running kernels."""

import argparse
import json


def check(mode):
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

        load_quant_lim_operator()
        load_quant_copy_sfa_operator()
    record["status"] = "ready"
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("baseline", "lim_copy_sfa"), default="lim_copy_sfa"
    )
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.mode), indent=2))
    except RuntimeError as error:
        parser.exit(2, str(error) + "\n")
