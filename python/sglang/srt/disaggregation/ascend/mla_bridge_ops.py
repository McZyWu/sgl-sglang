"""NPU data movement primitives for an experimental MLA PD bridge.

These helpers do not register memory or enable the bridge. The caller must
validate physical IDs, own all tensors until the stream completes, and gate
slot reuse and request success on transport/scatter completion.
"""


def pack_mla_rows(k, v, indices, packed, k_workspace, v_workspace):
    """Gather split BF16 rows into an owned [n, 576] staging view.

    All workspace is supplied by the caller; no temporary gather tensor is
    allocated here. Operations are enqueued on the current device stream.
    """
    import torch

    if any(t.dtype != torch.bfloat16 for t in (k, v, packed, k_workspace, v_workspace)):
        raise ValueError("MLA bridge requires BF16 storage")
    if k.ndim != 2 or k.shape[1] != 512 or v.ndim != 2 or v.shape[1] != 64:
        raise ValueError("Expected flattened split CKV512 and KPE64")
    if k.shape[0] != v.shape[0] or indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("Invalid physical indices or source row counts")
    n = indices.numel()
    if n == 0 or packed.shape != (n, 576):
        raise ValueError("Invalid packed row geometry")
    if k_workspace.shape != (n, 512) or v_workspace.shape != (n, 64):
        raise ValueError("Invalid gather workspace geometry")
    if not all(
        t.is_contiguous() for t in (k, v, indices, packed, k_workspace, v_workspace)
    ):
        raise ValueError("MLA bridge buffers must be contiguous")
    if len({t.device for t in (k, v, indices, packed, k_workspace, v_workspace)}) != 1:
        raise ValueError("MLA bridge buffers must share a device")
    buffers = (k, v, indices, packed, k_workspace, v_workspace)
    ranges = [
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in buffers
    ]
    if any(
        a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1 :]
    ):
        raise ValueError("MLA bridge buffers must not overlap")
    torch.index_select(k, 0, indices, out=k_workspace)
    torch.index_select(v, 0, indices, out=v_workspace)
    packed[:, :512].copy_(k_workspace)
    packed[:, 512:].copy_(v_workspace)


def scatter_mla_rows(packed, host, src_indices, dst_indices, valid, *, host_dev_ptr):
    """Enqueue packed rows into the physical host allocation via R6 unidex."""
    from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace

    if host_dev_ptr <= 0 or packed.ndim != 2 or host.ndim != 2:
        raise ValueError("Invalid host mapping or row rank")
    if packed.shape[1] != 576 or host.shape[1] != 576:
        raise ValueError("Invalid combined MLA geometry")
    return unidex_copy_inplace(
        packed, host, src_indices, dst_indices, valid, 1, 1, dst_ptr=host_dev_ptr
    )
