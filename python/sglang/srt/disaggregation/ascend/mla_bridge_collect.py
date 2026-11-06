"""Collect an Ascend sparse decode pool without exporting its absent KV planes.

This is not a manager factory and does not allocate or register a bridge ring.
The caller must keep the pool and sparse manager alive through all transfers.
"""

from dataclasses import dataclass

from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
    ROW_BYTES,
    BridgeEntry,
    BridgeGeometry,
)


@dataclass(frozen=True)
class SparseDecodeEntries:
    entries: tuple[BridgeEntry, ...]
    native_ptrs: tuple[int, ...]
    native_sizes: tuple[int, ...]
    native_strides: tuple[int, ...]
    host_tensors: tuple
    host_device_ptrs: tuple[int, ...]


def collect_sparse_decode_entries(
    pool, geometry: BridgeGeometry
) -> SparseDecodeEntries:
    geometry.validate_peer(geometry)
    if (
        not getattr(pool, "enable_sparsity_driven_kv_offload", False)
        or getattr(pool, "is_draft_worker", False)
        or getattr(pool, "k_buffer", None) is not None
        or getattr(pool, "v_buffer", None) is not None
        or getattr(pool, "dsa_kv_cache_store_fp8", False)
        or getattr(pool, "page_size", None) != 128
        or getattr(pool, "index_page_size", None) != 128
        or getattr(pool, "kv_lora_rank", None) != 512
        or getattr(pool, "qk_rope_head_dim", None) != 64
        or str(getattr(pool, "store_dtype", None)) != "torch.bfloat16"
    ):
        raise ValueError("Expected sparse target with disabled native BF16 MLA KV")
    manager = getattr(pool, "sparse_kv_manager", None)
    if manager is None:
        raise ValueError("Sparse manager must be bound before bridge collection")
    if (
        type(manager).__module__
        != "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager"
        or type(manager).__name__ != "SparseKVCacheManager"
    ):
        raise ValueError("Expected the fused sparse KV manager")
    hosts = tuple(getattr(manager, "host_kv_buffer", ()))
    host_ptrs = tuple(getattr(manager, "dev_ptr_list", ()))
    layers = tuple(getattr(pool, "model_layer_ids", ()))
    if not layers or len(hosts) != len(layers) or len(host_ptrs) != len(layers):
        raise ValueError("Sparse host buffers do not match main layer IDs")
    entries = []
    for layer, tensor, device_ptr in zip(layers, hosts, host_ptrs):
        if (
            str(tensor.dtype) != "torch.bfloat16"
            or tensor.shape[-1] != 576
            or not tensor.is_contiguous()
            or tensor.numel() % 576
            or type(device_ptr) is not int
            or device_ptr <= 0
        ):
            raise ValueError("Invalid sparse combined host mapping")
        entries.append(
            BridgeEntry(
                "staged",
                "main",
                layer,
                "combined",
                "bfloat16",
                ROW_BYTES,
                tensor.numel() // 576,
                "token",
                128,
            )
        )

    index_layers = tuple(getattr(pool, "indexer_layer_ids", ()))
    index_storage = getattr(pool, "index_k_buffer", None)
    scale_storage = getattr(pool, "index_k_scale_buffer", None)
    index = tuple(index_storage) if index_storage is not None else ()
    scales = tuple(scale_storage) if scale_storage is not None else ()
    if (
        not index_layers
        or len(index) != len(index_layers)
        or (scales and len(scales) != len(index_layers))
    ):
        raise ValueError("Index buffers do not match indexer layers")
    ptrs, sizes, strides = [], [], []
    for component, buffers in (("index", index), ("index_scale", scales)):
        for layer, tensor in zip(index_layers, buffers):
            if not tensor.is_contiguous() or tensor.numel() == 0:
                raise ValueError("Invalid native index buffer")
            stride = tensor[0].nbytes
            if stride <= 0 or tensor.nbytes % stride:
                raise ValueError("Invalid native index page geometry")
            ptrs.append(tensor.data_ptr())
            sizes.append(tensor.nbytes)
            strides.append(stride)
            entries.append(
                BridgeEntry(
                    "native",
                    "main",
                    layer,
                    component,
                    str(tensor.dtype).removeprefix("torch."),
                    stride,
                    tensor.nbytes // stride,
                    "page",
                    128,
                )
            )
    return SparseDecodeEntries(
        tuple(entries), tuple(ptrs), tuple(sizes), tuple(strides), hosts, host_ptrs
    )


@dataclass(frozen=True)
class NativePlane:
    entry: BridgeEntry
    tensor: object


def flatten_main_mla_plane(plane: NativePlane):
    """Expose token rows from a contiguous page-major P plane without copying."""
    entry, tensor = plane.entry, plane.tensor
    entry.validate()
    if entry.model != "main" or entry.component not in ("ckv", "kpe"):
        raise ValueError("Expected a native main MLA plane")
    width = 512 if entry.component == "ckv" else 64
    if (
        entry.item_stride != entry.page_size * width * 2
        or str(tensor.dtype) != "torch.bfloat16"
        or not tensor.is_contiguous()
        or tensor.shape[-1] != width
        or tensor.nbytes != entry.capacity * entry.item_stride
    ):
        raise ValueError("Native plane does not match page-major MLA storage")
    return tensor.view(-1, width)


def plan_native_component_blocks(
    source_planes,
    destination_entries,
    destination_ptrs,
    source_pages,
    destination_pages,
):
    """Plan native index/draft page writes from explicit manifest identities."""
    from sglang.srt.disaggregation.ascend.mla_bridge_contract import (
        pair_bridge_entries,
        plan_native_page_transfers,
    )

    source_planes = tuple(source_planes)
    destination_entries = tuple(destination_entries)
    destination_ptrs = tuple(destination_ptrs)
    pairs = pair_bridge_entries([p.entry for p in source_planes], destination_entries)
    native_destinations = [e for e in destination_entries if e.kind == "native"]
    if len(destination_ptrs) != len(native_destinations):
        raise ValueError("Native pointer count does not match the manifest")
    source_by_id = {
        (p.entry.model, p.entry.layer, p.entry.component): p for p in source_planes
    }
    dst_by_id = {
        (e.model, e.layer, e.component): (e, ptr)
        for e, ptr in zip(native_destinations, destination_ptrs)
    }
    blocks = []
    for src, dst in pairs:
        if dst.kind != "native":
            continue
        key = (src.model, src.layer, src.component)
        plane = source_by_id[key]
        target, ptr = dst_by_id[key]
        if target != dst or plane.tensor.nbytes != src.capacity * src.item_stride:
            raise ValueError("Native tensor no longer matches its manifest")
        planned = plan_native_page_transfers(
            source_pages,
            destination_pages,
            src_base=plane.tensor.data_ptr(),
            dst_base=ptr,
            src_bytes=plane.tensor.nbytes,
            dst_bytes=dst.capacity * dst.item_stride,
            src_stride=src.item_stride,
            dst_stride=dst.item_stride,
            item_bytes=src.item_stride,
        )
        blocks.append((key, planned))
    if len(blocks) != len(native_destinations):
        raise ValueError("Native transfer manifest is incomplete")
    return tuple(blocks)


def collect_native_prefill_planes(pool, draft_pool=None):
    """Retain ordered native P planes with explicit model/component identities."""
    planes = []
    for model, current in (("main", pool), ("draft", draft_pool)):
        if current is None:
            continue
        if (
            getattr(current, "enable_sparsity_driven_kv_offload", False)
            or getattr(current, "k_buffer", None) is None
            or getattr(current, "v_buffer", None) is None
            or getattr(current, "dsa_kv_cache_store_fp8", False)
            or getattr(current, "page_size", None) != 128
            or (
                getattr(current, "indexer_layer_ids", ())
                and getattr(current, "index_page_size", None) != 128
            )
            or getattr(current, "kv_lora_rank", None) != 512
            or getattr(current, "qk_rope_head_dim", None) != 64
            or bool(getattr(current, "is_draft_worker", False)) != (model == "draft")
        ):
            raise ValueError("Expected native BF16 split MLA prefill planes")
        layers = tuple(getattr(current, "model_layer_ids", ()))
        k, v = tuple(current.k_buffer), tuple(current.v_buffer)
        if (
            not layers
            or len(set(layers)) != len(layers)
            or len(k) != len(layers)
            or len(v) != len(layers)
        ):
            raise ValueError("Native KV planes do not match layer IDs")
        for component, buffers, width in (("ckv", k, 512), ("kpe", v, 64)):
            for layer, tensor in zip(layers, buffers):
                if (
                    not tensor.is_contiguous()
                    or str(tensor.dtype) != "torch.bfloat16"
                    or tensor.shape[-1] != width
                    or tensor.numel() % width
                ):
                    raise ValueError("Invalid native MLA plane")
                stride = tensor[0].nbytes
                if stride != 128 * width * 2 or tensor.nbytes % stride:
                    raise ValueError("Native MLA plane is not page aligned")
                capacity = tensor.nbytes // stride
                planes.append(
                    NativePlane(
                        BridgeEntry(
                            "native",
                            model,
                            layer,
                            component,
                            "bfloat16",
                            stride,
                            capacity,
                            "page",
                            128,
                        ),
                        tensor,
                    )
                )
        index_layers = tuple(getattr(current, "indexer_layer_ids", ()))
        index_storage = getattr(current, "index_k_buffer", None)
        scale_storage = getattr(current, "index_k_scale_buffer", None)
        if len(set(index_layers)) != len(index_layers) or (
            index_layers and index_storage is None
        ):
            raise ValueError("Native index layer set is invalid")
        for component, storage in (
            ("index", index_storage),
            ("index_scale", scale_storage),
        ):
            buffers = tuple(storage) if storage is not None else ()
            if buffers and len(buffers) != len(index_layers):
                raise ValueError("Native index planes do not match index layer IDs")
            for layer, tensor in zip(index_layers, buffers):
                if not tensor.is_contiguous() or tensor.numel() == 0:
                    raise ValueError("Invalid native index plane")
                stride = tensor[0].nbytes
                if stride <= 0 or tensor.nbytes % stride:
                    raise ValueError("Invalid native index page geometry")
                planes.append(
                    NativePlane(
                        BridgeEntry(
                            "native",
                            model,
                            layer,
                            component,
                            str(tensor.dtype).removeprefix("torch."),
                            stride,
                            tensor.nbytes // stride,
                            "page",
                            128,
                        ),
                        tensor,
                    )
                )
    return tuple(planes)
