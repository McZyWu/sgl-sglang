"""Optional backend hook for preparing decode KV transfer buffers."""


def prepare_decode_kv_buffers(
    *, is_ascend, manager_factory, native_export, kv_args, pool
):
    """Keep the original native exporter unless Ascend provides a factory.

    A backend factory must attach any allocation owner to ``kv_args`` and
    arrange its lifetime before returning a registration descriptor.
    """
    if is_ascend:
        prepare = getattr(manager_factory(), "prepare_decode_kv_buffers", None)
        if prepare is not None:
            return prepare(kv_args, pool)
    return native_export(pool)
