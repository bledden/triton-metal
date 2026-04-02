"""Triton type to MSL type mappings."""

# FP64 is silently downcast to FP32 — Metal has no double precision.
_UNSUPPORTED_TYPES = set()

# Triton dtype string -> MSL type string
_TYPE_MAP = {
    "fp16": "half",
    "f16": "half",
    "bf16": "bfloat",  # Metal 3.1+ supports bfloat
    "fp32": "float",
    "f32": "float",
    "fp64": "float",  # Metal has no double — downcast to float32
    "f64": "float",   # Metal has no double — downcast to float32
    "i1": "bool",
    "i8": "char",
    "i16": "short",
    "i32": "int",
    "i64": "long",
    "u8": "uchar",
    "u16": "ushort",
    "u32": "uint",
    "u64": "ulong",
}

# Pointer types map to device pointers.
_PTR_QUALIFIER = "device"


def _check_fp64(triton_type: str):
    """Reject FP64 types — Apple Silicon GPUs have no FP64 hardware."""
    base = triton_type.lstrip("*").strip()
    if base in _UNSUPPORTED_TYPES:
        raise TypeError(
            f"FP64 (double) is not supported on Apple Silicon GPUs. "
            f"Got type '{triton_type}'. Cast to float32 before running on Metal."
        )


def triton_type_to_msl(triton_type: str) -> str:
    """Convert a Triton type string to its MSL equivalent.

    Args:
        triton_type: e.g. "fp32", "*fp16", "i32"

    Returns:
        MSL type string, e.g. "float", "device half*", "int"

    Raises:
        TypeError: If the type is FP64 (not supported on Apple Silicon).
    """
    _check_fp64(triton_type)
    if triton_type.startswith("*"):
        inner = triton_type[1:]
        msl_inner = _TYPE_MAP.get(inner, inner)
        return f"{_PTR_QUALIFIER} {msl_inner}*"
    return _TYPE_MAP.get(triton_type, triton_type)


def triton_type_to_msl_const_ref(triton_type: str) -> str:
    """Convert a scalar Triton type to a constant reference MSL type.

    Used for kernel arguments passed as constant buffers.
    """
    msl_type = triton_type_to_msl(triton_type)
    return f"constant {msl_type}&"
