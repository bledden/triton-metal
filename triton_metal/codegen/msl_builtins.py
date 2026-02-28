"""Metal built-in function and qualifier mappings for Triton operations."""

# Triton's tt.get_program_id(axis) maps to Metal's threadgroup position.
# These are injected as kernel parameters with Metal attribute qualifiers.
PROGRAM_ID_QUALIFIERS = {
    0: ("uint", "tgid_x", "threadgroup_position_in_grid"),
    1: ("uint", "tgid_y", "threadgroup_position_in_grid"),
    2: ("uint", "tgid_z", "threadgroup_position_in_grid"),
}

# Thread-level position qualifiers.
THREAD_QUALIFIERS = {
    "thread_id": ("uint", "tid", "thread_position_in_grid"),
    "local_id": ("uint", "lid", "thread_position_in_threadgroup"),
    "simd_lane": ("uint", "simd_lane", "thread_index_in_simdgroup"),
    "simd_group": ("uint", "simd_group", "simdgroup_index_in_threadgroup"),
    "tg_size": ("uint", "tg_size", "threads_per_threadgroup"),
}

# SIMD-group reduction intrinsics (Metal built-ins).
SIMD_REDUCTIONS = {
    "sum": "simd_sum",
    "max": "simd_max",
    "min": "simd_min",
    "and": "simd_and",
    "or": "simd_or",
    "xor": "simd_xor",
}

# Metal memory barrier functions.
BARRIERS = {
    "threadgroup": "threadgroup_barrier(mem_flags::mem_threadgroup)",
    "device": "threadgroup_barrier(mem_flags::mem_device)",
    "all": "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)",
}

# Metal address space qualifiers.
ADDRESS_SPACES = {
    "global": "device",
    "shared": "threadgroup",
    "local": "thread",
    "constant": "constant",
}
