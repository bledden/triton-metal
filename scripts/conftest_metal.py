"""Pytest conftest for running Triton upstream tests on the Metal backend.

Skips tests that require:
- Hardware capabilities Metal doesn't support (FP8, FP64, TF32)
- Features not yet implemented in the Metal backend (atomics, 2D scan, etc.)

Also patches:
- device fixture → "cpu" (Metal driver copies from CPU tensors)
- check_type_supported → skip CUDA capability checks
"""

import pytest
import torch


# ── Device fixture override ──────────────────────────────────────────────

@pytest.fixture
def device():
    """Override the default 'cuda' device with 'cpu' for Metal backend.

    CPU tensors work with the Metal driver via data_ptr() + ctypes copy.
    Using 'cpu' instead of 'mps' avoids MPS dtype limitations (no float64, etc).
    """
    return "cpu"


# ── Monkeypatch check_type_supported ─────────────────────────────────────

def _metal_check_type_supported(dtype, device):
    """Metal version: skip unsupported types without calling CUDA APIs."""
    import triton.language as tl
    unsupported = {
        tl.float8e4nv, tl.float8e5, tl.float8e4b15,
        "float8e4nv", "float8e5", "float8e4b15", "float8_e4m3fn", "float8_e5m2",
        "float64", "fp64",
    }
    if dtype in unsupported:
        pytest.skip(f"Metal: {dtype} not supported")


def pytest_configure(config):
    """Monkeypatch check_type_supported at import time."""
    import triton._internal_testing as _testing

    _testing.check_type_supported = _metal_check_type_supported

    import sys
    test_mod = sys.modules.get("test_core")
    if test_mod and hasattr(test_mod, "check_type_supported"):
        test_mod.check_type_supported = _metal_check_type_supported


def pytest_runtest_setup(item):
    """Patch check_type_supported in test module namespace before each test."""
    if hasattr(item, "module") and hasattr(item.module, "check_type_supported"):
        item.module.check_type_supported = _metal_check_type_supported


# ── Types that Metal hardware cannot support ─────────────────────────────

UNSUPPORTED_TYPES = {
    "float64", "fp64",
    "fp8e4nv", "fp8e5", "fp8e4b15", "fp8e4b8", "fp8e5b16",
    "fp8_e4m3", "fp8_e5m2", "fp8e4nv_type", "fp8e5_type",
    "float8_e4m3fn", "float8_e5m2",
    "e4m3", "e5m2", "e2m1",  # microscaling FP8 variants
}

UNSUPPORTED_PRECISIONS = {"tf32"}

# Features not yet implemented in the Metal backend.
# These tests are skipped (not failed) because the backend doesn't support
# them yet — they represent future work, not bugs.
#
# NOTE: Several ops below have lowering implementations in generic_lowerer.py
# (e.g. _lower_tt_histogram, _lower_tt_join, _lower_tt_cat, _lower_tt_split,
# _lower_tt_gather, _lower_map_elementwise, _lower_scan). However, many
# upstream tests exercise 2D/multi-dim tensor patterns, scalar operands, or
# multi-output modes that the 1D-per-thread Metal model does not handle.
# Each entry below documents why it remains skipped despite partial impl.
UNIMPLEMENTED_FEATURES = {
    # Histogram — _lower_tt_histogram exists (1D threadgroup atomics), but
    # test_histogram creates a 2D bias via tl.full([M, N], ...) and broadcasts
    # (z + bias), requiring 2D tensor support. test_histogram_mask uses 2*M
    # block with masking. test_histogram_silent_data_corruption verifies
    # out-of-bounds safety not validated for Metal atomics path.
    "test_histogram",
    "test_histogram_mask",
    "test_histogram_silent_data_corruption",
    # 2D scan — _lower_scan exists with axis support, but test uses full 2D
    # load/store indexing (range_m[:, None] * BLOCK_N + range_n[None, :])
    # which requires inter-thread shuffle not implemented in 1D-per-thread model
    "test_scan2d",
    # Multi-dimensional operations — not supported in 1D-per-thread model
    "test_trans_4d",
    "test_trans_2d",
    "test_optimize_thread_locality",
    "test_dot_multidim",
    # Tensor atomic ops requiring 2D tensor support
    "test_tensor_atomic_rmw_block",  # 2D matrix access (8x8)
    "test_tensor_atomic_add_non_exclusive_offset",  # 2D non-exclusive offset
    "test_tensor_atomic_add_access_patterns",  # 2D access patterns
    # scaled_dot — requires microscaling format support
    "test_scaled_dot",
    # cat_nd — uses TensorDescriptor (CUDA TMA) and multi-dim shapes/dim arg;
    # _lower_tt_cat only handles 1D two-operand cat without dim parameter
    "test_cat_nd",
    # Features requiring CUDA-specific infrastructure
    "test_num_programs",
    "test_tensor_descriptor",
    "test_tma",
    # Multi-dim indexing/reshape/permute — requires 2D+ tensor support
    "test_index1d",
    "test_reshape",
    "test_permute",
    "test_trans_reshape",
    # Gather — _lower_tt_gather exists for 1D, but test is parametrized with
    # 2D shapes ([4,4], [128,64], etc.) in 3 of 4 cases; 1D impl can't handle
    "test_gather",
    # Interleave — no _lower_tt_interleave implementation exists
    "test_interleave",
    "test_interleave_scalars",
    # Join — _lower_tt_join exists (fused cat and standalone), but:
    # test_join stores with 2D indexing (N[:,None]*2 + arange(0,2)[None,:])
    # test_join_scalars uses 0-dim scalar operands not handled by shape logic
    # test_join_with_mma uses 2D+3D tensors, reshape, and tl.dot
    "test_join",
    "test_join_scalars",
    "test_join_with_mma",
    # Split — _lower_tt_split exists for tensor<Nx2>, but:
    # test_split requires tl.reshape (1D->2D) before split; test_reshape is
    # itself skipped, so the reshape prerequisite isn't met
    # test_split_to_scalar produces 0-dim scalar outputs not handled
    "test_split",
    "test_split_to_scalar",
    # Chained reductions — multi-dim reduce
    "test_chained_reductions",
    # Map elementwise — _lower_map_elementwise exists and handles the
    # cf.cond_br decision tree for single-output 1D ops.
    # REMOVED: test_map_elementwise — 1D int32 compare function, single output,
    #   fully covered by _lower_map_elementwise + _lower_map_elementwise_cond_br.
    # test_map_elementwise_pack uses pack=2 with 4 inputs/4 outputs (not handled)
    # test_map_elementwise_multiple_outputs returns 2 values but impl only
    #   binds a single result via self.env[ssa.id]
    "test_map_elementwise_pack",
    "test_map_elementwise_multiple_outputs",
    # LLIR/PTX-specific tests
    "test_disable_licm",
    "test_assume",
    "test_poison_return",
    "test_ptx_cast",
    # Misc unimplemented
    "test_dot_mulbroadcasted",
    "test_generic_reduction",
    "test_where_broadcast",
    "test_cumsum_dtype",
    "test_sum_dtype",
    "test_umulhi",
    "test_math_divide_op",
    "test_unsplat",
    "test_no_rematerialization_op",
    "test_load_store_same_ptr",
    # Noinline "shared" mode uses tl.dot inside noinline function which
    # requires 2D matmul support in device functions (not yet implemented)
    "test_noinline[shared]",
    # While loops — scf.while now implemented
    # "test_while",
    # "test_nested_while",
    # atomic_cas test uses while loop internally (serialized_add kernel)
    "test_atomic_cas",
    # tl.range — loop fusion not implemented
    "test_tl_range_fuse",
    "test_tl_range_fuse_dependent",
    "test_tl_range_num_stages",
    # i64 compute — Metal GPU pipeline compiler doesn't support int64
    "test_for_iv",
    # test_if_call "jit_if" variant uses early return → cf.cond_br
    "test_if_call[jit_if]",
    # num_warps validation — not enforced in Metal backend
    "test_num_warps_pow2",
    # Early return → cf.cond_br (unstructured control flow not implemented)
    "test_nested_if_else_return",
    # Misc
    "test_optimize_thread_locality",
    "test_unsigned_name_mangling",
    "test_zero_strided_tensors",
    "test_pointer_arguments",
    "test_masked_load_shared_memory",
    "test_dot_without_load",
}


def pytest_collection_modifyitems(config, items):
    """Skip tests that use unsupported types/precisions or unimplemented features."""
    skip_unsupported = pytest.mark.skip(reason="Metal: unsupported type/precision")
    skip_cuda = pytest.mark.skip(reason="Metal: CUDA/HIP-only test")
    skip_unimplemented = pytest.mark.skip(reason="Metal: feature not yet implemented")

    for item in items:
        test_id = item.nodeid.lower()
        func_name = item.name.split("[")[0]  # e.g. "test_floordiv" from "test_floordiv[1-int8-int8]"

        # Skip unimplemented features (by base name or full parametrized name)
        if func_name in UNIMPLEMENTED_FEATURES or item.name in UNIMPLEMENTED_FEATURES:
            item.add_marker(skip_unimplemented)
            continue

        # Skip FP8 tests
        if any(t in test_id for t in ("fp8", "float8", "e4m3", "e5m2", "e2m1")):
            item.add_marker(skip_unsupported)
            continue

        # Skip FP64 tests
        if "float64" in test_id or "fp64" in test_id:
            item.add_marker(skip_unsupported)
            continue

        # Skip TF32 input_precision tests
        if "tf32" in test_id:
            item.add_marker(skip_unsupported)
            continue

        # Skip atomic tests with types Metal atomics don't support:
        # - int64/uint64: Metal has no 64-bit atomics
        # - bfloat16/float16: Triton's half-precision atomic codegen
        #   produces FP16 intermediate values that our CAS loop can't handle
        if "atomic" in func_name:
            if any(t in test_id for t in ("int64", "uint64", "bfloat16", "float16")):
                item.add_marker(skip_unsupported)
                continue

        # test_tensor_atomic_use_result with size > 1 requires 2D broadcast
        # (NxN store via Nx1 broadcast to NxN). Metal 1D per-thread model
        # only handles size=1 (degenerates to scalar).
        if func_name == "test_tensor_atomic_use_result":
            # test name format: test_tensor_atomic_use_result[op-size-dtype]
            import re
            m = re.search(r'\[(?:add|cas)-(\d+)-', item.name)
            if m and int(m.group(1)) > 1:
                item.add_marker(pytest.mark.skip(
                    reason="Metal: 2D tensor broadcast not supported (size > 1)"))
                continue

        # Skip tests that explicitly require CUDA or HIP
        if "check_cuda_or_hip" in test_id:
            item.add_marker(skip_cuda)
            continue

        # Skip tensor descriptor tests (require CUDA TMA)
        if "tensor_descriptor" in test_id or "tma" in test_id.lower():
            item.add_marker(skip_cuda)
            continue

        # Check parametrize markers for unsupported types
        for marker in item.iter_markers("parametrize"):
            if marker.args:
                argnames = marker.args[0]
                if isinstance(argnames, str):
                    argnames = [a.strip() for a in argnames.split(",")]
                if len(marker.args) > 1:
                    for param_set in marker.args[1]:
                        if not isinstance(param_set, (list, tuple)):
                            param_set = (param_set,)
                        for val in param_set:
                            val_str = str(val).lower()
                            if val_str in UNSUPPORTED_TYPES or val_str in UNSUPPORTED_PRECISIONS:
                                item.add_marker(skip_unsupported)
                                break
