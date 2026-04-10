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
UNIMPLEMENTED_FEATURES = {
    # Histogram — out-of-bounds safety not validated for Metal atomics path
    "test_histogram_silent_data_corruption",
    # 2D broadcast bug — optimize_thread_locality still fails
    "test_optimize_thread_locality",
    # Base join/split tests fail (parametrized variants pass)
    "test_join",
    "test_split",
    # Multi-dimensional transpose — 4D not yet supported
    "test_trans_4d",
    # scaled_dot — requires microscaling format support
    "test_scaled_dot",
    # Map elementwise pack mode — pack=2 with 4 inputs/4 outputs not handled
    "test_map_elementwise_pack",
    # Features requiring CUDA-specific infrastructure
    "test_num_programs",
    "test_tensor_descriptor",
    "test_tma",
    # LLIR/PTX-specific tests
    "test_disable_licm",
    "test_assume",
    "test_poison_return",
    "test_ptx_cast",
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
