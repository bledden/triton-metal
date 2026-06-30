"""Regression anchor for examples/local_triton_dev.py — the kernels in the
'develop Triton locally on Apple Silicon' example must keep matching the CPU
reference (they are real @triton.jit kernels run through the Metal backend)."""

import importlib.util
import os

import pytest

_EX = os.path.join(os.path.dirname(__file__), "..", "examples", "local_triton_dev.py")

try:
    import numpy as np  # noqa: F401
    import torch
    import triton  # noqa: F401

    _HAS = hasattr(torch, "mps") and torch.backends.mps.is_available()
except Exception:  # noqa: BLE001
    _HAS = False

requires = pytest.mark.skipif(not _HAS, reason="needs torch + an Apple MPS GPU")


def _load_example():
    spec = importlib.util.spec_from_file_location("local_triton_dev", _EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@requires
@pytest.mark.parametrize("runner", ["run_vector_add", "run_fused_softmax", "run_matmul"])
def test_local_dev_example_matches_cpu(runner):
    import numpy as np

    np.random.seed(0)
    mod = _load_example()
    name, shape, err, tol = getattr(mod, runner)()
    assert err < tol, f"{name} ({shape}): max|Δ vs NumPy| = {err:.2e} >= tol {tol:.0e}"
