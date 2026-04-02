"""GPU correctness tests for the generic lowerer.

Each test compiles a @triton.jit kernel through the full Metal pipeline,
dispatches on GPU, reads back results, and compares against a torch/numpy
reference. These are the numerical counterparts to the compile-only tests
in test_generic_lowerer.py.
"""

import pytest
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import Metal
    HAS_METAL = Metal.MTLCreateSystemDefaultDevice() is not None
except ImportError:
    HAS_METAL = False

requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
requires_metal = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


# ---------------------------------------------------------------------------
# GPU correctness tests — kernels that only had compile-only validation
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_gpu_constant_mul():
    """x * 0.5 — validates arith.constant dense float lowering."""
    @triton.jit
    def const_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        result = x * 0.5
        tl.store(out_ptr + offsets, result, mask=mask)

    n = 1024
    x = torch.randn(n)
    out = torch.zeros(n)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    const_kernel[grid](x, out, n, BLOCK_SIZE=256)

    expected = x * 0.5
    assert torch.allclose(out, expected, atol=1e-5), \
        f"Max diff: {(out - expected).abs().max()}"


@requires_triton
@requires_metal
def test_gpu_exp():
    """tl.exp(x) — validates math.exp lowering."""
    @triton.jit
    def exp_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, tl.exp(x), mask=mask)

    n = 1024
    # Clamp inputs to avoid exp overflow
    x = torch.randn(n).clamp(-10, 10)
    out = torch.zeros(n)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    exp_kernel[grid](x, out, n, BLOCK_SIZE=256)

    expected = torch.exp(x)
    assert torch.allclose(out, expected, atol=1e-4), \
        f"Max diff: {(out - expected).abs().max()}"


@requires_triton
@requires_metal
def test_gpu_multi_op_chain():
    """(a * alpha + b) * (a - b) — validates long expression chains."""
    @triton.jit
    def chain_kernel(a_ptr, b_ptr, out_ptr, n, alpha,
                     BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        result = (a * alpha + b) * (a - b)
        tl.store(out_ptr + offsets, result, mask=mask)

    n = 1024
    a = torch.randn(n)
    b = torch.randn(n)
    out = torch.zeros(n)
    alpha = 2.5

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    chain_kernel[grid](a, b, out, n, alpha, BLOCK_SIZE=256)

    expected = (a * alpha + b) * (a - b)
    assert torch.allclose(out, expected, atol=1e-4), \
        f"Max diff: {(out - expected).abs().max()}"


@requires_triton
@requires_metal
def test_gpu_reduce_expr():
    """tl.sum(x * y + z) — validates reduction of fused expression."""
    @triton.jit
    def reduce_expr_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n,
                           BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
        expr = x * y + z
        result = tl.sum(expr, axis=0)
        tl.store(out_ptr + pid, result)

    n = 256
    x = torch.randn(n)
    y = torch.randn(n)
    z = torch.randn(n)
    out = torch.zeros(1)

    reduce_expr_kernel[(1,)](x, y, z, out, n, BLOCK_SIZE=256)

    expected = (x * y + z).sum()
    assert torch.allclose(out, expected.unsqueeze(0), atol=1e-3), \
        f"Reduce expr: got {out.item()}, expected {expected.item()}"


@requires_triton
@requires_metal
def test_gpu_int_to_float():
    """idx.to(float32) * 0.1 — validates arith.sitofp lowering."""
    @triton.jit
    def itof_kernel(idx_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        idx = tl.load(idx_ptr + offsets, mask=mask)
        result = idx.to(tl.float32) * 0.1
        tl.store(out_ptr + offsets, result, mask=mask)

    n = 256
    idx = torch.arange(n, dtype=torch.int32)
    out = torch.zeros(n, dtype=torch.float32)

    itof_kernel[(1,)](idx, out, n, BLOCK_SIZE=256)

    expected = idx.float() * 0.1
    assert torch.allclose(out, expected, atol=1e-5), \
        f"Max diff: {(out - expected).abs().max()}"


@requires_triton
@requires_metal
def test_gpu_gelu_sigmoid():
    """x * sigmoid(1.702 * x) — validates sigmoid GELU approximation."""
    @triton.jit
    def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        result = x * tl.sigmoid(1.702 * x)
        tl.store(out_ptr + offsets, result, mask=mask)

    n = 1024
    x = torch.randn(n)
    out = torch.zeros(n)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    gelu_kernel[grid](x, out, n, BLOCK_SIZE=256)

    expected = x * torch.sigmoid(1.702 * x)
    assert torch.allclose(out, expected, atol=1e-4), \
        f"Max diff: {(out - expected).abs().max()}"
