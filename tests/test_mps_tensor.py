"""Test MPS tensor integration with Metal Triton kernels.

Validates that Triton kernels can operate on MPS tensors without
corrupting PyTorch's MPS buffer tracking.
"""

import pytest

try:
    import torch
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False

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

requires_all = pytest.mark.skipif(
    not (HAS_TORCH and HAS_MPS and HAS_TRITON and HAS_METAL),
    reason="Requires PyTorch MPS + Triton + Metal"
)


@requires_all
def test_mps_vector_add():
    """Triton vector_add on MPS tensors produces correct results."""
    @triton.jit
    def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    n = 1024
    a = torch.randn(n, device="mps", dtype=torch.float32)
    b = torch.randn(n, device="mps", dtype=torch.float32)
    out = torch.empty(n, device="mps", dtype=torch.float32)

    grid = (n // 256,)
    add_kernel[grid](a, b, out, n, BLOCK_SIZE=256)

    # Verify results
    expected = a.cpu() + b.cpu()
    actual = out.cpu()
    assert torch.allclose(actual, expected, atol=1e-5), \
        f"Max diff: {(actual - expected).abs().max()}"

    # Verify MPS still works on the output tensor
    torch.mps.synchronize()
    mps_result = out * 2.0
    torch.mps.synchronize()
    expected_2x = expected * 2.0
    assert torch.allclose(mps_result.cpu(), expected_2x, atol=1e-5)
    print("MPS vector_add: PASS")


@requires_all
def test_mps_tensor_survives_triton():
    """MPS tensors remain usable after Triton kernel modifies them."""
    @triton.jit
    def scale_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * 2.0, mask=mask)

    n = 512
    x = torch.ones(n, device="mps", dtype=torch.float32)
    out = torch.empty(n, device="mps", dtype=torch.float32)

    grid = (n // 256,)
    scale_kernel[grid](x, out, n, BLOCK_SIZE=256)

    # out should be 2.0 everywhere
    torch.mps.synchronize()
    assert torch.allclose(out.cpu(), torch.full((n,), 2.0)), \
        f"Expected 2.0, got: {out[:8].tolist()}"

    # Chain multiple MPS operations on the result
    r1 = out + 1.0          # 3.0
    r2 = r1 * out           # 6.0
    r3 = torch.sum(r2)      # 6.0 * 512 = 3072.0
    torch.mps.synchronize()
    assert abs(r3.item() - 3072.0) < 1e-2, f"Expected 3072.0, got {r3.item()}"
    print("MPS tensor survival: PASS")


@requires_all
def test_mps_input_not_corrupted():
    """Triton kernel doesn't corrupt input MPS tensors."""
    @triton.jit
    def add_one_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + 1.0, mask=mask)

    n = 256
    x = torch.arange(n, device="mps", dtype=torch.float32)
    original_x = x.cpu().clone()
    out = torch.empty(n, device="mps", dtype=torch.float32)

    grid = (1,)
    add_one_kernel[grid](x, out, n, BLOCK_SIZE=256)

    # Input should be unchanged
    torch.mps.synchronize()
    assert torch.equal(x.cpu(), original_x), "Input tensor was corrupted!"

    # Output should be input + 1
    expected = original_x + 1.0
    assert torch.allclose(out.cpu(), expected, atol=1e-5)
    print("MPS input preservation: PASS")
