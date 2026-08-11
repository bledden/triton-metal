"""In-loop 2-D axis reduce (GEMV-via-tl.sum): correct-or-refuse.

A 2-D→1-D axis reduce (`tl.sum(x, axis=1)`) produces its result in the
row-broadcast layout (thread ``lid`` holds row ``lid/N``). When that result is
accumulated across a K-loop, the loop-carried accumulator + 1-D store instead
assume one-row-per-thread and silently collapse every output row to the first.
Until loop-carry layout propagation lands, this is refused (not mis-computed).

The single-tile form (no K-loop) is correct and must NOT be over-refused.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal

    from triton_msl.errors import MetalNonRecoverableError

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + triton needed")

if HAS:

    @triton.jit
    def _gemv_loop(x_ptr, w_ptr, o_ptr, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for _ in range(0, K, BK):
            x = tl.load(x_ptr + offs_k)
            w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk)
            acc += tl.sum(x[None, :] * w, axis=1)
            offs_k += BK
        tl.store(o_ptr + offs_n, acc)

    @triton.jit
    def _gemv_single(x_ptr, w_ptr, o_ptr, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        x = tl.load(x_ptr + offs_k)
        w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk)
        tl.store(o_ptr + offs_n, tl.sum(x[None, :] * w, axis=1))


@requires
def test_inloop_2d_axis_reduce_refuses():
    # K > BK forces the K-loop that carries the reduce result -> refuse, not
    # silently collapse every output row to row 0.
    torch.manual_seed(0)
    N, K, BK = 32, 64, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    with pytest.raises(MetalNonRecoverableError, match="accumulated across a loop"):
        _gemv_loop[(1,)](x, w, o, K, w.stride(0), w.stride(1), BN=N, BK=BK)


@requires
def test_single_tile_2d_axis_reduce_still_correct():
    # No K-loop (BK == K): the reduce result is consumed directly, correctly.
    torch.manual_seed(0)
    N, K = 32, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    _gemv_single[(1,)](x, w, o, w.stride(0), w.stride(1), BN=N, BK=K)
    torch.mps.synchronize()
    torch.testing.assert_close(o, w @ x, rtol=1e-3, atol=1e-3)
