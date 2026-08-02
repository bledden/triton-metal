"""Basic Gluon (Triton's lower-level, explicit-layout language) on Metal.

MetalBackend.get_target_name + populating metadata['shared'] in gluon_to_ttgir
let simple Gluon kernels — explicit BlockedLayout, load/store, elementwise —
compile and run correctly on Metal. This deliberately does NOT cover the
NVIDIA-specific Gluon surface (mma, warp specialization, mbarrier, TMA), which
is out of scope for Metal.
"""

import pytest

try:
    import torch
    import triton
    import Metal
    from triton.experimental import gluon
    from triton.experimental.gluon import language as ttgl

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
    TPW = triton.runtime.driver.active.get_current_target().warp_size
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + Gluon needed")

if HAS:

    @gluon.jit
    def _gluon_copy(Out, In, numel, XBLOCK: ttgl.constexpr, layout: ttgl.constexpr):
        xoff = ttgl.program_id(0) * XBLOCK + ttgl.arange(0, XBLOCK, layout=layout)
        m = xoff < numel
        ttgl.store(Out + xoff, ttgl.load(In + xoff, m), m)

    @gluon.jit
    def _gluon_add(A, B, Out, numel, XBLOCK: ttgl.constexpr, layout: ttgl.constexpr):
        xoff = ttgl.program_id(0) * XBLOCK + ttgl.arange(0, XBLOCK, layout=layout)
        m = xoff < numel
        ttgl.store(Out + xoff, ttgl.load(A + xoff, m) + ttgl.load(B + xoff, m), m)

    def _blocked(spt, warps):
        return ttgl.BlockedLayout(size_per_thread=[spt], threads_per_warp=[TPW], warps_per_cta=[warps], order=[0])


@requires
@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("spt", [1, 2, 4, 8])
def test_gluon_copy_blocked_layout(spt, warps):
    """A Gluon copy across blocked layouts (incl. multi-element-per-thread)."""
    layout = _blocked(spt, warps)
    XBLOCK = 256
    inp = torch.randn(XBLOCK * 4 - 7, device="mps")
    out = torch.empty_like(inp)
    _gluon_copy[(4,)](out, inp, inp.numel(), XBLOCK, layout, num_warps=warps)
    torch.testing.assert_close(out, inp)


@requires
def test_gluon_elementwise_add():
    """A two-input elementwise Gluon kernel."""
    layout = _blocked(4, 4)
    XBLOCK = 256
    a = torch.randn(XBLOCK * 4 - 7, device="mps")
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    _gluon_add[(4,)](a, b, out, a.numel(), XBLOCK, layout, num_warps=4)
    torch.testing.assert_close(out, a + b)
