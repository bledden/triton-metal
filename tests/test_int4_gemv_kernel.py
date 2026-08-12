"""make_int4_gemv: the fast weight-only INT4 decode GEMV kernel is correct.

Direct compile_shader test of the landed kernel (the @jit auto-routing for int4 is a
separate, larger piece — per-group scales + nibble-unpack detection). Guards the
kernel MSL against regressions so it is ready to wire.
"""

import math

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal

    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    from triton_msl.codegen._msl_templates import make_int4_gemv

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None and CompileShaderRuntime().available()
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + compile_shader needed")

GROUP = 128

if HAS:

    @triton.jit
    def _int4_gemv_jit(x_ptr, w_ptr, o_ptr, s_ptr, z_ptr, N, K, ng, swn, ssn,
                       BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
        pid = tl.program_id(0)
        on = pid * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k in range(0, K, BK):
            kk = k + ok
            packed = tl.load(w_ptr + on[:, None] * swn + (kk // 2)[None, :])
            w4 = (packed >> ((kk % 2) * 4)[None, :]) & 0xF
            g = kk // G
            s = tl.load(s_ptr + on[:, None] * ssn + g[None, :])
            z = tl.load(z_ptr + on[:, None] * ssn + g[None, :])
            acc += tl.sum(tl.load(x_ptr + kk)[None, :] * ((w4.to(tl.float32) - z) * s), axis=1)
        tl.store(o_ptr + on, acc)


@requires
@pytest.mark.parametrize("N,K", [(128, 256), (256, 512), (512, 4096)])
def test_int4_gemv_correct(N, K):
    rt = CompileShaderRuntime()
    ng = (K + GROUP - 1) // GROUP
    torch.manual_seed(0)
    x = torch.randn(K, device="mps", dtype=torch.float32)
    w4 = torch.randint(0, 16, (N, K), device="mps", dtype=torch.int32)
    packed = (w4[:, 0::2] | (w4[:, 1::2] << 4)).to(torch.uint8).contiguous()  # [N, K/2]
    scale = (torch.rand(N, ng, device="mps") * 0.02 + 0.005).contiguous()
    zero = torch.randint(0, 16, (N, ng), device="mps").float().contiguous()
    out = torch.zeros(N, device="mps", dtype=torch.float32)

    lib = rt.get_library(make_int4_gemv(GROUP))
    group = 256
    threads = math.ceil((N * 32) / group) * group
    rt.dispatch(lib, "int4_gemv", [x, packed, out, scale, zero, K, N], threads=threads, group_size=group)
    torch.mps.synchronize()

    gidx = torch.arange(K, device="mps") // GROUP
    ref = ((w4.float() - zero[:, gidx]) * scale[:, gidx]) @ x
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


@requires
@pytest.mark.parametrize("N,K,G", [(128, 256, 128), (256, 512, 128), (128, 256, 64)])
def test_int4_gemv_jit_routes_correct(N, K, G):
    # A canonical int4 @triton.jit GEMV auto-routes to make_int4_gemv (correct-or-refuse).
    ng = K // G
    torch.manual_seed(0)
    w4 = torch.randint(0, 16, (N, K), device="mps", dtype=torch.int32)
    packed = (w4[:, 0::2] | (w4[:, 1::2] << 4)).to(torch.uint8).contiguous()
    x = torch.randn(K, device="mps")
    s = (torch.rand(N, ng, device="mps") * 0.02 + 0.005).contiguous()
    z = torch.randint(0, 16, (N, ng), device="mps").float().contiguous()
    o = torch.zeros(N, device="mps")
    _int4_gemv_jit[(triton.cdiv(N, 32),)](x, packed, o, s, z, N, K, ng, packed.stride(0), s.stride(0),
                                          BN=32, BK=64, G=G)
    gi = torch.arange(K, device="mps") // G
    ref = ((w4.float() - z[:, gi]) * s[:, gi]) @ x
    torch.testing.assert_close(o, ref, rtol=2e-3, atol=2e-3)
