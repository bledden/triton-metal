"""make_int4_gemv: the fast weight-only INT4 decode GEMV kernel is correct.

Direct compile_shader test of the landed kernel (the @jit auto-routing for int4 is a
separate, larger piece — per-group scales + nibble-unpack detection). Guards the
kernel MSL against regressions so it is ready to wire.
"""

import math

import pytest

try:
    import torch
    import Metal

    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    from triton_msl.codegen._msl_templates import make_int4_gemv

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None and CompileShaderRuntime().available()
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + compile_shader needed")

GROUP = 128


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
