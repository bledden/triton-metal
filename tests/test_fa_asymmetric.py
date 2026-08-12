"""Asymmetric-head_dim FlashAttention (MLA / DeepSeek-style core: qk_head_dim != v_head_dim).

The simd FA kernel is parameterized so the QK CONTRACTION runs over head_dim (Dqk) while
the OUTPUT / V tiling runs over v_head_dim (Dv) -- registers scale with Dv only, so MLA's
qk=192 / v=128 has the SAME footprint as hd128. `v_head_dim=None` is symmetric (byte-identical
to before). This guards the kernel capability directly via compile_shader; the @jit auto-routing
for asymmetric attention (separate qk/v head dims in the detector) is a separate follow-on.
"""
import pytest

try:
    import torch
    import torch.nn.functional as F
    import Metal

    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime
    from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup as _mk

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None and CompileShaderRuntime().available()
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + compile_shader needed")


def _run(rt, Dqk, Dv, dt, causal, Z=1, H=8, N=512):
    elem = "fp16" if dt == torch.float16 else "fp32"
    lib = rt.get_library(_mk(Dqk, 32, 64, causal, elem, kernel_name="mla_fa", v_head_dim=Dv))
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, Dqk, device="mps", dtype=dt)
    k = torch.randn(Z, H, N, Dqk, device="mps", dtype=dt)
    v = torch.randn(Z, H, N, Dv, device="mps", dtype=dt)
    o = torch.empty(Z, H, N, Dv, device="mps", dtype=dt)
    nqb = N // 32
    args = ([q, k, v, o]
            + [q.stride(i) for i in range(4)] + [k.stride(i) for i in range(4)]
            + [v.stride(i) for i in range(4)] + [o.stride(i) for i in range(4)] + [Z, H, N])
    rt.dispatch(lib, "mla_fa", args, threads=(nqb * 256, Z * H, 1), group_size=(256, 1, 1))
    torch.mps.synchronize()
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=1.0 / (Dqk ** 0.5))
    return (o.float() - ref.float()).abs().max().item()


@requires
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("Dqk,Dv", [(192, 128), (128, 64), (192, 64)])
def test_mla_core_asymmetric_correct(Dqk, Dv, causal):
    """qk_head_dim != v_head_dim matches SDPA (the MLA attention core), fp16."""
    rt = CompileShaderRuntime()
    err = _run(rt, Dqk, Dv, torch.float16, causal)
    assert err < 5e-3, f"asymmetric FA qk={Dqk} v={Dv} causal={causal}: err {err}"


@requires
@pytest.mark.parametrize("D", [64, 128])
def test_v_head_dim_none_is_symmetric(D):
    """v_head_dim=None (symmetric) must be byte-identical to passing v_head_dim=D."""
    a = _mk(D, 32, 64, False, "fp16", kernel_name="k")
    b = _mk(D, 32, 64, False, "fp16", kernel_name="k", v_head_dim=D)
    assert a == b, "v_head_dim=D must reproduce the symmetric kernel exactly"


@requires
def test_symmetric_still_matches_sdpa():
    """Regression: the default symmetric path is unchanged and correct."""
    rt = CompileShaderRuntime()
    assert _run(rt, 128, 128, torch.float16, False) < 5e-3
    assert _run(rt, 64, 64, torch.float32, True) < 1e-2
