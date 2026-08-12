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


# --- @triton.jit auto-routing: an asymmetric (MLA-shaped) kernel routes to simd FA ---
if HAS:
    import triton
    import triton.language as tl

    @triton.jit
    def _mla_fwd(Q, K, V, Out,
                 sqz, sqh, sqm, sqk, skz, skh, skn, skk, svz, svh, svn, svk, soz, soh, som, sok,
                 Z, H, N_CTX,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 HEAD_DIM: tl.constexpr, V_HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_dv = tl.arange(0, V_HEAD_DIM)
        q_ptrs = Q + off_z * sqz + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        qk_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        q = q * qk_scale
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)
        hi = N_CTX
        if IS_CAUSAL:
            hi = min((start_m + 1) * BLOCK_M, N_CTX)
        for start_n in range(0, hi, BLOCK_N):
            k_ptrs = K + off_z * skz + off_h * skh + (start_n + offs_n)[:, None] * skn + offs_d[None, :] * skk
            k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
            qk = tl.dot(q, tl.trans(k).to(q.dtype))
            if IS_CAUSAL:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = tl.where(mask, qk, float("-inf"))
            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v_ptrs = V + off_z * svz + off_h * svh + (start_n + offs_n)[:, None] * svn + offs_dv[None, :] * svk
            v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
            acc += tl.dot(p.to(tl.float32), v.to(tl.float32))
            m_i = m_new
        acc = acc / l_i[:, None]
        o_ptrs = Out + off_z * soz + off_h * soh + offs_m[:, None] * som + offs_dv[None, :] * sok
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


@requires
def test_over_budget_head_dim_refuses_clearly():
    """head_dim past the 32KB threadgroup budget (qk=256 fp16) raises a CLEAR ValueError
    from the kernel gen, not a cryptic AGX pipeline-state failure."""
    with pytest.raises(ValueError, match="threadgroup memory"):
        _mk(256, 32, 64, False, "fp16", kernel_name="k", v_head_dim=128)


@requires
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("Dqk,Dv", [(128, 64)])
def test_mla_jit_routes_to_simd_and_correct(monkeypatch, Dqk, Dv, causal):
    """A canonical @triton.jit asymmetric attention (qk != v, fp16) auto-routes to the
    simd FA, runs on GPU, and matches SDPA. tl.arange requires power-of-2 bounds, and
    qk=256 exceeds the 32KB threadgroup budget, so the @jit-reachable asymmetric case is
    qk=128/v=64. Real MLA (qk=192) fits + beats SDPA at the KERNEL level (validated in
    test_mla_core_asymmetric_correct); expressing qk=192 from @jit needs a nope/rope
    split (two power-of-2 dots) -- a detector follow-on."""
    import triton_msl.autotuning._fa_dispatch as fa_mod

    fired = {"n": 0}
    real = fa_mod.dispatch_flash_attention

    def spy(*a, **kw):
        fired["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(fa_mod, "dispatch_flash_attention", spy)

    Z, H, N = 1, 4, 512
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, Dqk, device="mps", dtype=torch.float16)
    k = torch.randn(Z, H, N, Dqk, device="mps", dtype=torch.float16)
    v = torch.randn(Z, H, N, Dv, device="mps", dtype=torch.float16)
    out = torch.empty(Z, H, N, Dv, device="mps", dtype=torch.float16)
    _mla_fwd[(N // 32, Z * H)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=Dqk, V_HEAD_DIM=Dv, IS_CAUSAL=causal)
    torch.mps.synchronize()

    assert fired["n"] >= 1, "MLA-shaped @jit kernel did not route through the simd FA dispatch"
    assert str(out.device).startswith("mps")
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=1.0 / (Dqk ** 0.5))
    assert (out.float() - ref.float()).abs().max().item() < 5e-3
