"""Quantized (int8 weight-only) matmul: recognized + refused cleanly.

The simdgroup matmul template loads B directly from its pointer as float/half, so a
dequantized int8 weight cannot be computed there — it is detected and refused loudly
(never a cryptic `simdgroup_load(float, char*)` compile error, never a silent raw-int
read with no dequant). A plain fp32 matmul with the SAME 2-D-grid + extra-ptr-arg
structure must NOT be misdetected (guards the quantized detector + the matmul
role-resolution fix for kernels with extra ptr args like scale/zero).
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
    def _int8_wo_matmul(
        a_ptr,
        w_ptr,
        c_ptr,
        scale_ptr,
        zero_ptr,
        M,
        N,
        K,
        sam,
        sak,
        swk,
        swn,
        scm,
        scn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
        w_ptrs = w_ptr + offs_k[:, None] * swk + offs_n[None, :] * swn
        scale = tl.load(scale_ptr + offs_n)
        zero = tl.load(zero_ptr + offs_n)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs)
            w = (tl.load(w_ptrs).to(tl.float32) - zero[None, :]) * scale[None, :]
            acc += tl.dot(a, w)
            a_ptrs += BK * sak
            w_ptrs += BK * swk
        c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
        tl.store(c_ptrs, acc)

    @triton.jit
    def _fp32_matmul(
        a_ptr,
        w_ptr,
        c_ptr,
        M,
        N,
        K,
        sam,
        sak,
        swk,
        swn,
        scm,
        scn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
        w_ptrs = w_ptr + offs_k[:, None] * swk + offs_n[None, :] * swn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(a_ptrs), tl.load(w_ptrs))
            a_ptrs += BK * sak
            w_ptrs += BK * swk
        c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
        tl.store(c_ptrs, acc)


@requires
def test_quantized_matmul_refuses_cleanly():
    M, N, K, BM, BN, BK = 64, 128, 128, 32, 32, 32
    a = torch.randn(M, K, device="mps")
    w = torch.randint(-127, 127, (K, N), device="mps", dtype=torch.int8)
    scale = torch.rand(N, device="mps") * 0.02 + 0.005
    zero = torch.randint(-8, 8, (N,), device="mps").float()
    c = torch.zeros(M, N, device="mps")
    with pytest.raises(MetalNonRecoverableError, match="quantized matmul"):
        _int8_wo_matmul[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
            a,
            w,
            c,
            scale,
            zero,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            w.stride(0),
            w.stride(1),
            c.stride(0),
            c.stride(1),
            BM=BM,
            BN=BN,
            BK=BK,
        )


@requires
def test_plain_fp32_matmul_not_misdetected():
    M, N, K, BM, BN, BK = 64, 128, 128, 32, 32, 32
    a = torch.randn(M, K, device="mps")
    w = torch.randn(K, N, device="mps")
    c = torch.zeros(M, N, device="mps")
    _fp32_matmul[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a,
        w,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        c.stride(0),
        c.stride(1),
        BM=BM,
        BN=BN,
        BK=BK,
    )
    torch.testing.assert_close(c, a @ w, rtol=1e-2, atol=1e-2)
