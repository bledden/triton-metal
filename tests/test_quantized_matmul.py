"""Quantized (weight-only int8) matmul: routed to the fast dequant kernel, or refused.

A canonical weight-only int8 matmul — ``out = a @ ((w_i8.to(f32) - zero) * scale)`` with
the standard ``(input, weight, output, scale, zero, M, N, K, ...strides)`` signature,
weight stored [K,N] contiguous, per-N float scale/zero, fp32 in/out — is recognized at
compile time (``_maybe_quant_matmul_descriptor``) and dispatched to
``make_int8_matmul_fast(layout='kn')`` via compile_shader. The simdgroup MMA runs on the
dequantized tile (near-bit-exact vs the float reference).

Correct-or-refuse everywhere else: a shape the edge-free fast kernel can't tile
(M % 32, N % 16, or K % 32 nonzero) is refused loudly, NOT silently mis-tiled; a plain
fp32 matmul with the same 2-D-grid + extra-ptr-arg structure is NOT misdetected.
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

    @triton.jit
    def _int8_gemv(x_ptr, w_ptr, o_ptr, scale_ptr, zero_ptr, N, K, swn, swk,
                   BN: tl.constexpr, BK: tl.constexpr):
        # GPTQ-style weight-only int8 decode GEMV: weight [N,K], per-N scale/zero.
        pid = tl.program_id(0)
        offs_n = pid * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        scale = tl.load(scale_ptr + offs_n)
        zero = tl.load(zero_ptr + offs_n)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k in range(0, K, BK):
            x = tl.load(x_ptr + offs_k + k)
            w = tl.load(w_ptr + offs_n[:, None] * swn + (offs_k[None, :] + k) * swk).to(tl.float32)
            acc += tl.sum(x[None, :] * (w - zero[:, None]), axis=1)
        tl.store(o_ptr + offs_n, acc * scale)

    @triton.jit
    def _fp32_gemv(x_ptr, w_ptr, o_ptr, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        offs_n = pid * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k in range(0, K, BK):
            x = tl.load(x_ptr + offs_k + k)
            w = tl.load(w_ptr + offs_n[:, None] * swn + (offs_k[None, :] + k) * swk)
            acc += tl.sum(x[None, :] * w, axis=1)
        tl.store(o_ptr + offs_n, acc)

    def _run_quant(M, N, K, BM=32, BN=32, BK=32):
        a = torch.randn(M, K, device="mps")
        w = torch.randint(-127, 127, (K, N), device="mps", dtype=torch.int8)
        scale = torch.rand(N, device="mps") * 0.02 + 0.005
        zero = torch.randint(-8, 8, (N,), device="mps").float()
        c = torch.zeros(M, N, device="mps")
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
        ref = a @ ((w.float() - zero[None, :]) * scale[None, :])
        return c, ref


@requires
@pytest.mark.parametrize("M,N,K", [(64, 128, 128), (256, 256, 256), (512, 512, 512)])
def test_quantized_matmul_runs_correctly(M, N, K):
    torch.manual_seed(0)
    c, ref = _run_quant(M, N, K)
    torch.testing.assert_close(c, ref, rtol=2e-3, atol=2e-3)


@requires
def test_quantized_matmul_unaligned_refuses():
    # M = 48 is not a multiple of the row tile (32); the edge-free fast kernel
    # cannot tile it, so the driver refuses rather than silently mis-tiling.
    torch.manual_seed(0)
    with pytest.raises(MetalNonRecoverableError, match="quantized"):
        _run_quant(48, 128, 128)


@requires
@pytest.mark.parametrize("N,K", [(128, 256), (256, 128), (512, 512)])
def test_quantized_gemv_runs_correctly(N, K):
    # Weight-only int8 decode GEMV routed to make_int8_gemv.
    torch.manual_seed(0)
    BN, BK = 32, 32
    x = torch.randn(K, device="mps")
    w = torch.randint(-127, 127, (N, K), device="mps", dtype=torch.int8)
    scale = torch.rand(N, device="mps") * 0.02 + 0.005
    zero = torch.randint(-8, 8, (N,), device="mps").float()
    o = torch.zeros(N, device="mps")
    _int8_gemv[(triton.cdiv(N, BN),)](x, w, o, scale, zero, N, K, w.stride(0), w.stride(1), BN=BN, BK=BK)
    ref = ((w.float() - zero[:, None]) * scale[:, None]) @ x
    torch.testing.assert_close(o, ref, rtol=2e-3, atol=2e-3)


@requires
def test_fp32_gemv_not_misrouted():
    # A plain fp32 GEMV (no dequant) must NOT be routed to the int8 GEMV kernel; it
    # hits the loop-carried-reduce guard and refuses (correct-or-refuse).
    torch.manual_seed(0)
    N, K, BN, BK = 128, 256, 32, 32
    x = torch.randn(K, device="mps")
    w = torch.randn(N, K, device="mps")
    o = torch.zeros(N, device="mps")
    with pytest.raises(MetalNonRecoverableError, match="accumulated across a loop"):
        _fp32_gemv[(triton.cdiv(N, BN),)](x, w, o, N, K, w.stride(0), w.stride(1), BN=BN, BK=BK)


@requires
def test_plain_fp32_matmul_not_misdetected():
    torch.manual_seed(0)
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
