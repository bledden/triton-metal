"""Thermal-robust benchmark of the quantized fast paths vs the honest torch baseline.

The quantized win prop: dequant + matmul in ONE kernel that reads the int8 weight
directly (half/quarter the bytes of fp32), vs the naive path of materializing an
fp32 weight (`(w.float() - z) * s`) and calling torch matmul. Decode (GEMV) is
memory-bound, so reading int8 instead of fp32 is the win.

Thermal robustness: the M4 Max clock-ramps, so median-of-N drifts (a kernel run
"second" is systematically faster). We ALTERNATE msl/torch every iteration and
average each, so both see the same thermal envelope — the ratio is drift-cancelled.

Run:  python benchmarks/bench_quantized.py
"""

import time

import torch
import triton
import triton.language as tl


# ---- @triton.jit kernels (routed to the fast quantized paths on Metal) ----
@triton.jit
def _gemv(x, w, o, s, z, N, K, swn, swk, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    sc = tl.load(s + on)
    ze = tl.load(z + on)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, K, BK):
        xv = tl.load(x + ok + k)
        wv = tl.load(w + on[:, None] * swn + (ok[None, :] + k) * swk).to(tl.float32)
        acc += tl.sum(xv[None, :] * (wv - ze[:, None]), axis=1)
    tl.store(o + on, acc * sc)  # per-N scale as epilogue (the routable canonical form)


@triton.jit
def _gemm_kn(a, w, c, s, z, M, N, K, sam, sak, swk, swn, scm, scn,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = a + om[:, None] * sam + ok[None, :] * sak
    wp = w + ok[:, None] * swk + on[None, :] * swn
    sc = tl.load(s + on)
    ze = tl.load(z + on)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        av = tl.load(ap)
        wv = (tl.load(wp).to(tl.float32) - ze[None, :]) * sc[None, :]
        acc += tl.dot(av, wv)
        ap += BK * sak
        wp += BK * swk
    tl.store(c + om[:, None] * scm + on[None, :] * scn, acc)


def _time(fn, iters):
    fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def _alternating(msl_fn, torch_fn, rounds=6, iters=30):
    """Interleave msl/torch each round so both see the same thermal envelope."""
    mt, tt = [], []
    for _ in range(rounds):
        mt.append(_time(msl_fn, iters))
        tt.append(_time(torch_fn, iters))
    mt.sort()
    tt.sort()
    return mt[len(mt) // 2], tt[len(tt) // 2]  # median across rounds


def bench_gemv(N, K):
    BN, BK = 32, 32
    torch.manual_seed(0)
    x = torch.randn(K, device="mps")
    w = torch.randint(-127, 127, (N, K), device="mps", dtype=torch.int8)
    s = torch.rand(N, device="mps") * 0.02 + 0.005
    z = torch.randint(-8, 8, (N,), device="mps").float()
    o = torch.zeros(N, device="mps")
    grid = (triton.cdiv(N, BN),)

    wf = ((w.float() - z[:, None]) * s[:, None]).contiguous()  # pre-materialized fp32 weight

    def msl():
        _gemv[grid](x, w, o, s, z, N, K, w.stride(0), w.stride(1), BN=BN, BK=BK)

    def torch_fp32():
        # honest deploy baseline: fp32 weight ALREADY materialized (cached), matvec only.
        # Isolates the real win: int8 reads 1/4 the weight bytes (memory-bound decode).
        return wf @ x

    msl()
    torch.mps.synchronize()
    ok = torch.allclose(o, wf @ x, rtol=2e-3, atol=2e-3)
    m, t = _alternating(msl, torch_fp32)
    msl_gbs = (N * K) / m / 1e9
    return ok, m, t, msl_gbs


def bench_gemm(N, K, M=256):
    BM = BN = BK = 32
    torch.manual_seed(0)
    a = torch.randn(M, K, device="mps")
    w = torch.randint(-127, 127, (K, N), device="mps", dtype=torch.int8)
    s = torch.rand(N, device="mps") * 0.02 + 0.005
    z = torch.randint(-8, 8, (N,), device="mps").float()
    c = torch.zeros(M, N, device="mps")
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    args = (a, w, c, s, z, M, N, K, a.stride(0), a.stride(1), w.stride(0), w.stride(1), c.stride(0), c.stride(1))

    wf = ((w.float() - z[None, :]) * s[None, :]).contiguous()  # pre-materialized fp32 weight

    def msl():
        _gemm_kn[grid](*args, BM=BM, BN=BN, BK=BK)

    def torch_fp32():
        # honest deploy baseline: fp32 weight cached, MPS BLAS matmul only.
        return a @ wf

    msl()
    torch.mps.synchronize()
    ok = torch.allclose(c, a @ wf, rtol=2e-3, atol=2e-3)
    m, t = _alternating(msl, torch_fp32)
    tf = 2 * M * N * K / m / 1e12
    return ok, m, t, tf


def main():
    print("== int8 decode GEMV (M=1) — alternating-A/B vs torch fp32 matvec (cached weight) ==")
    print("   (the honest win: int8 reads 1/4 the weight bytes; decode is memory-bound)")
    for (N, K) in [(4096, 4096), (11008, 4096), (32000, 4096)]:
        ok, m, t, gbs = bench_gemv(N, K)
        print(f"  N={N:6d} K={K}: correct={ok}  msl {m*1e6:7.1f}us ({gbs:5.0f} GB/s int8)  torch-fp32 {t*1e6:7.1f}us  speedup {t/m:.2f}x")
    print("== int8 prefill GEMM (M=256) — alternating-A/B vs torch fp32 matmul (cached weight) ==")
    print("   (prefill is compute-bound; MPS BLAS fp32 is highly tuned — int8's value here is footprint, not speed)")
    for (N, K) in [(4096, 4096), (2048, 2048)]:
        ok, m, t, tf = bench_gemm(N, K)
        print(f"  N={N:6d} K={K}: correct={ok}  msl {m*1e6:7.1f}us ({tf:5.2f} TF)  torch-fp32 {t*1e6:7.1f}us  speedup {t/m:.2f}x")


if __name__ == "__main__":
    main()
