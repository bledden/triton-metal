"""Split-K for skinny/deep fp32 matmul: correct-or-refuse + deterministic.

Skinny/deep shapes (small M/N, deep K) route to the deterministic two-pass split-K
path (partials buffer + reduce, no atomics). This exercises that path — which the
base matmul fuzzer does not reach (its shapes are shallow) — across random data +
odd tile counts, and asserts run-to-run determinism.
"""

import pytest

try:
    import torch
    import triton
    import triton.language as tl
    import Metal

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + triton needed")

if HAS:

    @triton.jit
    def _mm(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        om = pm * BM + tl.arange(0, BM)
        on = pn * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        ap = a + om[:, None] * sam + ok[None, :] * sak
        bp = b + ok[:, None] * sbk + on[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(ap), tl.load(bp))
            ap += BK * sak
            bp += BK * sbk
        tl.store(c + om[:, None] * scm + on[None, :] * scn, acc)

    def _run(M, N, K):
        a = torch.randn(M, K, device="mps")
        b = torch.randn(K, N, device="mps")
        c = torch.zeros(M, N, device="mps")
        grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
        _mm[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                  c.stride(0), c.stride(1), BM=32, BN=32, BK=32)
        return a, b, c


# skinny/deep (routes to split-K) + shapes that must fall through unchanged.
@requires
@pytest.mark.parametrize("M,N,K", [
    (64, 64, 8192), (128, 128, 8192), (32, 32, 4096), (64, 32, 8192),
    (32, 64, 4096), (96, 64, 4096), (256, 256, 4096), (2048, 2048, 2048),
])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matmul_correct(M, N, K, seed):
    torch.manual_seed(seed)
    a, b, c = _run(M, N, K)
    torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)


@requires
def test_splitk_deterministic():
    # The two-pass split-K path must be byte-identical run-to-run (no atomics).
    torch.manual_seed(0)
    _, _, c1 = _run(64, 64, 8192)
    torch.mps.synchronize()
    torch.manual_seed(0)
    _, _, c2 = _run(64, 64, 8192)
    torch.mps.synchronize()
    assert torch.equal(c1, c2)
