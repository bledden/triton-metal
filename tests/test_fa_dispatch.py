"""FlashAttention zero-copy dispatch: the 2-D-grid compile_shader routing.

The simdgroup FA kernel's 2-D grid disqualified it from the generic 1-D fast-path,
so it fell to the ~3x-slower host-roundtrip path. `dispatch_flash_attention` routes
it via compile_shader with its native grid (fail-OPEN: a miss falls through to the
still-correct host path). These guard (1) the dispatch computes the right 2-D launch
and fails open on bad input, and (2) a real compiled FA kernel is actually wired to it.
"""

import pytest

from triton_msl.autotuning._fa_dispatch import dispatch_flash_attention


class _FakeRT:
    def __init__(self, raise_on_dispatch=False):
        self.calls = []
        self.unsupported = set()
        self.raise_on_dispatch = raise_on_dispatch

    def is_unsupported(self, msl):
        return msl in self.unsupported

    def mark_unsupported(self, msl):
        self.unsupported.add(msl)

    def get_library(self, msl):
        return ("lib", msl)

    def dispatch(self, lib, name, kargs, *, threads, group_size):
        if self.raise_on_dispatch:
            raise RuntimeError("boom")
        self.calls.append((lib, name, list(kargs), threads, group_size))


DESC = ("flash_attention", "#include <metal_stdlib>\n// fa msl", 256)
KARGS = ["q", "k", "v", "out"] + list(range(16)) + [1, 8, 1024]  # Q,K,V,Out,16 strides,Z,H,N


def test_dispatch_computes_native_2d_grid():
    rt = _FakeRT()
    ok = dispatch_flash_attention(rt, DESC, "fa_kernel", KARGS, gridX=32, gridY=8, gridZ=1)
    assert ok is True
    assert len(rt.calls) == 1
    lib, name, kargs, threads, group_size = rt.calls[0]
    assert name == "fa_kernel"
    assert kargs == KARGS                 # buffer order passed through verbatim
    assert threads == (32 * 256, 8, 1)    # gridX*tg in x, gridY in y, gridZ in z
    assert group_size == (256, 1, 1)      # tg threads per group, in x only


def test_exit_hook_fires_on_success():
    rt = _FakeRT()
    seen = []
    ok = dispatch_flash_attention(
        rt, DESC, "fa_kernel", KARGS, 32, 8, 1,
        launch_exit_hook=lambda md: seen.append(md), launch_metadata="MD")
    assert ok and seen == ["MD"]


@pytest.mark.parametrize("desc", [
    ("gemv", "msl", 256),                 # wrong tag
    ("flash_attention", None, 256),       # no MSL
    ("flash_attention", "msl"),           # too short
    None,
])
def test_fail_open_on_bad_descriptor(desc):
    rt = _FakeRT()
    assert dispatch_flash_attention(rt, desc, "fa_kernel", KARGS, 32, 8, 1) is False
    assert rt.calls == []


@pytest.mark.parametrize("gx,gy,gz", [(0, 8, 1), (32, 0, 1), (32, 8, 0)])
def test_fail_open_on_degenerate_grid(gx, gy, gz):
    rt = _FakeRT()
    assert dispatch_flash_attention(rt, DESC, "fa_kernel", KARGS, gx, gy, gz) is False
    assert rt.calls == []


def test_fail_open_on_missing_kernel_name():
    rt = _FakeRT()
    assert dispatch_flash_attention(rt, DESC, None, KARGS, 32, 8, 1) is False


def test_dispatch_error_marks_unsupported_and_fails_open():
    rt = _FakeRT(raise_on_dispatch=True)
    assert dispatch_flash_attention(rt, DESC, "fa_kernel", KARGS, 32, 8, 1) is False
    assert DESC[1] in rt.unsupported    # marked so we don't retry a broken shader


def test_skips_already_unsupported_msl():
    rt = _FakeRT()
    rt.mark_unsupported(DESC[1])
    assert dispatch_flash_attention(rt, DESC, "fa_kernel", KARGS, 32, 8, 1) is False
    assert rt.calls == []


# --- causal work-skipping is emitted only for the causal kernel ---
def test_causal_kernel_emits_upper_triangle_skip():
    from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup as mk
    causal = mk(128, 32, 64, True, "fp32", kernel_name="k")
    plain = mk(128, 32, 64, False, "fp32", kernel_name="k")
    # causal: break past the diagonal in the full-block loop + a guarded tail block.
    assert "if (kv_start > q_start + BM - 1u) break;" in causal
    assert "n_full * BN <= q_start + BM - 1u" in causal
    # non-causal must NOT prune (every block is needed) — silent-wrong otherwise.
    assert "break;" not in plain
    assert "q_start + BM - 1u" not in plain


# --- integration: a real compiled FA kernel is actually wired to the dispatch ---
try:
    import torch
    import triton
    import triton.language as tl
    import Metal

    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None and CompileShaderRuntime().available()
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + compile_shader needed")


@requires
@pytest.mark.parametrize("D,dt,causal", [
    (128, torch.float32, False),
    (64, torch.float32, False),
    (64, torch.float16, False),   # fp16 hd64 previously REFUSED -> CPU-fallback
    (64, torch.float16, True),
])
def test_real_fa_routes_through_dispatch(monkeypatch, D, dt, causal):
    """A compiled contiguous head_dim in {64,128} FA on MPS routes through the simd
    dispatch, runs ON GPU, and matches SDPA. hd64 (esp. fp16) is the regression guard:
    it used to lower through the generic path where the K^T 32x64 tt.trans exceeds the
    1024-thread cap -> silent CPU-fallback. Now it must run on the device."""
    import torch.nn.functional as F
    import sys
    sys.path.insert(0, "tests")
    from test_flash_attention import _flash_attn_fwd
    import triton_msl.autotuning._fa_dispatch as fa_mod

    fired = {"n": 0}
    real = fa_mod.dispatch_flash_attention

    def spy(*a, **kw):
        fired["n"] += 1
        return real(*a, **kw)

    # the driver imports the symbol at call time from the module, so patch there
    monkeypatch.setattr(fa_mod, "dispatch_flash_attention", spy)

    Z, H, N = 1, 4, 512
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, D, device="mps", dtype=dt)
    k = torch.randn(Z, H, N, D, device="mps", dtype=dt)
    v = torch.randn(Z, H, N, D, device="mps", dtype=dt)
    out = torch.empty_like(q)
    _flash_attn_fwd[(N // 32, Z * H)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N, BLOCK_M=32, BLOCK_N=32, HEAD_DIM=D, IS_CAUSAL=causal)
    torch.mps.synchronize()

    assert fired["n"] >= 1, "compiled FA kernel did not route through the simd FA dispatch"
    assert str(out.device).startswith("mps"), "FA output left the GPU (CPU-fallback regression)"
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=1.0 / (D ** 0.5))
    tol = 0.02 if dt == torch.float32 else 0.05
    assert (out.float() - ref.float()).abs().max().item() < tol
