"""Gated DeltaNet / Kimi Delta Attention (KDA) prefill kernel.

Linear/delta-rule attention (per-key-dim gate + delta correction), the variant the
2026 frontier models use. It is NOT softmax attention and cannot be expressed as a
single ``@triton.jit`` kernel (the chunked form needs a UT-transform triangular solve),
so it ships as a direct compile_shader op, validated here against a recurrent gated-delta
ground truth. See ``make_kda_kernel`` for the algorithm.
"""

import pytest

# Import triton FIRST so backend discovery completes before triton_msl.backend is touched.
import triton  # noqa: F401

from triton_msl.codegen._msl_templates import make_kda_kernel

try:
    import torch

    import Metal
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None and CompileShaderRuntime().available()
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + compile_shader needed")


def _gdn_recurrent(q, k, v, a, beta):
    """Recurrent gated-delta ground truth: Sg=diag(a)S; u=k^T Sg; S=Sg+b k(v-u)^T; o=q^T S."""
    ZH, T, d = q.shape
    O = torch.zeros_like(v)
    for h in range(ZH):
        S = torch.zeros(d, d)
        for t in range(T):
            Sg = a[h, t].unsqueeze(1) * S
            u = k[h, t] @ Sg
            S = Sg + beta[h, t] * torch.outer(k[h, t], v[h, t] - u)
            O[h, t] = q[h, t] @ S
    return O


@requires
@pytest.mark.parametrize("T", [64, 512])
def test_kda_prefill_matches_recurrent(T):
    ZH, D = 8, 64  # the kernel is fixed at D=64, C=8; one threadgroup per head
    torch.manual_seed(0)
    q = torch.randn(ZH, T, D)
    k = torch.nn.functional.normalize(torch.randn(ZH, T, D), dim=-1)
    v = torch.randn(ZH, T, D)
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn(ZH, T, D))  # per-key-dim gate in (0.9, 1)
    beta = torch.sigmoid(torch.randn(ZH, T))
    ref = _gdn_recurrent(q, k, v, a, beta)

    rt = CompileShaderRuntime()
    lib = rt.get_library(make_kda_kernel())
    args = [t.contiguous().to("mps") for t in (q, k, v, a, beta)]
    out = torch.empty(ZH, T, D, device="mps")
    rt.dispatch(lib, "kda_prefill", args + [out, T], threads=(ZH * 256, 1, 1), group_size=(256, 1, 1))
    torch.mps.synchronize()

    rel = (out.cpu() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-4, f"KDA prefill rel err {rel:.2e} (T={T})"


@requires
def test_kda_attention_op_matches_recurrent():
    """The public op wraps the dispatch (reshape, cached library) and stays correct."""
    from triton_msl.kda import kda_attention

    ZH, D, T = 8, 64, 256
    torch.manual_seed(1)
    q = torch.randn(ZH, T, D)
    k = torch.nn.functional.normalize(torch.randn(ZH, T, D), dim=-1)
    v = torch.randn(ZH, T, D)
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn(ZH, T, D))
    beta = torch.sigmoid(torch.randn(ZH, T))
    ref = _gdn_recurrent(q, k, v, a, beta)

    out = kda_attention(q.to("mps"), k.to("mps"), v.to("mps"), a.to("mps"), beta.to("mps"))
    torch.mps.synchronize()
    rel = (out.cpu() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-4, f"kda_attention op rel err {rel:.2e}"


@requires
def test_kda_attention_op_rejects_bad_shape():
    """T not divisible by the chunk size, and wrong head dim, fail loudly (not silently wrong)."""
    from triton_msl.kda import kda_attention

    z = lambda *s: torch.zeros(*s, device="mps")
    with pytest.raises(ValueError):
        kda_attention(z(8, 60, 64), z(8, 60, 64), z(8, 60, 64), z(8, 60, 64), z(8, 60))  # T=60 not %8
    with pytest.raises(ValueError):
        kda_attention(z(8, 64, 32), z(8, 64, 32), z(8, 64, 32), z(8, 64, 32), z(8, 64))  # head dim != 64


@requires
def test_kda_decode_step_matches_recurrent():
    """Autoregressive decode (state threaded across steps in place) matches the recurrent form."""
    from triton_msl.kda import kda_decode_step

    ZH, D, T = 8, 64, 48
    torch.manual_seed(2)
    q = torch.randn(ZH, T, D)
    k = torch.nn.functional.normalize(torch.randn(ZH, T, D), dim=-1)
    v = torch.randn(ZH, T, D)
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn(ZH, T, D))
    beta = torch.sigmoid(torch.randn(ZH, T))
    ref = _gdn_recurrent(q, k, v, a, beta)

    S = torch.zeros(ZH, D, D, device="mps")
    out = torch.empty(ZH, T, D)
    for t in range(T):
        o = kda_decode_step(
            q[:, t].to("mps"),
            k[:, t].to("mps"),
            v[:, t].to("mps"),
            a[:, t].to("mps"),
            beta[:, t].to("mps"),
            S,
        )
        torch.mps.synchronize()
        out[:, t] = o.cpu()
    rel = (out - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-4, f"kda_decode_step rel err {rel:.2e}"


@requires
def test_kda_attention_fp16():
    """fp16 I/O (fp32 accumulate + state) routes to the half kernel; fp16-grade tolerance."""
    from triton_msl.kda import kda_attention

    ZH, D, T = 8, 64, 256
    torch.manual_seed(3)
    q = torch.randn(ZH, T, D)
    k = torch.nn.functional.normalize(torch.randn(ZH, T, D), dim=-1)
    v = torch.randn(ZH, T, D)
    a = 0.9 + 0.1 * torch.sigmoid(torch.randn(ZH, T, D))
    beta = torch.sigmoid(torch.randn(ZH, T))
    ref = _gdn_recurrent(q, k, v, a, beta)

    args = [t.half().to("mps") for t in (q, k, v, a, beta)]
    out = kda_attention(*args)
    torch.mps.synchronize()
    assert out.dtype == torch.float16
    rel = (out.float().cpu() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 2e-2, f"fp16 KDA rel {rel:.2e}"
