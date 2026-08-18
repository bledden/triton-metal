"""FlashAttention Metal backward pass — trainable attention.

Cross-checks ``triton_msl.fa_backward.flash_attention``'s autograd (Metal dQ/dK/dV kernels)
against torch's own SDPA gradients, full and causal. This is the training half of the
backend (the forward FA was inference-only).
"""

import pytest

import triton  # noqa: F401  (import first — backend discovery)

try:
    import torch
    import torch.nn.functional as F

    import Metal
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None and CompileShaderRuntime().available()
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + compile_shader needed")


@requires
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_backward_matches_autograd(causal):
    from triton_msl.fa_backward import flash_attention

    Z, H, N, Dh = 2, 4, 128, 64
    scale = Dh**-0.5
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, Dh, device="mps", requires_grad=True)
    k = torch.randn(Z, H, N, Dh, device="mps", requires_grad=True)
    v = torch.randn(Z, H, N, Dh, device="mps", requires_grad=True)
    dO = torch.randn(Z, H, N, Dh, device="mps")

    # reference: torch's own SDPA gradients
    qr, kr, vr = (t.detach().clone().requires_grad_() for t in (q, k, v))
    Oref = F.scaled_dot_product_attention(qr, kr, vr, is_causal=causal, scale=scale)
    Oref.backward(dO)

    # ours: forward output + Metal backward
    O = flash_attention(q, k, v, scale=scale, causal=causal)
    O.backward(dO)
    torch.mps.synchronize()

    assert (O - Oref).abs().max().item() < 1e-4, "forward mismatch"
    for name, ours, ref in (("dQ", q.grad, qr.grad), ("dK", k.grad, kr.grad), ("dV", v.grad, vr.grad)):
        rel = (ours - ref).abs().max().item() / ref.abs().max().item()
        assert rel < 1e-3, f"{name} rel {rel:.2e} (causal={causal})"


@requires
def test_flash_attention_rejects_bad_shape():
    from triton_msl.fa_backward import flash_attention

    with pytest.raises(ValueError):
        flash_attention(
            torch.zeros(2, 4, 130, 64, device="mps"),
            torch.zeros(2, 4, 130, 64, device="mps"),
            torch.zeros(2, 4, 130, 64, device="mps"),
        )  # N=130 not %16
    with pytest.raises(ValueError):
        flash_attention(
            torch.zeros(2, 4, 128, 128, device="mps"),
            torch.zeros(2, 4, 128, 128, device="mps"),
            torch.zeros(2, 4, 128, 128, device="mps"),
        )  # head dim != 64
