"""FlashAttention with a Metal backward pass — enables TRAINING.

The backend's forward FlashAttention is inference-only (torch SDPA routes to the simdgroup
kernel, but no gradient). This exposes ``flash_attention`` as a ``torch.autograd.Function``:
the forward computes the output (and the logsumexp the backward needs), and the backward
dispatches the tiled, MMA-optimized Metal dK/dV and dQ kernels (FA-2 backward). So an
attention call inside a training loop gets its gradients from Metal.

Head dim is fixed at 64, ``N % 16 == 0`` (the backward tile). The forward currently
recomputes the logsumexp in torch (a fused forward that saves it is a follow-up). See
``make_fa_backward_dkv_kernel`` / ``make_fa_backward_dq_kernel`` for the kernels and
``tests/test_fa_backward.py`` for the autograd cross-check.
"""

import torch
import torch.nn.functional as F

from triton_msl.codegen._msl_templates import (
    make_fa_backward_dkv_kernel,
    make_fa_backward_dq_kernel,
)

_RT = None
_LIBS = {}  # bool(causal) -> (dkv_lib, dq_lib)


def _dispatch_backward(q, k, v, dO, L, D, scale, causal):
    """Run the two Metal backward kernels. All tensors [ZH,N,64]/[ZH,N] contiguous mps float32."""
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    global _RT
    if _RT is None:
        _RT = CompileShaderRuntime()
    key = bool(causal)
    if key not in _LIBS:
        _LIBS[key] = (
            _RT.get_library(make_fa_backward_dkv_kernel(causal)),
            _RT.get_library(make_fa_backward_dq_kernel(causal)),
        )
    dkv_lib, dq_lib = _LIBS[key]
    ZH, N, _ = q.shape
    dK, dV, dQ = torch.empty_like(q), torch.empty_like(q), torch.empty_like(q)
    grid = ((N // 16) * 256, ZH, 1)
    base = [q, k, v, dO, L, D]
    _RT.dispatch(dkv_lib, "fa_bwd_dkv", base + [dK, dV, N, scale], threads=grid, group_size=(256, 1, 1))
    _RT.dispatch(dq_lib, "fa_bwd_dq", base + [dQ, N, scale], threads=grid, group_size=(256, 1, 1))
    return dQ, dK, dV


class _FlashAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal):
        O = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
        S = (q @ k.transpose(-2, -1)) * scale
        if causal:
            N = q.shape[-2]
            S = S.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool, device=q.device), 1), float("-inf"))
        L = torch.logsumexp(S, dim=-1)
        ctx.save_for_backward(q, k, v, O, L)
        ctx.scale, ctx.causal = scale, causal
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, O, L = ctx.saved_tensors
        D = (dO * O).sum(-1)
        dQ, dK, dV = _dispatch_backward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            dO.contiguous(),
            L.contiguous(),
            D.contiguous(),
            ctx.scale,
            ctx.causal,
        )
        return dQ, dK, dV, None, None


def flash_attention(q, k, v, scale=None, causal=False):
    """FlashAttention whose backward runs on Metal (trainable).

    Args:
        q, k, v: ``[..., N, 64]`` float32 on ``mps``; leading dims are folded to ZH.
        scale: softmax scale (default ``1/sqrt(64)``).
        causal: causal masking.

    Returns an output of the same shape; ``.backward()`` produces dQ/dK/dV via Metal.
    """
    if q.shape[-1] != 64:
        raise ValueError(f"flash_attention (Metal backward) supports head_dim=64, got {q.shape[-1]}")
    if q.shape[-2] % 16 != 0:
        raise ValueError(f"N must be divisible by 16, got {q.shape[-2]}")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    orig = q.shape
    q2, k2, v2 = (t.reshape(-1, orig[-2], orig[-1]) for t in (q, k, v))
    out = _FlashAttentionFn.apply(q2, k2, v2, scale, causal)
    return out.reshape(orig)
