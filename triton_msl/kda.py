"""Kimi Delta Attention (KDA) / gated DeltaNet — a direct Metal op.

Linear/delta-rule attention: a per-key-dimension forget gate combined with the delta
rule's fast-weight correction. This is the attention the 2026 frontier models (Kimi,
DeltaNet family) are adopting, and it is *not* standard softmax attention. Its chunked
(parallel-prefill) form needs a UT-transform triangular solve, so it cannot be expressed
as a single ``@triton.jit`` kernel and does not route through the FlashAttention path.
It is dispatched directly through ``compile_shader``.

See ``triton_msl.codegen._msl_templates.make_kda_kernel`` for the algorithm and the
chunked derivation, and ``tests/test_kda.py`` for correctness against the recurrent form.
"""

from triton_msl.codegen._msl_templates import make_kda_decode_kernel, make_kda_kernel

_RT = None
_LIB = None
_DEC_LIB = None


def _kernel():
    """Lazily build + cache the compiled KDA prefill library (one compile per process)."""
    global _RT, _LIB
    if _LIB is None:
        from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

        _RT = CompileShaderRuntime()
        _LIB = _RT.get_library(make_kda_kernel())
    return _RT, _LIB


def kda_attention(q, k, v, a, beta):
    """Chunked-prefill gated-delta (KDA) attention.

    Args:
        q, k, v: ``[ZH, T, 64]`` float32 on the ``mps`` device (ZH = batch*heads).
        a:       ``[ZH, T, 64]`` float32 per-key-dim forget gate, values in (0, 1).
        beta:    ``[ZH, T]`` float32 delta-rule step (typically a sigmoid, in (0, 1)).

    Returns:
        ``[ZH, T, 64]`` float32 output on ``mps``.

    Constraints: head dim is fixed at 64, ``T % 8 == 0`` (chunk size 8). One threadgroup
    per head. fp32 accumulate and state; feed fp16 by casting the inputs (a half-I/O
    kernel variant exists in the scratchpad and is a follow-up).
    """
    import torch

    if q.dim() != 3 or q.shape[2] != 64:
        raise ValueError(f"kda_attention expects q of shape [ZH, T, 64], got {tuple(q.shape)}")
    ZH, T, D = q.shape
    if T % 8 != 0:
        raise ValueError(f"kda_attention requires T % 8 == 0 (chunk size 8), got T={T}")

    rt, lib = _kernel()
    out = torch.empty(ZH, T, D, device=q.device, dtype=torch.float32)
    args = [
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        beta.contiguous(),
        out,
        T,
    ]
    rt.dispatch(lib, "kda_prefill", args, threads=(ZH * 256, 1, 1), group_size=(256, 1, 1))
    return out


def _decode_kernel():
    """Lazily build + cache the KDA decode library."""
    global _RT, _DEC_LIB
    if _DEC_LIB is None:
        from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

        if _RT is None:
            _RT = CompileShaderRuntime()
        _DEC_LIB = _RT.get_library(make_kda_decode_kernel())
    return _RT, _DEC_LIB


def kda_decode_step(q, k, v, a, beta, S):
    """One autoregressive KDA decode step, updating the recurrent state in place.

    Args:
        q, k, v, a: ``[ZH, 64]`` float32 on ``mps`` (one token).
        beta:       ``[ZH]`` float32.
        S:          ``[ZH, 64, 64]`` float32 recurrent state, **updated in place**; pass the
                    same tensor across steps (start from ``torch.zeros``, or the prefill's
                    final state, to continue a sequence).

    Returns:
        ``[ZH, 64]`` float32 output for this token.
    """
    import torch

    if q.dim() != 2 or q.shape[1] != 64:
        raise ValueError(f"kda_decode_step expects q of shape [ZH, 64], got {tuple(q.shape)}")
    ZH, D = q.shape
    rt, lib = _decode_kernel()
    out = torch.empty(ZH, D, device=q.device, dtype=torch.float32)
    args = [q.contiguous(), k.contiguous(), v.contiguous(), a.contiguous(), beta.contiguous(), S, out]
    rt.dispatch(lib, "kda_decode", args, threads=(ZH * 256, 1, 1), group_size=(256, 1, 1))
    return out
