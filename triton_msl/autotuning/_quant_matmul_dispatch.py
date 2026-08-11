# triton_msl/autotuning/_quant_matmul_dispatch.py
"""Quantized-matmul dispatch (mirror of _fast_matmul_dispatch).

Routes a recognized weight-only int8 quantized matmul to the dedicated
``make_int8_matmul_fast(layout='kn')`` simdgroup kernel via compile_shader.
The descriptor is built at compile time by
``_maybe_quant_matmul_descriptor`` (correct-or-refuse; only the CANONICAL
shape produces a descriptor). Kept in its own module so the dispatch is unit-
testable without triggering Triton backend discovery.

Signature:
    dispatch_quant_matmul(rt, descriptor, kargs, *, launch_exit_hook=None,
                          launch_metadata=None) -> bool

Returns True if it dispatched (caller returns immediately), False otherwise
(caller decides the fallback — for quantized the driver REFUSES rather than
run the mismatched compiled kernel, so a False here is a hard error upstream).
On any internal error returns False (no exception escapes).
"""

import math as _math

# The kernel bk (K-tile) is fixed at 32 in the descriptor's make_int8_matmul_fast.
_BK = 32


def dispatch_quant_matmul(rt, descriptor, kargs, *, launch_exit_hook=None, launch_metadata=None):
    """Attempt to dispatch the fast quantized-matmul kernel.

    Parameters
    ----------
    rt : CompileShaderRuntime-like
        Exposes is_unsupported / mark_unsupported / get_library / dispatch.
    descriptor : tuple
        (fast_msl, m_idx, n_idx, k_idx, tile_m, tile_n, stride_checks).
    kargs : list
        Non-constexpr kernel args in buffer order. kargs[:8] =
        [input, weight, output, scale, zero, M, N, K]; strides follow.

    Returns
    -------
    bool : True if dispatched; False if skipped/failed (misaligned shape,
           runtime stride mismatch, or any error).
    """
    try:
        fast_msl = descriptor[0]
        m_idx, n_idx, k_idx = descriptor[1], descriptor[2], descriptor[3]
        tile_m, tile_n = descriptor[4], descriptor[5]
        stride_checks = descriptor[6] if len(descriptor) > 6 else ()
    except (TypeError, ValueError, IndexError):
        return False

    if fast_msl is None or rt.is_unsupported(fast_msl):
        return False

    try:
        M = int(kargs[m_idx])
        N = int(kargs[n_idx])
        K = int(kargs[k_idx])

        # RUNTIME STRIDE CONTRACT: the fast kernel assumes row-major (A row == K,
        # B row == N, C row == N). A stride arg that differs at runtime (a
        # transposed / column-sliced operand) would be SILENTLY WRONG -> skip.
        for arg_idx, expected_idx in stride_checks:
            actual = int(kargs[arg_idx])
            expected = 1 if expected_idx < 0 else int(kargs[expected_idx])
            if actual != expected:
                return False

        # SIZE CONTRACT: the kernel has NO edge handling and floor-divides N by the
        # column tile, so all three dims must divide exactly. A non-conforming shape
        # is skipped (the driver then refuses -> never silent-wrong).
        if not (M > 0 and N > 0 and K > 0 and M % tile_m == 0 and N % tile_n == 0 and K % _BK == 0):
            return False

        n_groups = _math.ceil(M / tile_m) * _math.ceil(N / tile_n)
        lib = rt.get_library(fast_msl)
        # The kernel declares exactly 8 buffers (input,weight,output,scale,zero,M,N,K);
        # pass only those (kargs[:8]) — trailing stride args have no buffer slot.
        rt.dispatch(lib, "int8_matmul_fast", kargs[:8], threads=n_groups * 32, group_size=32)
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)
        return True

    except Exception:
        try:
            rt.mark_unsupported(fast_msl)
        except Exception:
            pass
        return False
