# triton_msl/autotuning/_fa_dispatch.py
"""FlashAttention zero-copy dispatch (compile_shader, native 2-D grid).

The simdgroup FlashAttention kernel (head_dim=128, BM=32/BN=64) is emitted with a
2-D threadgroup grid ``(n_q_blocks, Z*H, 1)``. That 2-D grid disqualifies it from
the generic 1-D compile_shader fast-path in the driver (which hard-requires
``gridY == gridZ == 1``), so without this it falls through to the host-roundtrip
metallib path -- measured ~2.5-4.2x SLOWER than dispatching the SAME kernel via
compile_shader (the entire gap is dispatch overhead, not the kernel: cs-2d == cs-1d
in cold A/B). On the zero-copy path the kernel beats PyTorch SDPA up to 2.27x
(fp16 full), median 1.29x across the dtype x causal x N matrix.

Unlike the quantized path (fail-CLOSED: the compiled kernel IS the fast dequant
kernel and the host path can't run it), FlashAttention is fail-OPEN: the host
metallib path produces the SAME correct result, just slower. So any miss here
(non-MPS, compile_shader unavailable, opt-out, or an error) returns False and the
caller simply falls through to the host path -- never a wrong result, never a hard
refusal.

Signature:
    dispatch_flash_attention(rt, descriptor, kernel_name, kargs, gridX, gridY, gridZ,
                             *, launch_exit_hook=None, launch_metadata=None) -> bool

descriptor = ("flash_attention", msl_src, threadgroup_size). kargs is the ordered
non-constexpr arg list (matches the kernel's [[buffer(i)]] order: Q,K,V,Out, 16
strides, Z,H,N). The 2-D dispatch is threads=(gridX*tg, gridY, gridZ),
group_size=(tg, 1, 1) -- exactly the (validated) native-grid launch.
"""


def _dispatch_mla(rt, descriptor, kargs, *, launch_exit_hook=None, launch_metadata=None):
    """MLA (nope/rope) dispatch: concat the split QK tensors and run the qk=head_dim /
    v=v_head_dim kernel. descriptor = ('mla', msl, name, tg, q_nope, q_rope, k_nope,
    k_rope, v, out, Z, H, N) with the last 9 being indices into kargs. FAIL-CLOSED in the
    caller: a False here (non-MPS / bad shape / error) makes the driver refuse, since the
    qk=head_dim kernel's ABI differs from the @jit kernel's (the host path would mis-run)."""
    try:
        import torch

        msl, name, tg = descriptor[1], descriptor[2], int(descriptor[3])
        qn_i, qr_i, kn_i, kr_i, v_i, o_i = descriptor[4:10]
        if msl is None or not name or rt.is_unsupported(msl):
            return False
        qn, qr, kn, kr = kargs[qn_i], kargs[qr_i], kargs[kn_i], kargs[kr_i]
        v, out = kargs[v_i], kargs[o_i]
        for t in (qn, qr, kn, kr, v, out):
            if not (hasattr(t, "data_ptr") and str(getattr(t, "device", "")).startswith("mps")):
                return False
        # Concat along head-dim -> contiguous [Z,H,N,head_dim]; V/Out used directly.
        q = torch.cat([qn, qr], dim=-1).contiguous()
        k = torch.cat([kn, kr], dim=-1).contiguous()
        Z, H, N = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
        n_qb = (N + 31) // 32
        buffers = ([q, k, v, out]
                   + list(q.stride()) + list(k.stride()) + list(v.stride()) + list(out.stride())
                   + [Z, H, N])
        lib = rt.get_library(msl)
        rt.dispatch(lib, name, buffers, threads=(n_qb * tg, Z * H, 1), group_size=(tg, 1, 1))
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)
        return True
    except Exception:
        try:
            rt.mark_unsupported(descriptor[1])
        except Exception:
            pass
        return False


def dispatch_flash_attention(
    rt, descriptor, kernel_name, kargs, gridX, gridY, gridZ, *,
    launch_exit_hook=None, launch_metadata=None,
):
    try:
        if isinstance(descriptor, (tuple, list)) and len(descriptor) >= 10 and descriptor[0] == "mla":
            return _dispatch_mla(rt, descriptor, kargs,
                                 launch_exit_hook=launch_exit_hook, launch_metadata=launch_metadata)
        if not (isinstance(descriptor, (tuple, list)) and len(descriptor) >= 3
                and descriptor[0] == "flash_attention"):
            return False
        msl = descriptor[1]
        tg = int(descriptor[2])
        if msl is None or not kernel_name or rt.is_unsupported(msl):
            return False
        gx, gy, gz = int(gridX), int(gridY), int(gridZ)
        if gx <= 0 or gy <= 0 or gz <= 0 or tg <= 0:
            return False

        lib = rt.get_library(msl)
        # Native 2-D/3-D grid: gx*gy*gz threadgroups, tg threads each (in x).
        # threadgroup_position_in_grid -> (q_block, zh, 0); thread_index -> 0..tg-1.
        rt.dispatch(lib, kernel_name, kargs,
                    threads=(gx * tg, gy, gz), group_size=(tg, 1, 1))
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)
        return True
    except Exception:
        try:
            rt.mark_unsupported(descriptor[1])
        except Exception:
            pass
        return False
