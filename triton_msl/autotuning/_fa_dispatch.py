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


def dispatch_flash_attention(
    rt, descriptor, kernel_name, kargs, gridX, gridY, gridZ, *,
    launch_exit_hook=None, launch_metadata=None,
):
    try:
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
