"""Opt-in half-accumulate FA (TRITON_MSL_FA_HALF_ACCUM): fp16 MMA accumulators.

Contract:
  * DEFAULT (env unset) is byte-identical to the shipped float-accumulate kernel.
  * The opt-in emits half8x8 accumulators + half scratch, ONLY for fp16.
  * half_accumulate on fp32 is a hard error (correct-or-refuse).
  * The env flag keys the MSL cache (toggling it must not replay a stale kernel).
  * With the flag on, an fp16 kernel stays correct (looser fp16-accumulate tolerance).
This is a latency/accuracy trade (~4%, ~1% max-abs error), mirroring int8/int4 opt-ins.
"""
import pytest

# Import triton FIRST so its backend discovery runs to completion before anything
# imports triton_msl.backend.compiler — importing compiler mid-triton-init otherwise
# re-enters discovery on a half-loaded module ("Found 0 concrete subclasses").
import triton  # noqa: F401

from triton_msl.codegen._msl_templates import make_flash_attention_kernel_simdgroup as mk
from triton_msl.codegen.generic_lowerer import _fa_half_accumulate
from triton_msl.backend.compiler import _msl_cache_key

try:
    import torch
    import torch.nn.functional as F
    import Metal
    from triton_msl.backend.compile_shader_runtime import CompileShaderRuntime

    HAS = Metal.MTLCreateSystemDefaultDevice() is not None and CompileShaderRuntime().available()
except Exception:
    HAS = False

requires = pytest.mark.skipif(not HAS, reason="Metal + torch + compile_shader needed")


# ----------------------- generation (no GPU) -----------------------

@pytest.mark.parametrize("causal", [False, True])
def test_default_is_float_accumulate(causal):
    """Default (half_accumulate=False) keeps fp32 accumulators + float scratch."""
    msl = mk(128, 32, 64, causal=causal, out_dtype="fp16", kernel_name="fa")
    assert "simdgroup_float8x8 s0(0.0f)" in msl
    assert "simdgroup_float8x8 o[4][TPG];" in msl
    assert "threadgroup float  tg_S[" in msl
    assert "simdgroup_half8x8 s0" not in msl


@pytest.mark.parametrize("causal", [False, True])
def test_half_accumulate_emits_half(causal):
    """Opt-in flips QK/PV accumulators + rescale/normalize + backing scratch to half."""
    msl = mk(128, 32, 64, causal=causal, out_dtype="fp16", kernel_name="fa", half_accumulate=True)
    assert "simdgroup_half8x8 s0(0.0h)" in msl
    assert "simdgroup_half8x8 o[4][TPG];" in msl
    assert "simdgroup_half8x8 ad0, ad1, ad2, ad3, tmp;" in msl
    assert "simdgroup_half8x8 ld, on;" in msl
    assert "threadgroup half  tg_S[" in msl
    assert "threadgroup half  adiag[" in msl
    assert "threadgroup half on_scratch[" in msl
    assert "simdgroup_float8x8" not in msl  # no fp32 accumulators remain


@pytest.mark.parametrize("dt", ["fp32", "f32"])
def test_half_accumulate_on_fp32_refuses(dt):
    """fp32 + half_accumulate is a hard error — no half input to accumulate."""
    with pytest.raises(ValueError, match="requires fp16"):
        mk(128, 32, 64, causal=False, out_dtype=dt, kernel_name="fa", half_accumulate=True)


def test_half_accumulate_default_off_matches_explicit_false():
    """Omitting the flag == passing half_accumulate=False (byte-identical)."""
    a = mk(128, 32, 64, causal=False, out_dtype="fp16", kernel_name="fa")
    b = mk(128, 32, 64, causal=False, out_dtype="fp16", kernel_name="fa", half_accumulate=False)
    assert a == b


# ----------------------- opt-in helper + cache key (no GPU) -----------------------

def test_opt_in_helper(monkeypatch):
    monkeypatch.delenv("TRITON_MSL_FA_HALF_ACCUM", raising=False)
    assert _fa_half_accumulate("f16") is False           # default OFF
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", "1")
    assert _fa_half_accumulate("f16") is True             # fp16 opts in
    assert _fa_half_accumulate("fp16") is True
    assert _fa_half_accumulate("f32") is False            # never for fp32
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", "0")
    assert _fa_half_accumulate("f16") is False            # explicit off


def test_flag_keys_the_cache(monkeypatch):
    """Toggling TRITON_MSL_FA_HALF_ACCUM must change the MSL cache key."""
    monkeypatch.delenv("TRITON_MSL_FA_HALF_ACCUM", raising=False)
    off = _msl_cache_key("ttgir-text", "opts")
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", "1")
    on = _msl_cache_key("ttgir-text", "opts")
    assert off != on, "half-accum flag must key the cache (else stale kernel replay)"


# ----------------------- end-to-end correctness (GPU) -----------------------

def _run(rt, causal, half_accumulate, N=512, Z=1, H=8, D=128):
    lib = rt.get_library(
        mk(D, 32, 64, causal=causal, out_dtype="fp16", kernel_name="fa_ha",
           half_accumulate=half_accumulate)
    )
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, D, device="mps", dtype=torch.float16)
    k = torch.randn(Z, H, N, D, device="mps", dtype=torch.float16)
    v = torch.randn(Z, H, N, D, device="mps", dtype=torch.float16)
    o = torch.empty_like(q)
    nqb = N // 32
    args = ([q, k, v, o]
            + [q.stride(i) for i in range(4)] + [k.stride(i) for i in range(4)]
            + [v.stride(i) for i in range(4)] + [o.stride(i) for i in range(4)] + [Z, H, N])
    rt.dispatch(lib, "fa_ha", args, threads=(nqb * 256, Z * H, 1), group_size=(256, 1, 1))
    torch.mps.synchronize()
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=1.0 / (D ** 0.5))
    return (o.float() - ref.float()).abs().max().item()


@requires
@pytest.mark.parametrize("causal", [False, True])
def test_half_accumulate_runs_and_is_correct(causal):
    """The opt-in kernel runs on GPU and stays within fp16-accumulate tolerance.

    Half-accumulate trades accuracy for ~4% speed: max-abs error is ~1e-2 (vs ~1e-4
    for float-accumulate) — still a valid fp16 attention, not silently wrong."""
    rt = CompileShaderRuntime()
    err = _run(rt, causal, half_accumulate=True)
    assert err < 5e-2, f"half-accum FA causal={causal}: err {err} exceeds fp16-accum tol"


@requires
@pytest.mark.parametrize("causal", [False, True])
def test_half_accumulate_less_accurate_than_float(causal):
    """Sanity: half-accumulate is measurably less accurate than float-accumulate
    (confirms the accumulators actually changed), while both stay correct."""
    rt = CompileShaderRuntime()
    err_float = _run(rt, causal, half_accumulate=False)
    err_half = _run(rt, causal, half_accumulate=True)
    assert err_float < 5e-3
    assert err_half > err_float, "half-accum should be less accurate than float-accum"
