"""Metal-compatible libdevice replacement for Inductor.

CUDA's libdevice maps math functions to __nv_* extern calls resolved against
libdevice.10.bc. Metal has no such library, but MSL has built-in equivalents
for most of these. This module provides @triton.jit implementations that
generate standard MLIR math ops our Metal backend already handles.
"""

import types

import triton
import triton.language as tl


# --- Direct tl.math wrappers (generate MLIR math ops) ---

@triton.jit
def exp(x):
    return tl.math.exp(x)


@triton.jit
def exp2(x):
    return tl.math.exp2(x)


@triton.jit
def log(x):
    return tl.math.log(x)


@triton.jit
def log2(x):
    return tl.math.log2(x)


@triton.jit
def sqrt(x):
    return tl.math.sqrt(x)


@triton.jit
def rsqrt(x):
    return tl.math.rsqrt(x)


@triton.jit
def sin(x):
    return tl.math.sin(x)


@triton.jit
def cos(x):
    return tl.math.cos(x)


@triton.jit
def ceil(x):
    return tl.math.ceil(x)


@triton.jit
def floor(x):
    return tl.math.floor(x)


@triton.jit
def abs(x):
    return tl.math.abs(x)


@triton.jit
def erf(x):
    return tl.math.erf(x)


@triton.jit
def fma(x, y, z):
    return tl.math.fma(x, y, z)


# --- Composite implementations ---

@triton.jit
def tanh(x):
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    e2x = tl.math.exp(x + x)
    return (e2x - 1.0) / (e2x + 1.0)


@triton.jit
def fast_tanhf(x):
    # Same as tanh for Metal
    e2x = tl.math.exp(x + x)
    return (e2x - 1.0) / (e2x + 1.0)


@triton.jit
def sinh(x):
    # sinh(x) = (exp(x) - exp(-x)) / 2
    ex = tl.math.exp(x)
    emx = tl.math.exp(-x)
    return (ex - emx) * 0.5


@triton.jit
def cosh(x):
    # cosh(x) = (exp(x) + exp(-x)) / 2
    ex = tl.math.exp(x)
    emx = tl.math.exp(-x)
    return (ex + emx) * 0.5


@triton.jit
def tan(x):
    return tl.math.sin(x) / tl.math.cos(x)


@triton.jit
def asin(x):
    # Approximation using identity: asin(x) = atan2(x, sqrt(1 - x*x))
    # Use Triton's extern_elementwise if available, otherwise polynomial approx
    return _asin_approx(x)


@triton.jit
def _asin_approx(x):
    # Handbook of Mathematical Functions (Abramowitz & Stegun) 4.4.46
    # asin(x) ≈ pi/2 - sqrt(1-x) * (a0 + a1*x + a2*x^2 + a3*x^3)
    # For |x| <= 1. Max error ~5e-5.
    ax = tl.math.abs(x)
    s = tl.math.sqrt(1.0 - ax)
    r = -0.0187293 * ax + 0.0742610
    r = r * ax - 0.2121144
    r = r * ax + 1.5707288
    result = 1.5707963267948966 - s * r
    return tl.where(x < 0.0, -result, result)


@triton.jit
def acos(x):
    return 1.5707963267948966 - _asin_approx(x)


@triton.jit
def atan(x):
    # Polynomial approximation for atan
    return _atan_approx(x)


@triton.jit
def _atan_approx(x):
    # Range reduction + polynomial
    ax = tl.math.abs(x)
    # For |x| > 1, use atan(x) = pi/2 - atan(1/x)
    inv = 1.0 / ax
    t = tl.where(ax > 1.0, inv, ax)
    # Polynomial coefficients (minimax on [0,1])
    s = t * t
    r = -0.0464964749 * s + 0.15931422
    r = r * s - 0.327622764
    r = r * s + 0.999995630
    r = r * t
    r = tl.where(ax > 1.0, 1.5707963267948966 - r, r)
    return tl.where(x < 0.0, -r, r)


@triton.jit
def atan2(y, x):
    # atan2(y, x) via atan(y/x) with quadrant correction
    ratio = y / x
    base = _atan_approx(ratio)
    # Quadrant adjustment
    pi = 3.141592653589793
    adj = tl.where(y >= 0.0, pi, -pi)
    result = tl.where(x >= 0.0, base, base + adj)
    # Handle x == 0
    half_pi = 1.5707963267948966
    on_y_axis = tl.where(y >= 0.0, half_pi, -half_pi)
    return tl.where(x == 0.0, on_y_axis, result)


@triton.jit
def expm1(x):
    return tl.math.exp(x) - 1.0


@triton.jit
def log1p(x):
    return tl.math.log(1.0 + x)


@triton.jit
def log10(x):
    return tl.math.log2(x) * 0.30102999566398114  # 1/log2(10)


@triton.jit
def pow(x, y):
    return tl.math.exp(y * tl.math.log(x))


@triton.jit
def hypot(x, y):
    return tl.math.sqrt(x * x + y * y)


@triton.jit
def copysign(x, y):
    ax = tl.math.abs(x)
    return tl.where(y >= 0.0, ax, -ax)


@triton.jit
def fmod(x, y):
    # fmod(x, y) = x - trunc(x/y) * y
    # Use floor-based approach: x - floor(x/y) * y gives remainder with floor division
    # fmod uses truncation: x - int(x/y) * y
    return x - (x / y).to(tl.int32).to(tl.float32) * y


@triton.jit
def trunc(x):
    return x.to(tl.int32).to(tl.float32)


@triton.jit
def nearbyint(x):
    # Round to nearest even (banker's rounding)
    return (x + 0.5).to(tl.int32).to(tl.float32)


@triton.jit
def llrint(x):
    return (x + 0.5).to(tl.int64)


@triton.jit
def erfc(x):
    return 1.0 - tl.math.erf(x)


@triton.jit
def erfinv(x):
    # Rational approximation for erfinv
    # Based on J.M. Blair approximation
    a = tl.math.abs(x)
    w = -tl.math.log((1.0 - a) * (1.0 + a))
    p = tl.where(w < 5.0,
                 _erfinv_small(w),
                 _erfinv_large(w))
    return tl.where(x < 0.0, -p, p)


@triton.jit
def _erfinv_small(w):
    w = w - 2.5
    p = 2.81022636e-08
    p = 3.43273939e-07 + p * w
    p = -3.5233877e-06 + p * w
    p = -4.39150654e-06 + p * w
    p = 0.00021858087 + p * w
    p = -0.00125372503 + p * w
    p = -0.00417768164 + p * w
    p = 0.246640727 + p * w
    p = 1.50140941 + p * w
    return p


@triton.jit
def _erfinv_large(w):
    w = tl.math.sqrt(w) - 3.0
    p = -0.000200214257
    p = 0.000100950558 + p * w
    p = 0.00134934322 + p * w
    p = -0.00367342844 + p * w
    p = 0.00573950773 + p * w
    p = -0.0076224613 + p * w
    p = 0.00943887047 + p * w
    p = 1.00167406 + p * w
    p = 2.83297682 + p * w
    return p


from triton.language import core as _core


# NOTE: we lower these via tt.extern_elementwise rather than `x == x` / `x != inf`
# tricks because the Metal shader compiler runs with fast-math ON by default,
# which assumes no NaN/Inf and folds such comparisons to constants. Our
# generic_lowerer maps these __nv_* symbols to MSL's native isinf/isnan/isfinite
# built-ins, which are NOT affected by fast-math.

@_core.extern
def isinf(arg0, _semantic=None):
    return _core.extern_elementwise(
        "", "", [arg0], {
            (_core.dtype("fp32"), ): ("__nv_isinff", _core.dtype("int32")),
            (_core.dtype("fp64"), ): ("__nv_isinfd", _core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(_core.int1, _semantic=_semantic)


@_core.extern
def isnan(arg0, _semantic=None):
    return _core.extern_elementwise(
        "", "", [arg0], {
            (_core.dtype("fp32"), ): ("__nv_isnanf", _core.dtype("int32")),
            (_core.dtype("fp64"), ): ("__nv_isnand", _core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(_core.int1, _semantic=_semantic)


@_core.extern
def finitef(arg0, _semantic=None):
    return _core.extern_elementwise(
        "", "", [arg0], {
            (_core.dtype("fp32"), ): ("__nv_finitef", _core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(_core.int1, _semantic=_semantic)


@_core.extern
def isfinited(arg0, _semantic=None):
    return _core.extern_elementwise(
        "", "", [arg0], {
            (_core.dtype("fp32"), ): ("__nv_isfinited", _core.dtype("int32")),
            (_core.dtype("fp64"), ): ("__nv_isfinited", _core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(_core.int1, _semantic=_semantic)


@triton.jit
def signbit(x):
    return x < 0.0


@triton.jit
def lgamma(x):
    # Stirling's approximation for lgamma
    # lgamma(x) ≈ 0.5*ln(2*pi) + (x-0.5)*ln(x) - x + 1/(12*x)
    return 0.9189385332046727 + (x - 0.5) * tl.math.log(x) - x + 1.0 / (12.0 * x)


@triton.jit
def nextafter(x, y):
    # Simplified: return x + tiny_step towards y
    eps = 1.1920928955078125e-07  # FP32 epsilon
    step = tl.where(y > x, eps, -eps)
    return tl.where(x == y, x, x + step * tl.math.abs(x) + step)


@triton.jit
def asinh(x):
    return tl.math.log(x + tl.math.sqrt(x * x + 1.0))


@triton.jit
def acosh(x):
    return tl.math.log(x + tl.math.sqrt(x * x - 1.0))


@triton.jit
def atanh(x):
    return 0.5 * tl.math.log((1.0 + x) / (1.0 - x))


@triton.jit
def ilogb(x):
    # Integer part of log2(|x|)
    return tl.math.floor(tl.math.log2(tl.math.abs(x))).to(tl.int32)


@triton.jit
def ldexp(x, n):
    # x * 2^n
    return x * tl.math.exp2(n.to(tl.float32))


# Package this module's functions as a module object.
#
# Used in two places:
#   1. Inductor's triton_compat.libdevice (monkey-patched in inductor/__init__.py)
#   2. MetalBackend.get_module_map() remaps triton.language.extra.libdevice →
#      metal_libdevice during ast_to_ttir, so `from triton.language.extra import
#      libdevice` followed by `libdevice.exp(x)` inside a @triton.jit resolves
#      here. See triton_metal/backend/compiler.py.
#
# Completeness: the compiler walks the kernel's module globals and for any
# global whose __module__ == "triton.language.extra.libdevice" does
# `getattr(metal_libdevice, <name>)`. Missing attrs raise AttributeError during
# make_ir even if the kernel never calls them. So we first copy ALL names from
# the upstream stub module (so lookups never miss), then override with our
# implementations.
metal_libdevice = types.ModuleType("metal_libdevice")

# Seed with every public name from the upstream stub. These are mostly bodies
# of `...` which would return None if called, but that's fine — they're only
# problematic when invoked. This way, unused references (e.g. `my_fast_dividef
# = libdevice.fast_dividef` at test-module scope) don't break unrelated kernels.
try:
    from triton.language.extra import libdevice as _upstream_stub
    for _name in dir(_upstream_stub):
        if _name.startswith("_"):
            continue
        metal_libdevice.__dict__[_name] = getattr(_upstream_stub, _name)
except ImportError:
    pass

# Override with our Metal implementations.
# Include both @triton.jit functions (hasattr 'fn') and @core.extern builtins
# (hasattr 'signature' + marked with TRITON_BUILTIN).
from triton.language.core import TRITON_BUILTIN as _TRITON_BUILTIN
metal_libdevice.__dict__.update({
    name: obj for name, obj in globals().items()
    if not name.startswith("_") and name != "metal_libdevice"
    and callable(obj)
    and (hasattr(obj, "fn") or getattr(obj, _TRITON_BUILTIN, False))
})
# Also add non-jit module references
metal_libdevice.tl = tl
metal_libdevice.triton = triton
