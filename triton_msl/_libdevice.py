"""Provide ``tl.extra.libdevice`` on the Metal backend.

Triton's top-level ``triton.language.extra.libdevice`` is a stub — every function
body is ``...`` — because the real implementation is meant to be supplied by the
active backend's ``language/extra`` sub-package (NVIDIA ships one under
``triton.language.extra.cuda.libdevice``). The Metal backend is an out-of-tree
package and can't drop a sub-package into ``triton/language/extra/``, so without
this shim ``tl.extra.libdevice.rint(x)`` (and the whole family) returns ``None``
at the frontend and no op is ever emitted.

The Metal codegen already lowers the CUDA ``__nv_*`` extern-elementwise symbols to
Metal builtins (``generic_lowerer`` strips the ``__nv_`` prefix / trailing ``f``
and maps the handful that differ). The CUDA libdevice implementations emit exactly
those ``__nv_*`` symbols, so we bind them onto the stub at import.

Correctness boundary: this shim only makes the frontend *emit* the op. Whether it
lowers is decided in codegen, which either maps a symbol to a correct Metal builtin
or produces an MSL that fails to compile (a loud error) — never a silent-wrong. A
symbol with no Metal equivalent therefore surfaces as a compile failure, honoring
the correct-or-refuse contract.
"""


def install() -> int:
    """Point ``tl.extra.libdevice`` at the real (CUDA) libdevice implementation.

    Returns 1 on success, 0 if the CUDA libdevice isn't available. Idempotent.

    We replace the whole module (in ``sys.modules`` and as the ``extra.libdevice``
    attribute) rather than copying functions onto the stub: per-function ``setattr``
    is fragile against the stub being re-executed while ``triton.language.extra``
    finishes loading, whereas a ``sys.modules`` entry is authoritative for any
    later ``import`` / attribute access.
    """
    import sys
    import importlib

    try:
        extra = importlib.import_module("triton.language.extra")
        real = importlib.import_module("triton.language.extra.cuda.libdevice")
    except Exception:
        return 0

    # Already pointing at a real (non-stub) libdevice? Nothing to do.
    current = getattr(extra, "libdevice", None)
    if current is real:
        return 1

    sys.modules["triton.language.extra.libdevice"] = real
    setattr(extra, "libdevice", real)
    return 1
