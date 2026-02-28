"""Triton Metal backend plugin.

Exports MetalBackend and MetalDriver for Triton backend discovery.
The entry point `triton.backends: metal = triton_metal.backend` expects
this package to expose `backend` and `driver` at the module level.
"""

from triton_metal.backend.compiler import MetalBackend as backend
from triton_metal.backend.driver import MetalDriver as driver

__all__ = ["backend", "driver"]
