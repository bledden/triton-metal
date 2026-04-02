"""Debug utilities for triton-metal.

Provides multi-level debug output controlled by environment variables:
  TRITON_METAL_DEBUG   - Debug verbosity level (0=off, 1=dump IR/MSL, 2=+timing)
  TRITON_METAL_DUMP_DIR - Directory for debug dumps (default: /tmp/triton_metal_debug)
"""

import os

# Sentinel to distinguish "not yet cached" from "cached value is 0".
_UNSET = object()

_cached_debug_level = _UNSET
_cached_dump_dir = _UNSET


def _debug_level() -> int:
    """Return the current debug level from TRITON_METAL_DEBUG env var.

    Returns 0 if not set. Caches the result to avoid repeated os.environ lookups.
    Call _reset_debug_cache() in tests to clear the cached value.
    """
    global _cached_debug_level
    if _cached_debug_level is _UNSET:
        raw = os.environ.get("TRITON_METAL_DEBUG", "")
        try:
            _cached_debug_level = int(raw) if raw else 0
        except ValueError:
            _cached_debug_level = 0
    return _cached_debug_level


def _dump_dir() -> str:
    """Return the debug dump directory from TRITON_METAL_DUMP_DIR env var.

    Defaults to /tmp/triton_metal_debug if not set.
    Caches the result to avoid repeated os.environ lookups.
    """
    global _cached_dump_dir
    if _cached_dump_dir is _UNSET:
        _cached_dump_dir = os.environ.get(
            "TRITON_METAL_DUMP_DIR", "/tmp/triton_metal_debug"
        )
    return _cached_dump_dir


def _reset_debug_cache():
    """Reset cached debug settings. Intended for use in tests."""
    global _cached_debug_level, _cached_dump_dir
    _cached_debug_level = _UNSET
    _cached_dump_dir = _UNSET
