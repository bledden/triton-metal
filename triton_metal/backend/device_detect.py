"""Metal device detection for triton-metal.

Detects Apple Silicon chip generation, Metal version, GPU capabilities,
and Neural Accelerator availability.  Results are cached per-process
(the Metal device is a system singleton on macOS).

Usage:
    from triton_metal.backend.device_detect import get_device_info
    info = get_device_info()
    print(info.chip_family, info.metal_version, info.supports_metal4)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# DeviceInfo dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceInfo:
    """Cached snapshot of the Metal device capabilities."""

    chip_family: str  # "M1", "M2", "M3", "M4", "M5", "unknown"
    chip_variant: str  # "base", "Pro", "Max", "Ultra"
    gpu_core_count: int
    max_threads_per_threadgroup: int
    metal_version: str  # "3.0", "3.1", "3.2", "4.0", "4.1"
    has_neural_accelerator: bool  # True for M5+
    has_bfloat16: bool  # True for Metal 3.1+
    supports_metal4: bool
    supports_tensor_ops: bool  # M5 Neural Accelerators in GPU pipeline

    @property
    def metal_std_flag(self) -> str:
        """Return the ``-std=`` flag for xcrun metal compilation."""
        return f"-std=metal{self.metal_version}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Chip family pattern: "Apple M4 Max" -> ("M4", "Max")
_CHIP_RE = re.compile(
    r"Apple\s+(M[1-9]\d*)\s*(Pro|Max|Ultra)?",
    re.IGNORECASE,
)

# Maps chip family to a rough default GPU core count per variant.
# Real core counts vary by binning; these are canonical full-config values.
_CORE_COUNTS: dict[str, dict[str, int]] = {
    "M1": {"base": 8, "Pro": 16, "Max": 32, "Ultra": 64},
    "M2": {"base": 10, "Pro": 19, "Max": 38, "Ultra": 76},
    "M3": {"base": 10, "Pro": 18, "Max": 40, "Ultra": 80},
    "M4": {"base": 10, "Pro": 20, "Max": 40, "Ultra": 80},
    "M5": {"base": 12, "Pro": 24, "Max": 48, "Ultra": 96},
}


def _parse_chip(device_name: str) -> tuple[str, str]:
    """Extract (chip_family, chip_variant) from a Metal device name string.

    Returns ("unknown", "base") if the name doesn't match.
    """
    if not device_name:
        return ("unknown", "base")
    m = _CHIP_RE.search(device_name)
    if m is None:
        return ("unknown", "base")
    family = m.group(1).upper()  # "m4" -> "M4"
    variant = m.group(2) or "base"
    if variant != "base":
        # Normalize case: "pro" -> "Pro"
        variant = variant.capitalize()
    return (family, variant)


def _estimate_core_count(family: str, variant: str) -> int:
    """Return the expected GPU core count for a chip family/variant."""
    family_map = _CORE_COUNTS.get(family)
    if family_map is None:
        # Unknown future chip — guess based on variant
        return {"base": 10, "Pro": 20, "Max": 40, "Ultra": 80}.get(variant, 10)
    return family_map.get(variant, family_map.get("base", 10))


def _detect_sdk_version() -> Optional[str]:
    """Return the macOS SDK version string via ``xcrun --show-sdk-version``.

    Returns None if xcrun is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["xcrun", "--show-sdk-version"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _infer_metal_version(sdk_version: Optional[str], chip_family: str) -> str:
    """Infer the highest supported Metal Shading Language version.

    The mapping is based on the macOS SDK version (which correlates with
    the Xcode toolchain shipping the Metal compiler):

        macOS 16.0+  (Xcode 17) -> Metal 4.1  (M5 tensor ops)
        macOS 15.4+  (Xcode 16.3) -> Metal 4.0
        macOS 15.0+  (Xcode 16) -> Metal 3.2
        macOS 14.0+  (Xcode 15) -> Metal 3.1
        macOS 13.0+  (Xcode 14) -> Metal 3.0
        older / unknown          -> Metal 3.0

    Additionally, Metal 4.x features require M4+ hardware.
    """
    if sdk_version is None:
        return "3.0"

    try:
        parts = sdk_version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return "3.0"

    # Determine max MSL version the SDK toolchain supports.
    if major >= 16:
        sdk_metal = "4.1"
    elif major == 15 and minor >= 4:
        sdk_metal = "4.0"
    elif major >= 15:
        sdk_metal = "3.2"
    elif major >= 14:
        sdk_metal = "3.1"
    else:
        sdk_metal = "3.0"

    # Metal 4.x requires M4+ hardware.  Clamp to 3.2 on older chips.
    if sdk_metal.startswith("4"):
        chip_gen = _chip_generation(chip_family)
        if chip_gen < 4:
            return "3.2" if major >= 15 else ("3.1" if major >= 14 else "3.0")

    return sdk_metal


def _chip_generation(family: str) -> int:
    """Return the numeric generation from a chip family string (e.g. "M4" -> 4)."""
    m = re.match(r"M(\d+)", family, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_cached_info: Optional[DeviceInfo] = None


def get_device_info() -> DeviceInfo:
    """Return cached device info.  Detection runs at most once per process."""
    global _cached_info
    if _cached_info is not None:
        return _cached_info
    _cached_info = _detect_device_info()
    return _cached_info


def _detect_device_info() -> DeviceInfo:
    """Run full device detection (called once, then cached)."""
    # Lazy-import Metal to avoid crash on non-macOS or missing PyObjC.
    try:
        import Metal  # type: ignore[import-untyped]
        device = Metal.MTLCreateSystemDefaultDevice()
    except ImportError:
        device = None

    if device is None:
        # Fallback: no Metal device available.
        return DeviceInfo(
            chip_family="unknown",
            chip_variant="base",
            gpu_core_count=0,
            max_threads_per_threadgroup=0,
            metal_version="3.0",
            has_neural_accelerator=False,
            has_bfloat16=False,
            supports_metal4=False,
            supports_tensor_ops=False,
        )

    device_name = device.name() or ""
    family, variant = _parse_chip(device_name)
    core_count = _estimate_core_count(family, variant)
    max_threads = device.maxThreadsPerThreadgroup().width

    sdk_version = _detect_sdk_version()
    metal_version = _infer_metal_version(sdk_version, family)

    gen = _chip_generation(family)

    # Metal 3.1+ supports bfloat16 in shaders.
    metal_major, metal_minor = (int(x) for x in metal_version.split("."))
    has_bfloat16 = (metal_major, metal_minor) >= (3, 1)

    supports_metal4 = metal_major >= 4

    # Neural Accelerators (GPU tensor ops) are available on M5+ with Metal 4.1+.
    has_neural_accel = gen >= 5
    supports_tensor_ops = has_neural_accel and (metal_major, metal_minor) >= (4, 1)

    return DeviceInfo(
        chip_family=family,
        chip_variant=variant,
        gpu_core_count=core_count,
        max_threads_per_threadgroup=max_threads,
        metal_version=metal_version,
        has_neural_accelerator=has_neural_accel,
        has_bfloat16=has_bfloat16,
        supports_metal4=supports_metal4,
        supports_tensor_ops=supports_tensor_ops,
    )


def reset_device_cache() -> None:
    """Clear cached device info.  Intended for testing."""
    global _cached_info
    _cached_info = None
