"""AIR-version probe for the opt-in C++ lowering path.

The C++ path emits LLVM IR that `xcrun metal -c -x ir` compiles directly; its
`air.version` / `air.language_version` metadata must match the installed
toolchain or the module is rejected and silently falls back to CPU (macOS 26
wants AIR 2.8 / Metal 4.0; older wanted 2.7 / 3.2). device_detect probes the
real version at runtime instead of hardcoding it.
"""

import platform

import pytest

from triton_msl.backend.device_detect import (
    _AIR_VERSION_FALLBACK,
    _extract_air_metadata_ints,
    air_version_components,
)


# --- Pure parsing: deterministic, no toolchain needed ---


def test_extract_air_version_triple():
    ir = "\n".join(
        [
            "!air.version = !{!18}",
            "!air.language_version = !{!19}",
            "!18 = !{i32 2, i32 8, i32 0}",
            '!19 = !{!"Metal", i32 4, i32 0, i32 0}',
        ]
    )
    assert _extract_air_metadata_ints(ir, "air.version") == [2, 8, 0]


def test_extract_language_version_skips_metal_string():
    ir = "\n".join(
        [
            "!air.language_version = !{!19}",
            '!19 = !{!"Metal", i32 4, i32 0, i32 0}',
        ]
    )
    # The leading !"Metal" string operand is ignored; only i32 components returned.
    assert _extract_air_metadata_ints(ir, "air.language_version") == [4, 0, 0]


def test_extract_missing_named_metadata_returns_none():
    assert _extract_air_metadata_ints("!something.else = !{!0}", "air.version") is None


def test_extract_does_not_collide_on_node_prefix():
    # Resolving !18 must not accidentally match !180's value node.
    ir = "\n".join(
        [
            "!air.version = !{!18}",
            "!18 = !{i32 2, i32 7, i32 0}",
            "!180 = !{i32 9, i32 9, i32 9}",
        ]
    )
    assert _extract_air_metadata_ints(ir, "air.version") == [2, 7, 0]


def test_air_version_name_not_confused_with_language_version():
    # "air.version" must not match inside "air.language_version".
    ir = "\n".join(
        [
            "!air.language_version = !{!19}",
            '!19 = !{!"Metal", i32 4, i32 0, i32 0}',
        ]
    )
    assert _extract_air_metadata_ints(ir, "air.version") is None


def test_fallback_is_pre_tahoe_default():
    # AIR 2.7 / Metal 3.2 — the long-standing pre-macOS-26 toolchain profile.
    assert _AIR_VERSION_FALLBACK == (2, 7, 3, 2)


# --- Live probe (needs the Metal toolchain) ---


@pytest.mark.skipif(platform.system() != "Darwin", reason="needs the Metal toolchain")
def test_air_version_components_sane_on_darwin():
    comp = air_version_components()
    assert isinstance(comp, tuple) and len(comp) == 4
    assert all(isinstance(x, int) for x in comp)
    air_major, air_minor, lang_major, lang_minor = comp
    # AIR major has been 2 for every Metal toolchain; minor tracks the toolchain
    # (2.7 pre-Tahoe, 2.8 on macOS 26). Metal language major is 3 or 4.
    assert air_major == 2
    assert lang_major in (3, 4)
