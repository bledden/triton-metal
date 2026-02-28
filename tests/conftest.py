import platform

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests on non-macOS platforms."""
    if platform.system() != "Darwin":
        skip = pytest.mark.skip(reason="Metal tests require macOS")
        for item in items:
            item.add_marker(skip)


@pytest.fixture
def metal_device():
    """Provide a Metal device for tests that need one."""
    try:
        import Metal

        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            pytest.skip("No Metal GPU available")
        return device
    except ImportError:
        pytest.skip("pyobjc-framework-Metal not installed")
