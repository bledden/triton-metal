import platform

from setuptools import setup, find_packages

install_requires = []

# PyObjC Metal bindings are only available on macOS.
if platform.system() == "Darwin":
    install_requires.extend([
        "pyobjc-core>=10.0",
        "pyobjc-framework-Metal>=10.0",
        "pyobjc-framework-MetalPerformanceShaders>=10.0",
        "pyobjc-framework-Cocoa>=10.0",
    ])

setup(
    name="triton-metal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        "triton.backends": [
            "metal = triton_metal.backend",
        ],
    },
)
