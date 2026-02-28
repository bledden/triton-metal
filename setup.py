from setuptools import setup, find_packages

setup(
    name="triton-metal",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "triton.backends": [
            "metal = triton_metal.backend",
        ],
    },
)
