"""Triton Metal backend plugin.

Triton discovers backends via entry points. The entry point
`triton.backends: metal = triton_metal.backend` tells Triton to import
`triton_metal.backend.compiler` and `triton_metal.backend.driver`
directly — no re-exports needed here.
"""
