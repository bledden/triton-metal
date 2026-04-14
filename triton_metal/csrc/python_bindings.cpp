// pybind11 requires RTTI, but LLVM/MLIR headers are built without RTTI.
// To avoid ABI conflicts, this file does NOT include any MLIR headers.
// Instead it calls through a thin C-linkage wrapper defined in
// python_bindings_bridge.cpp (compiled with -fno-rtti alongside the
// rest of the MLIR code).

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Defined in python_bindings_bridge.cpp (compiled with -fno-rtti).
extern "C" void triton_metal_register_passes();

PYBIND11_MODULE(_triton_metal_cpp, m) {
    m.doc() = "C++ MLIR passes for triton-metal";
    m.def("register_metal_passes", []() {
        triton_metal_register_passes();
    }, "Register all Metal conversion passes with MLIR's pass infrastructure");
}
