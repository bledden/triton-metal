// Bridge between the pybind11 module (RTTI-enabled) and MLIR code (no-RTTI).
// This file is compiled with -fno-rtti to match LLVM/MLIR conventions.

#include "triton_metal/Conversion/TritonMetalToLLVM.h"

extern "C" void triton_metal_register_passes() {
    mlir::triton_metal::registerTritonMetalToLLVMPasses();
}
