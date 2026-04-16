// ===-- SharedMemoryOpToLLVM.cpp - TTG shared memory op lowering ------===//
//
// Conversion patterns for TritonGPU shared memory ops to LLVM IR.
// Maps !ttg.memdesc<...> to LLVM ptr in addrspace(3) (Metal threadgroup).
//
// ===------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton_metal {

// Per-module counter for unique shared memory globals.
static unsigned sharedMemoryCounter = 0;

void resetSharedMemoryCounter() { sharedMemoryCounter = 0; }

// Pattern populator (called from TritonMetalToLLVM.cpp).
void populateSharedMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

} // namespace triton_metal
} // namespace mlir

namespace mlir {
namespace triton_metal {

void populateSharedMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  // Patterns added in later tasks.
  (void)typeConverter;
  (void)patterns;
}

} // namespace triton_metal
} // namespace mlir
