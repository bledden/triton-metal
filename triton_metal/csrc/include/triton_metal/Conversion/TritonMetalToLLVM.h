#ifndef TRITON_METAL_CONVERSION_TRITONMETALTOLLVM_H
#define TRITON_METAL_CONVERSION_TRITONMETALTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ModuleOp;
class RewritePatternSet;
namespace triton_metal {

/// Create a pass that converts TritonGPU operations to LLVM IR
/// suitable for Metal GPU compilation.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonMetalToLLVMPass();

/// Register all Metal conversion passes with MLIR's pass infrastructure.
void registerTritonMetalToLLVMPasses();

void populateSharedMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns);

void resetSharedMemoryCounter();

} // namespace triton_metal
} // namespace mlir

#endif // TRITON_METAL_CONVERSION_TRITONMETALTOLLVM_H
