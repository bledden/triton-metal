#ifndef TRITON_METAL_CONVERSION_TRITONMETALTOLLVM_H
#define TRITON_METAL_CONVERSION_TRITONMETALTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace llvm { class Module; }

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

void populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns);

void resetSharedMemoryCounter();

/// Coalesce addrspace(3) globals with non-overlapping live ranges via
/// greedy graph coloring. Runs on the LLVM Module after MLIR -> LLVM IR
/// translation.
void aliasSharedMemoryGlobals(llvm::Module &mod);

/// Opportunistically coalesce 4 consecutive scalar loads/stores on
/// addrspace(3) pointers with contiguous GEP indices into a single vector
/// op. Matches the `#shared<{vec=4}>` encoding hint from TTGIR. If the
/// preconditions aren't met, scalar ops remain untouched.
void vectorizeSharedMemoryAccess(llvm::Module &mod);

} // namespace triton_metal
} // namespace mlir

#endif // TRITON_METAL_CONVERSION_TRITONMETALTOLLVM_H
