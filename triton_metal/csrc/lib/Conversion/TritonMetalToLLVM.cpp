#include "triton_metal/Conversion/TritonMetalToLLVM.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

// Forward declaration — defined in ElementwiseOpToLLVM.cpp
namespace mlir {
namespace triton_metal {
void populateTritonMetalToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns);
} // namespace triton_metal
} // namespace mlir

namespace {

class ConvertTritonMetalToLLVM
    : public mlir::PassWrapper<ConvertTritonMetalToLLVM,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTritonMetalToLLVM)

  llvm::StringRef getArgument() const override {
    return "convert-triton-metal-to-llvm";
  }

  llvm::StringRef getDescription() const override {
    return "Convert TritonGPU operations to LLVM IR for Metal GPU compilation";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    // Set up type converter — handles standard type conversions
    // (i32 -> i32, f32 -> f32, index -> i64, etc.)
    mlir::LLVMTypeConverter typeConverter(ctx);

    // Collect rewrite patterns — this also registers custom type conversions
    // for Triton tensor and pointer types on the typeConverter.
    mlir::RewritePatternSet patterns(ctx);
    mlir::triton_metal::populateTritonMetalToLLVMPatterns(typeConverter,
                                                          patterns);

    // Also populate standard arith->LLVM and cf->LLVM patterns so they
    // share our type converter (tensor<NxT> -> T). This is critical:
    // arith ops on tensor types need to see our scalar conversion.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);

    // Set up conversion target — LLVM dialect is legal, everything else
    // is illegal.
    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    // Mark Triton ops as illegal so the conversion framework requires them
    // to be lowered by our patterns.
    target.addIllegalDialect<mlir::triton::TritonDialect>();

    // Also mark arith and cf as illegal — they will be lowered in this
    // same conversion pass using the shared type converter.
    target.addIllegalDialect<mlir::arith::ArithDialect>();
    target.addIllegalDialect<mlir::cf::ControlFlowDialect>();

    // Apply partial conversion — only ops with matching patterns are lowered.
    // This is intentional: we add patterns incrementally across tasks.
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace triton_metal {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertTritonMetalToLLVMPass() {
  return std::make_unique<ConvertTritonMetalToLLVM>();
}

void registerTritonMetalToLLVMPasses() {
  mlir::PassRegistration<ConvertTritonMetalToLLVM>();
}

} // namespace triton_metal
} // namespace mlir
