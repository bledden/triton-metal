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

// Helper: create a unique threadgroup global for the given memdesc shape/type.
static LLVM::GlobalOp createTgGlobal(ModuleOp module,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc,
                                      ArrayRef<int64_t> shape,
                                      Type elemTy) {
  uint64_t numElems = 1;
  for (int64_t d : shape) numElems *= d;
  auto arrTy = LLVM::LLVMArrayType::get(elemTy, numElems);
  std::string name = "__tg_shared_" + std::to_string(sharedMemoryCounter++);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return LLVM::GlobalOp::create(
      rewriter, loc, arrTy, /*isConstant=*/false,
      LLVM::Linkage::Internal, name,
      /*value=*/Attribute(),
      /*alignment=*/16, /*addrSpace=*/3);
}

class LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalAllocOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto memdescTy = op.getType();
    auto shape = memdescTy.getShape();
    auto elemTy = getTypeConverter()->convertType(
        memdescTy.getElementType());
    if (!elemTy) return failure();

    auto module = op->getParentOfType<ModuleOp>();
    auto globalOp = createTgGlobal(module, rewriter, loc, shape, elemTy);

    auto tgPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    Value basePtr = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy,
                                                globalOp.getSymName());

    // Initialized form (one operand): store init value at thread's index.
    // This is the per-thread model — each thread writes one element.
    if (op.getNumOperands() == 1 && adaptor.getOperands().size() == 1) {
      Value initVal = adaptor.getOperands()[0];
      auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
      auto lidFnTy = LLVM::LLVMFunctionType::get(i32Ty, {});
      auto lidFn = module.lookupSymbol<LLVM::LLVMFuncOp>(
          "__metal_get_local_id");
      if (!lidFn) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        lidFn = LLVM::LLVMFuncOp::create(rewriter, loc,
                                          "__metal_get_local_id", lidFnTy);
      }
      auto lid = LLVM::CallOp::create(rewriter, loc, lidFn, ValueRange{});
      auto arrTy = LLVM::LLVMArrayType::get(elemTy, shape[0]);
      Value zero = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
      Value slotPtr = LLVM::GEPOp::create(
          rewriter, loc, tgPtrTy, arrTy, basePtr,
          ValueRange{zero, lid.getResult()});
      LLVM::StoreOp::create(rewriter, loc, initVal, slotPtr);
    }

    rewriter.replaceOp(op, basePtr);
    return success();
  }
};

void populateSharedMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  // Convert !ttg.memdesc<...> to llvm.ptr in addrspace(3) (Metal threadgroup).
  typeConverter.addConversion(
      [](triton::gpu::MemDescType mdt) -> Type {
        return LLVM::LLVMPointerType::get(mdt.getContext(), /*addrspace=*/3);
      });
  patterns.add<LocalAllocOpConversion>(typeConverter);
}

} // namespace triton_metal
} // namespace mlir
