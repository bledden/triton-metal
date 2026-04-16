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

class LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) return failure();

    Value srcPtr = adaptor.getSrc();

    // Per-thread model: each thread loads element at its lid
    // (unless a subview has already narrowed the pointer — subview
    // pattern adjusts the base accordingly)
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto module = op->getParentOfType<ModuleOp>();
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

    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
    Value slotPtr = LLVM::GEPOp::create(
        rewriter, loc, tgPtrTy, resultTy, srcPtr,
        ValueRange{lid.getResult()});

    Value loaded = LLVM::LoadOp::create(rewriter, loc, resultTy, slotPtr);
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

class LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value srcVal = adaptor.getSrc();
    Value dstPtr = adaptor.getDst();

    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto module = op->getParentOfType<ModuleOp>();
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

    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
    Value slotPtr = LLVM::GEPOp::create(
        rewriter, loc, tgPtrTy, srcVal.getType(), dstPtr,
        ValueRange{lid.getResult()});

    LLVM::StoreOp::create(rewriter, loc, srcVal, slotPtr);
    rewriter.eraseOp(op);
    return success();
  }
};

class LocalDeallocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalDeallocOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalDeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::LocalDeallocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Metal threadgroup memory is function-scoped; no dealloc needed.
    rewriter.eraseOp(op);
    return success();
  }
};

class AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto module = op->getParentOfType<ModuleOp>();

    auto barrierFnTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty});
    auto barrierFn = module.lookupSymbol<LLVM::LLVMFuncOp>("air.wg.barrier");
    if (!barrierFn) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      barrierFn = LLVM::LLVMFuncOp::create(rewriter, loc,
                                            "air.wg.barrier", barrierFnTy);
    }

    Value two = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(2));
    Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, barrierFn, ValueRange{two, one});
    rewriter.eraseOp(op);
    return success();
  }
};

class MemDescSubsliceOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescSubsliceOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescSubsliceOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::MemDescSubsliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto srcTy = op.getSrc().getType();
    auto srcShape = srcTy.getShape();
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    if (!elemTy) return failure();

    // Offsets are a static DenseI32Array attribute. Compute the linear
    // offset at compile time (row-major):
    //   offset = sum(idx[i] * prod(shape[i+1..]))
    auto offsets = op.getOffsets();
    int64_t linearOffset = 0;
    for (unsigned i = 0; i < offsets.size(); ++i) {
      int64_t stride = 1;
      for (unsigned j = i + 1; j < srcShape.size(); ++j)
        stride *= srcShape[j];
      linearOffset += static_cast<int64_t>(offsets[i]) * stride;
    }

    Value offsetVal = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(linearOffset)));

    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
    Value subPtr = LLVM::GEPOp::create(
        rewriter, loc, tgPtrTy, elemTy, adaptor.getSrc(),
        ValueRange{offsetVal});

    rewriter.replaceOp(op, subPtr);
    return success();
  }
};

class MemDescTransOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescTransOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescTransOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::gpu::MemDescTransOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // No data movement. The result type's order attribute signals
    // transposed access to downstream consumers (tt.dot handles it).
    rewriter.replaceOp(op, adaptor.getSrc());
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
  patterns.add<LocalAllocOpConversion,
               LocalLoadOpConversion,
               LocalStoreOpConversion,
               LocalDeallocOpConversion,
               AsyncWaitOpConversion,
               MemDescSubsliceOpConversion,
               MemDescTransOpConversion>(typeConverter);
}

} // namespace triton_metal
} // namespace mlir
