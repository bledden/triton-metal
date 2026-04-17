// ===-- DotOpToLLVM.cpp - tt.dot -> simdgroup MMA lowering ------------===//
//
// Lowers tt.dot to Metal simdgroup_matrix_multiply_accumulate intrinsics.
// 8x8 MMA is the only size AIR supports; larger matmuls tile over 8x8 blocks
// (tiling is Task 12; this file implements the single-tile case).
//
// ===------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton_metal {

// AIR intrinsic name selectors.
static StringRef getSimdLoadIntrinsic(Type elemTy) {
  if (elemTy.isF16()) return "air.simdgroup_load_indirect_matrix_8x8.f16";
  if (elemTy.isBF16()) return "air.simdgroup_load_indirect_matrix_8x8.bf16";
  if (elemTy.isF32()) return "air.simdgroup_load_indirect_matrix_8x8.f32";
  return "";
}

static StringRef getSimdStoreIntrinsic(Type elemTy) {
  if (elemTy.isF32()) return "air.simdgroup_store_indirect_matrix_8x8.f32";
  if (elemTy.isF16()) return "air.simdgroup_store_indirect_matrix_8x8.f16";
  return "";
}

static StringRef getSimdMmaIntrinsic(Type aTy, Type bTy, Type cTy) {
  if (aTy.isF16() && bTy.isF16() && cTy.isF32())
    return "air.simdgroup_matrix_multiply_accumulate_8x8.f16.f32";
  if (aTy.isBF16() && bTy.isBF16() && cTy.isF32())
    return "air.simdgroup_matrix_multiply_accumulate_8x8.bf16.f32";
  if (aTy.isF32() && bTy.isF32() && cTy.isF32())
    return "air.simdgroup_matrix_multiply_accumulate_8x8.f32.f32";
  return "";
}

class DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
public:
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      triton::DotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto aTy = cast<RankedTensorType>(op.getA().getType());
    auto bTy = cast<RankedTensorType>(op.getB().getType());
    auto cTy = cast<RankedTensorType>(op.getC().getType());

    int64_t M = aTy.getShape()[0];
    int64_t K = aTy.getShape()[1];
    int64_t N = bTy.getShape()[1];

    if (M % 8 != 0 || N % 8 != 0 || K % 8 != 0)
      return rewriter.notifyMatchFailure(op, "tt.dot requires 8-aligned shapes");

    Type aElem = aTy.getElementType();
    Type bElem = bTy.getElementType();
    Type cElem = cTy.getElementType();

    StringRef loadAFn = getSimdLoadIntrinsic(aElem);
    StringRef loadBFn = getSimdLoadIntrinsic(bElem);
    StringRef mmaFn = getSimdMmaIntrinsic(aElem, bElem, cElem);
    StringRef storeFn = getSimdStoreIntrinsic(cElem);
    if (loadAFn.empty() || mmaFn.empty() || storeFn.empty())
      return rewriter.notifyMatchFailure(op, "unsupported dot element types");

    // For this task: only support M=K=N=8 (single MMA block).
    // Larger tiles added in Task 12.
    if (M != 8 || K != 8 || N != 8)
      return rewriter.notifyMatchFailure(op,
                                          "only 8x8x8 supported in Task 11");

    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto aRegTy = LLVM::LLVMArrayType::get(aElem, 8);
    auto bRegTy = LLVM::LLVMArrayType::get(bElem, 8);
    auto cRegTy = LLVM::LLVMArrayType::get(cElem, 8);
    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
    auto i64Ty = IntegerType::get(ctx, 64);

    auto getOrInsertFn = [&](StringRef name, Type retTy,
                              ArrayRef<Type> argTys) {
      auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
      if (!fn) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto fnTy = LLVM::LLVMFunctionType::get(retTy, argTys);
        fn = LLVM::LLVMFuncOp::create(rewriter, loc, name, fnTy);
      }
      return fn;
    };

    auto loadAFnOp = getOrInsertFn(loadAFn, aRegTy, {tgPtrTy, i64Ty});
    auto loadBFnOp = getOrInsertFn(loadBFn, bRegTy, {tgPtrTy, i64Ty});
    auto mmaFnOp = getOrInsertFn(mmaFn, cRegTy, {aRegTy, bRegTy, cRegTy});

    // Get A, B pointers (memdesc → ptr addrspace(3))
    Value aPtr = adaptor.getA();
    Value bPtr = adaptor.getB();
    // C is the accumulator — it's a scalar in our per-thread model.
    // Build an 8-element register from the scalar value.
    Value cScalar = adaptor.getC();

    // Build C register from scalar (broadcast across all 8 elements)
    Value cReg = LLVM::UndefOp::create(rewriter, loc, cRegTy);
    for (int i = 0; i < 8; ++i) {
      cReg = LLVM::InsertValueOp::create(
          rewriter, loc, cReg, cScalar, ArrayRef<int64_t>{i});
    }

    // Load A and B tiles (stride = column count K or N)
    Value strideK = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(K));
    Value strideN = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(N));

    Value aMat = LLVM::CallOp::create(rewriter, loc, loadAFnOp,
                                        ValueRange{aPtr, strideK}).getResult();
    Value bMat = LLVM::CallOp::create(rewriter, loc, loadBFnOp,
                                        ValueRange{bPtr, strideN}).getResult();

    // MMA: D = A * B + C
    Value dMat = LLVM::CallOp::create(rewriter, loc, mmaFnOp,
                                        ValueRange{aMat, bMat, cReg}).getResult();

    // Extract a scalar result (per-thread model: each thread owns one element)
    // For now, extract element 0. In the tiled case (Task 12), we write back to
    // threadgroup memory and each thread reads its own element.
    Value dScalar = LLVM::ExtractValueOp::create(
        rewriter, loc, dMat, ArrayRef<int64_t>{0});

    rewriter.replaceOp(op, dScalar);
    return success();
  }
};

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.add<DotOpConversion>(typeConverter);
}

} // namespace triton_metal
} // namespace mlir
