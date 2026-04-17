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

    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto aRegTy = LLVM::LLVMArrayType::get(aElem, 8);
    auto bRegTy = LLVM::LLVMArrayType::get(bElem, 8);
    auto cRegTy = LLVM::LLVMArrayType::get(cElem, 8);
    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
    auto i32Ty = IntegerType::get(ctx, 32);
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
    auto storeFnOp = getOrInsertFn(storeFn,
        LLVM::LLVMVoidType::get(ctx), {tgPtrTy, cRegTy, i64Ty});

    // Allocate output tile in threadgroup memory.
    // sharedMemoryCounter in SharedMemoryOpToLLVM.cpp is static there; we use
    // our own counter so names don't collide with that pool.
    static unsigned dotOutCounter = 0;
    auto outArrTy = LLVM::LLVMArrayType::get(cElem, M * N);
    std::string outName = "__tg_dot_out_" + std::to_string(dotOutCounter++);
    LLVM::GlobalOp outGlobal;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      outGlobal = LLVM::GlobalOp::create(
          rewriter, loc, outArrTy, /*isConstant=*/false,
          LLVM::Linkage::Internal, outName,
          /*value=*/Attribute(),
          /*alignment=*/16, /*addrSpace=*/3);
    }
    Value outBasePtr = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy,
                                                   outGlobal.getSymName());

    // Get A and B base pointers. In TTGIR, tt.dot's operands are tensor
    // values produced by ttg.local_load on memdesc sources — but our type
    // converter maps tensor<NxT> -> T (scalar), so adaptor.getA() is an
    // f16 scalar, not a pointer. We need the underlying memdesc pointer.
    // Walk back through the defining ttg.local_load op and ask the rewriter
    // for the remapped (converted) source memdesc, which our SharedMemoryOp
    // conversion maps to ptr addrspace(3).
    auto getMemdescPtr = [&](Value tensorOperand) -> Value {
      auto localLoad = tensorOperand.getDefiningOp<triton::gpu::LocalLoadOp>();
      if (!localLoad) return nullptr;
      return rewriter.getRemappedValue(localLoad.getSrc());
    };
    Value aBasePtr = getMemdescPtr(op.getA());
    Value bBasePtr = getMemdescPtr(op.getB());
    if (!aBasePtr || !bBasePtr)
      return rewriter.notifyMatchFailure(op,
          "tt.dot operands must come from ttg.local_load");

    int64_t tilesM = M / 8, tilesN = N / 8, tilesK = K / 8;
    Value strideK = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(K));
    Value strideN = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(N));

    // Unrolled tile loops
    for (int64_t mi = 0; mi < tilesM; ++mi) {
      for (int64_t ni = 0; ni < tilesN; ++ni) {
        // Initialize C accumulator to zero
        Value cElemZero = LLVM::ConstantOp::create(
            rewriter, loc, cElem, rewriter.getZeroAttr(cElem));
        Value acc = LLVM::UndefOp::create(rewriter, loc, cRegTy);
        for (int64_t i = 0; i < 8; ++i) {
          acc = LLVM::InsertValueOp::create(
              rewriter, loc, acc, cElemZero, ArrayRef<int64_t>{i});
        }

        for (int64_t ki = 0; ki < tilesK; ++ki) {
          // A tile at (mi*8, ki*8) — offset = mi*8*K + ki*8
          int64_t aOffset = mi * 8 * K + ki * 8;
          Value aOffsetC = LLVM::ConstantOp::create(
              rewriter, loc, i32Ty,
              rewriter.getI32IntegerAttr(aOffset));
          Value aTilePtr = LLVM::GEPOp::create(
              rewriter, loc, tgPtrTy, aElem, aBasePtr,
              ValueRange{aOffsetC});
          Value aMat = LLVM::CallOp::create(
              rewriter, loc, loadAFnOp,
              ValueRange{aTilePtr, strideK}).getResult();

          // B tile at (ki*8, ni*8) — offset = ki*8*N + ni*8
          int64_t bOffset = ki * 8 * N + ni * 8;
          Value bOffsetC = LLVM::ConstantOp::create(
              rewriter, loc, i32Ty,
              rewriter.getI32IntegerAttr(bOffset));
          Value bTilePtr = LLVM::GEPOp::create(
              rewriter, loc, tgPtrTy, bElem, bBasePtr,
              ValueRange{bOffsetC});
          Value bMat = LLVM::CallOp::create(
              rewriter, loc, loadBFnOp,
              ValueRange{bTilePtr, strideN}).getResult();

          // MMA: acc = A * B + acc
          acc = LLVM::CallOp::create(
              rewriter, loc, mmaFnOp,
              ValueRange{aMat, bMat, acc}).getResult();
        }

        // Store C tile at (mi*8, ni*8) — offset = mi*8*N + ni*8
        int64_t cOffset = mi * 8 * N + ni * 8;
        Value cOffsetC = LLVM::ConstantOp::create(
            rewriter, loc, i32Ty,
            rewriter.getI32IntegerAttr(cOffset));
        Value cTilePtr = LLVM::GEPOp::create(
            rewriter, loc, tgPtrTy, cElem, outBasePtr,
            ValueRange{cOffsetC});
        LLVM::CallOp::create(rewriter, loc, storeFnOp,
                               ValueRange{cTilePtr, acc, strideN});
      }
    }

    // Barrier so all threads see the result
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto barrierFn = getOrInsertFn("air.wg.barrier", voidTy, {i32Ty, i32Ty});
    Value two = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(2));
    Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, barrierFn, ValueRange{two, one});

    // Each thread reads its own element from the output tile.
    // Per-thread model: thread's lid maps to (row, col) in the M×N output.
    // lid ranges over [0, M*N), so each thread reads outBasePtr[lid].
    auto lidFn = getOrInsertFn("__metal_get_local_id", i32Ty, {});
    auto lid = LLVM::CallOp::create(rewriter, loc, lidFn, ValueRange{});
    Value cResultPtr = LLVM::GEPOp::create(
        rewriter, loc, tgPtrTy, cElem, outBasePtr,
        ValueRange{lid.getResult()});
    Value cResult = LLVM::LoadOp::create(rewriter, loc, cElem, cResultPtr);

    rewriter.replaceOp(op, cResult);
    return success();
  }
};

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.add<DotOpConversion>(typeConverter);
}

} // namespace triton_metal
} // namespace mlir
