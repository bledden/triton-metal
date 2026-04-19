// ===-- DotOpToLLVM.cpp - tt.dot -> simdgroup MMA lowering ------------===//
//
// Lowers tt.dot to Apple AIR simdgroup_matrix_8x8 intrinsics, using the
// validated signatures obtained by disassembling MSL-compiled probes:
//
//   %m = call <64 x T> @air.simdgroup_matrix_8x8_init_filled.v64*.T(T)
//   %a = call <64 x T> @air.simdgroup_matrix_8x8_load.v64*.p3T(
//            T addrspace(3)*, i64 stride, <2 x i64> origin, i1 transpose)
//   %d = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate
//                             .v64f32.v64T.v64T.v64f32(<64 x T>, <64 x T>,
//                                                      <64 x float>)
//   call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(
//            <64 x float>, float addrspace(3)*, i64, <2 x i64>, i1)
//
// Matrices are <64 x T> vectors (8x8 = 64 elements) distributed across a
// simdgroup, NOT [8 x T] arrays. Opaque pointer declarations are emitted
// here and rewritten to typed pointers by _opaque_to_typed_ptrs on the
// Python side (it inspects the .pNT suffix of the intrinsic name).
//
// For the per-thread scalar model the tile result is spilled to a
// threadgroup global, barriered, then read back at the calling thread's lid.
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

    // Only f16*f16->f32 and f32*f32->f32 supported currently.
    bool isF16 = aElem.isF16() && bElem.isF16() && cElem.isF32();
    bool isF32 = aElem.isF32() && bElem.isF32() && cElem.isF32();
    if (!isF16 && !isF32)
      return rewriter.notifyMatchFailure(op, "unsupported dot element types");

    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();

    // LLVM / MLIR primitive types.
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto vec2I64 = VectorType::get({2}, i64Ty);
    auto aMatTy = VectorType::get({64}, aElem);
    auto bMatTy = VectorType::get({64}, bElem);
    auto cMatTy = VectorType::get({64}, cElem);
    auto tgPtrTy = LLVM::LLVMPointerType::get(ctx, 3);

    // Intrinsic name suffixes (based on the validated signatures).
    const char *aSuffix = isF16 ? "v64f16" : "v64f32";
    const char *bSuffix = aSuffix;       // same as A in both supported mixes
    const char *cSuffix = "v64f32";      // accumulator is f32 in both cases
    const char *pSuffixA = isF16 ? "p3f16" : "p3f32";
    const char *pSuffixB = pSuffixA;

    std::string loadAName =
        std::string("air.simdgroup_matrix_8x8_load.") + aSuffix + "." + pSuffixA;
    std::string loadBName =
        std::string("air.simdgroup_matrix_8x8_load.") + bSuffix + "." + pSuffixB;
    std::string initName =
        "air.simdgroup_matrix_8x8_init_filled.v64f32.f32";
    std::string mmaName =
        std::string("air.simdgroup_matrix_8x8_multiply_accumulate.") +
        cSuffix + "." + aSuffix + "." + bSuffix + "." + cSuffix;
    std::string storeName =
        "air.simdgroup_matrix_8x8_store.v64f32.p3f32";

    auto getOrInsertFn = [&](StringRef name, Type retTy,
                              ArrayRef<Type> argTys) -> LLVM::LLVMFuncOp {
      auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
      if (!fn) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto fnTy = LLVM::LLVMFunctionType::get(retTy, argTys);
        fn = LLVM::LLVMFuncOp::create(rewriter, loc, name, fnTy);
      }
      return fn;
    };

    auto loadAFn = getOrInsertFn(loadAName, aMatTy,
                                  {tgPtrTy, i64Ty, vec2I64, i1Ty});
    auto loadBFn = getOrInsertFn(loadBName, bMatTy,
                                  {tgPtrTy, i64Ty, vec2I64, i1Ty});
    auto initFn = getOrInsertFn(initName, cMatTy, {cElem});
    auto mmaFn = getOrInsertFn(mmaName, cMatTy, {aMatTy, bMatTy, cMatTy});
    auto storeFn = getOrInsertFn(storeName,
                                  LLVM::LLVMVoidType::get(ctx),
                                  {cMatTy, tgPtrTy, i64Ty, vec2I64, i1Ty});

    // Per-thread model: spill the full MxN tile to a threadgroup buffer
    // so that each thread can read its own scalar element back at its lid.
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

    // Resolve A / B base pointers. In TTGIR the DotOp's operands are
    // tensor-typed and produced by ttg.local_load on a memdesc source.
    // Our type converter maps tensor -> scalar, so we can't take
    // adaptor.getA() directly as the tile base; we need the source memdesc,
    // which the SharedMemoryOp converter has lowered to ptr addrspace(3).
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

    // Barrier before we read A/B tiles: every thread's per-thread store to
    // @__tg_shared_* must be visible to the simdgroup before the 8x8 load
    // intrinsic runs. Triton's pipeliner usually inserts this via
    // ttg.async_wait, but when the path from `tl.load` straight into
    // `tt.dot` bypasses pipelining we synthesize it here for safety.
    auto preBarrierFn = module.lookupSymbol<LLVM::LLVMFuncOp>("air.wg.barrier");
    if (!preBarrierFn) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto barrierTy = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(ctx), {i32Ty, i32Ty});
      preBarrierFn = LLVM::LLVMFuncOp::create(rewriter, loc,
                                                "air.wg.barrier", barrierTy);
    }
    Value preTwo = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getI32IntegerAttr(2));
    Value preOne = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, preBarrierFn,
                          ValueRange{preTwo, preOne});

    Value strideK = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(K));
    Value strideN = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(N));
    Value zeroI64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(0));
    Value idx0 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));
    Value idx1 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(1));
    Value zeroOffset = LLVM::UndefOp::create(rewriter, loc, vec2I64);
    zeroOffset = LLVM::InsertElementOp::create(
        rewriter, loc, zeroOffset, zeroI64, idx0);
    zeroOffset = LLVM::InsertElementOp::create(
        rewriter, loc, zeroOffset, zeroI64, idx1);
    Value falseBool = LLVM::ConstantOp::create(rewriter, loc, i1Ty,
                                                rewriter.getBoolAttr(false));
    Value cZero = LLVM::ConstantOp::create(rewriter, loc, cElem,
                                            rewriter.getZeroAttr(cElem));

    int64_t tilesM = M / 8, tilesN = N / 8, tilesK = K / 8;
    for (int64_t mi = 0; mi < tilesM; ++mi) {
      for (int64_t ni = 0; ni < tilesN; ++ni) {
        Value acc = LLVM::CallOp::create(rewriter, loc, initFn,
                                           ValueRange{cZero}).getResult();

        for (int64_t ki = 0; ki < tilesK; ++ki) {
          // A tile at (mi*8, ki*8); row-major offset = mi*8*K + ki*8
          int64_t aOffset = mi * 8 * K + ki * 8;
          Value aOffsetC = LLVM::ConstantOp::create(
              rewriter, loc, i32Ty,
              rewriter.getI32IntegerAttr(aOffset));
          Value aTilePtr = LLVM::GEPOp::create(
              rewriter, loc, tgPtrTy, aElem, aBasePtr,
              ValueRange{aOffsetC});
          Value aMat = LLVM::CallOp::create(
              rewriter, loc, loadAFn,
              ValueRange{aTilePtr, strideK, zeroOffset, falseBool}).getResult();

          // B tile at (ki*8, ni*8); row-major offset = ki*8*N + ni*8
          int64_t bOffset = ki * 8 * N + ni * 8;
          Value bOffsetC = LLVM::ConstantOp::create(
              rewriter, loc, i32Ty,
              rewriter.getI32IntegerAttr(bOffset));
          Value bTilePtr = LLVM::GEPOp::create(
              rewriter, loc, tgPtrTy, bElem, bBasePtr,
              ValueRange{bOffsetC});
          Value bMat = LLVM::CallOp::create(
              rewriter, loc, loadBFn,
              ValueRange{bTilePtr, strideN, zeroOffset, falseBool}).getResult();

          // MMA: acc = A * B + acc
          acc = LLVM::CallOp::create(
              rewriter, loc, mmaFn,
              ValueRange{aMat, bMat, acc}).getResult();
        }

        // Store C tile at (mi*8, ni*8); row-major offset = mi*8*N + ni*8
        int64_t cOffset = mi * 8 * N + ni * 8;
        Value cOffsetC = LLVM::ConstantOp::create(
            rewriter, loc, i32Ty,
            rewriter.getI32IntegerAttr(cOffset));
        Value cTilePtr = LLVM::GEPOp::create(
            rewriter, loc, tgPtrTy, cElem, outBasePtr,
            ValueRange{cOffsetC});
        LLVM::CallOp::create(rewriter, loc, storeFn,
                               ValueRange{acc, cTilePtr, strideN,
                                          zeroOffset, falseBool});
      }
    }

    // Make the spilled tile visible to every thread in the threadgroup.
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto barrierFn = getOrInsertFn("air.wg.barrier", voidTy, {i32Ty, i32Ty});
    Value two = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(2));
    Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, barrierFn, ValueRange{two, one});

    // Per-thread model: each thread reads outBasePtr[lid].
    // Use the array-typed GEP form (`[M*N x cElem]`) so that the Python
    // opaque-to-typed post-pass rewrites it to
    //   getelementptr [M*N x float], [M*N x float] addrspace(3)* @g, i32 0, i32 lid
    // which Metal's typed-pointer mode accepts.
    auto lidFn = getOrInsertFn("__metal_get_local_id", i32Ty, {});
    auto lid = LLVM::CallOp::create(rewriter, loc, lidFn, ValueRange{});
    Value zeroIdx = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0));
    Value cResultPtr = LLVM::GEPOp::create(
        rewriter, loc, tgPtrTy, outArrTy, outBasePtr,
        ValueRange{zeroIdx, lid.getResult()});
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
