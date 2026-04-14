// ===-- ElementwiseOpToLLVM.cpp - Triton op → LLVM lowering patterns ------===//
//
// Conversion patterns for Triton IR ops to LLVM IR, targeting the Metal
// per-thread execution model.
//
// Key insight: in our 1D per-thread model each thread handles ONE element,
// so tensor<NxT> degenerates to scalar T. The LLVMTypeConverter is
// configured with a custom conversion for RankedTensorType that strips the
// tensor wrapper, returning just the element type.
//
// Metal builtins (program_id, local_id) are emitted as calls to external
// functions that will be resolved during MSL code generation.
//
// ===---------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton_metal {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get or insert an external function declaration into the parent module.
static LLVM::LLVMFuncOp getOrInsertFunction(ModuleOp module,
                                             ConversionPatternRewriter &rewriter,
                                             StringRef name,
                                             LLVM::LLVMFunctionType fnTy) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), name, fnTy);
}

// ---------------------------------------------------------------------------
// tt.get_program_id → call @__metal_get_program_id_{X,Y,Z}
// ---------------------------------------------------------------------------
struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertOpToLLVMPattern<triton::GetProgramIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {});

    // Pick the function name based on axis (X=0, Y=1, Z=2)
    unsigned axis = static_cast<unsigned>(op.getAxis());
    std::string fnName = "__metal_get_program_id_" + std::to_string(axis);

    auto module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertFunction(module, rewriter, fnName, fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, fn, ValueRange{});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.make_range → call @__metal_get_local_id  (+ start offset)
//
// In our per-thread model, make_range {start, end} returns
//     start + thread_position_in_threadgroup
// (one element per thread).
// ---------------------------------------------------------------------------
struct MakeRangeOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeRangeOp> {
  using ConvertOpToLLVMPattern<triton::MakeRangeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {});

    auto module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertFunction(module, rewriter, "__metal_get_local_id", fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, fn, ValueRange{});
    Value lid = call.getResult();

    // Add the start offset if non-zero
    uint32_t start = op.getStart();
    if (start != 0) {
      auto startConst = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(start));
      lid = LLVM::AddOp::create(rewriter, loc, i32Ty, lid, startConst);
    }

    rewriter.replaceOp(op, lid);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.splat → passthrough (scalar is already per-thread)
// ---------------------------------------------------------------------------
struct SplatOpConversion
    : public ConvertOpToLLVMPattern<triton::SplatOp> {
  using ConvertOpToLLVMPattern<triton::SplatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // In the per-thread model, splat is a no-op: the scalar value is already
    // the per-thread value. The type converter handles tensor<NxT> → T.
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.addptr → LLVM getelementptr
// ---------------------------------------------------------------------------
struct AddPtrOpConversion
    : public ConvertOpToLLVMPattern<triton::AddPtrOp> {
  using ConvertOpToLLVMPattern<triton::AddPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto resultTy = op.getType();

    // Resolve the element type through Triton's PointerType.
    // The result could be tensor<N x !tt.ptr<f32>> or just !tt.ptr<f32>.
    Type ptrElemTy;
    if (auto tensorTy = dyn_cast<RankedTensorType>(resultTy)) {
      auto ttPtrTy = cast<triton::PointerType>(tensorTy.getElementType());
      ptrElemTy = getTypeConverter()->convertType(ttPtrTy.getPointeeType());
    } else {
      auto ttPtrTy = cast<triton::PointerType>(resultTy);
      ptrElemTy = getTypeConverter()->convertType(ttPtrTy.getPointeeType());
    }

    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    Value result = LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrElemTy, ptr,
                                        offset);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.load → LLVM load (with optional mask → select)
// ---------------------------------------------------------------------------
struct LoadOpConversion
    : public ConvertOpToLLVMPattern<triton::LoadOp> {
  using ConvertOpToLLVMPattern<triton::LoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Determine the scalar element type for the load.
    Type origResultTy = op.getType();
    Type elemTy;
    if (auto tensorTy = dyn_cast<RankedTensorType>(origResultTy))
      elemTy = getTypeConverter()->convertType(tensorTy.getElementType());
    else
      elemTy = getTypeConverter()->convertType(origResultTy);

    Value ptr = adaptor.getPtr();
    Value loaded = LLVM::LoadOp::create(rewriter, loc, elemTy, ptr);

    // If there's a mask, use select: mask ? loaded : other
    Value mask = adaptor.getMask();
    if (mask) {
      Value other = adaptor.getOther();
      if (other) {
        loaded = LLVM::SelectOp::create(rewriter, loc, mask, loaded, other);
      }
      // If mask exists but no other value, we still use the loaded value.
      // In practice, Triton always provides 'other' when there's a mask.
    }

    rewriter.replaceOp(op, loaded);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.store → LLVM store (with optional mask → conditional via select/nop)
//
// For simplicity in the nano backend, we emit an unconditional store.
// A production backend would emit a conditional branch around the store.
// ---------------------------------------------------------------------------
struct StoreOpConversion
    : public ConvertOpToLLVMPattern<triton::StoreOp> {
  using ConvertOpToLLVMPattern<triton::StoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value ptr = adaptor.getPtr();
    Value value = adaptor.getValue();
    Value mask = adaptor.getMask();

    if (mask) {
      // Conditional store: create if/then around the store.
      // For the nano backend, use a simple LLVM conditional branch pattern:
      //   if (mask) store(ptr, value)
      //
      // We implement this with an scf-style block split. However, since we're
      // already in LLVM dialect territory, we use LLVM::CondBrOp.
      auto *currentBlock = rewriter.getInsertionBlock();
      auto *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      auto *storeBlock =
          rewriter.createBlock(afterBlock);

      // Current block: conditional branch
      rewriter.setInsertionPointToEnd(currentBlock);
      LLVM::CondBrOp::create(rewriter, loc, mask, storeBlock, afterBlock);

      // Store block: do the store, then branch to after
      rewriter.setInsertionPointToStart(storeBlock);
      LLVM::StoreOp::create(rewriter, loc, value, ptr);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBlock);

      // Continue at afterBlock
      rewriter.setInsertionPointToStart(afterBlock);
    } else {
      // Unconditional store
      LLVM::StoreOp::create(rewriter, loc, value, ptr);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.func → llvm.func
//
// Converts Triton's function op to an LLVM function. This is needed because
// the partial conversion will mark tt.func as illegal. We convert the
// function signature (all types through the type converter) and move the body.
// ---------------------------------------------------------------------------
struct FuncOpConversion
    : public ConvertOpToLLVMPattern<triton::FuncOp> {
  using ConvertOpToLLVMPattern<triton::FuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *typeConverter = getTypeConverter();

    // Convert the function signature
    TypeConverter::SignatureConversion signatureConversion(
        op.getNumArguments());
    auto funcType = op.getFunctionType();

    // Convert argument types
    SmallVector<Type> convertedArgTypes;
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      Type converted = typeConverter->convertType(funcType.getInput(i));
      if (!converted)
        return failure();
      convertedArgTypes.push_back(converted);
      signatureConversion.addInputs(i, converted);
    }

    // Convert result types
    SmallVector<Type> convertedResultTypes;
    for (Type resTy : funcType.getResults()) {
      Type converted = typeConverter->convertType(resTy);
      if (!converted)
        return failure();
      convertedResultTypes.push_back(converted);
    }

    auto llvmFuncType = LLVM::LLVMFunctionType::get(
        convertedResultTypes.empty()
            ? LLVM::LLVMVoidType::get(rewriter.getContext())
            : convertedResultTypes.front(),
        convertedArgTypes);

    auto newFunc = LLVM::LLVMFuncOp::create(rewriter, loc,
                                              op.getName(), llvmFuncType);

    // Copy over any attributes we want to preserve
    if (op->hasAttr("sym_visibility"))
      newFunc->setAttr("sym_visibility", op->getAttr("sym_visibility"));

    // Move the function body
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(),
                                newFunc.end());
    if (failed(rewriter.convertRegionTypes(&newFunc.getBody(), *typeConverter,
                                           &signatureConversion)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.broadcast → passthrough (per-thread model: tensor shapes are irrelevant)
// ---------------------------------------------------------------------------
struct BroadcastOpConversion
    : public ConvertOpToLLVMPattern<triton::BroadcastOp> {
  using ConvertOpToLLVMPattern<triton::BroadcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // In the per-thread model, broadcast is a no-op: the scalar is already
    // the per-thread value. The type converter handles tensor<NxT> → T.
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.expand_dims → passthrough (per-thread model: dimension changes are no-ops)
// ---------------------------------------------------------------------------
struct ExpandDimsOpConversion
    : public ConvertOpToLLVMPattern<triton::ExpandDimsOp> {
  using ConvertOpToLLVMPattern<triton::ExpandDimsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.reshape → passthrough (per-thread model: reshape is a no-op on scalars)
// ---------------------------------------------------------------------------
struct ReshapeOpConversion
    : public ConvertOpToLLVMPattern<triton::ReshapeOp> {
  using ConvertOpToLLVMPattern<triton::ReshapeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.extern_elementwise → LLVM intrinsic for known math functions
//
// Triton's frontend emits tt.extern_elementwise for libdevice calls like
// __nv_tanhf, __nv_sinf, etc. We map these to standard LLVM intrinsics
// which Metal's compiler understands.
// ---------------------------------------------------------------------------
struct ExternElementwiseOpConversion
    : public ConvertOpToLLVMPattern<triton::ExternElementwiseOp> {
  using ConvertOpToLLVMPattern<triton::ExternElementwiseOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto symbol = op.getSymbol();

    // Map NVIDIA libdevice symbols to Metal AIR intrinsic names.
    // Metal's GPU runtime recognizes air.fast_* as built-in math functions.
    // Using these instead of llvm.* intrinsics ensures compatibility at
    // both compilation and metallib linking stages.
    static const llvm::StringMap<llvm::StringRef> nvToAIR = {
        {"__nv_tanhf", "air.fast_tanh.f32"},
        {"__nv_tanf", "air.fast_tan.f32"},
        {"__nv_sinf", "air.fast_sin.f32"},
        {"__nv_cosf", "air.fast_cos.f32"},
        {"__nv_expf", "air.fast_exp.f32"},
        {"__nv_exp2f", "air.fast_exp2.f32"},
        {"__nv_logf", "air.fast_log.f32"},
        {"__nv_log2f", "air.fast_log2.f32"},
        {"__nv_sqrtf", "air.fast_sqrt.f32"},
        {"__nv_fabsf", "air.fast_fabs.f32"},
        {"__nv_floorf", "air.fast_floor.f32"},
        {"__nv_ceilf", "air.fast_ceil.f32"},
        {"__nv_roundf", "air.fast_round.f32"},
        {"__nv_fmaf", "air.fast_fma.f32"},
        {"__nv_powf", "air.fast_pow.f32"},
        {"__nv_fmaxf", "air.fast_fmax.f32"},
        {"__nv_fminf", "air.fast_fmin.f32"},
        {"__nv_copysignf", "air.fast_copysign.f32"},
    };

    auto it = nvToAIR.find(symbol);
    if (it == nvToAIR.end()) {
      return rewriter.notifyMatchFailure(
          op, "unknown extern_elementwise symbol: " + symbol.str());
    }

    // Get the converted result type
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    // Build the LLVM intrinsic function type
    SmallVector<Type> argTypes;
    for (auto operand : adaptor.getSrcs())
      argTypes.push_back(operand.getType());

    auto fnTy = LLVM::LLVMFunctionType::get(resultTy, argTypes);

    // Get or insert the intrinsic declaration
    auto module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertFunction(module, rewriter, it->second, fnTy);

    // Call the intrinsic
    SmallVector<Value> args(adaptor.getSrcs().begin(),
                            adaptor.getSrcs().end());
    auto call = LLVM::CallOp::create(rewriter, loc, fn, args);
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.get_num_programs → call @__metal_get_num_programs_{0,1,2}
// ---------------------------------------------------------------------------
struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto fnTy = LLVM::LLVMFunctionType::get(i32Ty, {});

    unsigned axis = static_cast<unsigned>(op.getAxis());
    std::string fnName = "__metal_get_num_programs_" + std::to_string(axis);

    auto module = op->getParentOfType<ModuleOp>();
    auto fn = getOrInsertFunction(module, rewriter, fnName, fnTy);

    auto call = LLVM::CallOp::create(rewriter, loc, fn, ValueRange{});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// ---------------------------------------------------------------------------
// tt.return → llvm.return
// ---------------------------------------------------------------------------
struct ReturnOpConversion
    : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::ReturnOp::create(rewriter, op->getLoc(), adaptor.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

// Pattern for arith.constant with tensor types → scalar LLVM constant.
// Standard arith-to-LLVM doesn't know our tensor→scalar mapping.
class ArithConstantOpConversion
    : public ConvertOpToLLVMPattern<mlir::arith::ConstantOp> {
public:
  using ConvertOpToLLVMPattern<mlir::arith::ConstantOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return failure();
    auto value = op.getValue();
    // For dense tensor constants, extract the scalar splat value
    if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(value)) {
      if (denseAttr.isSplat()) {
        auto splatVal = denseAttr.getSplatValue<Attribute>();
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, splatVal);
        return success();
      }
      // Non-splat: use first element (per-thread model)
      auto firstVal = *denseAttr.value_begin<Attribute>();
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, firstVal);
      return success();
    }
    // Scalar constants pass through
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, value);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
void populateTritonMetalToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();

  // Register custom type conversion for Triton's tensor types.
  // In our per-thread model, tensor<NxT> → T (scalar element type).
  typeConverter.addConversion([&typeConverter](RankedTensorType tensorTy) -> Type {
    return typeConverter.convertType(tensorTy.getElementType());
  });

  // Register custom type conversion for Triton's pointer type.
  // tt.ptr<T> → llvm.ptr (opaque pointer)
  typeConverter.addConversion([](triton::PointerType ptrTy) -> Type {
    return LLVM::LLVMPointerType::get(ptrTy.getContext());
  });

  // Add all conversion patterns
  // Higher benefit (10) so our pattern beats arith-to-LLVM's constant pattern
  // for tensor-typed constants (dense<0.0> : tensor<256xf32> → scalar 0.0f)
  patterns.add<ArithConstantOpConversion>(typeConverter, /*benefit=*/10);
  patterns.add<GetProgramIdOpConversion>(typeConverter);
  patterns.add<GetNumProgramsOpConversion>(typeConverter);
  patterns.add<MakeRangeOpConversion>(typeConverter);
  patterns.add<SplatOpConversion>(typeConverter);
  patterns.add<BroadcastOpConversion>(typeConverter);
  patterns.add<ExpandDimsOpConversion>(typeConverter);
  patterns.add<ReshapeOpConversion>(typeConverter);
  patterns.add<ExternElementwiseOpConversion>(typeConverter);
  patterns.add<AddPtrOpConversion>(typeConverter);
  patterns.add<LoadOpConversion>(typeConverter);
  patterns.add<StoreOpConversion>(typeConverter);
  patterns.add<FuncOpConversion>(typeConverter);
  patterns.add<ReturnOpConversion>(typeConverter);
}

} // namespace triton_metal
} // namespace mlir
