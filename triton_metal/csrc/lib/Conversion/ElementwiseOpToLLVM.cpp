#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace triton_metal {

void populateTritonMetalToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  // Patterns will be added in Task 3
}

} // namespace triton_metal
} // namespace mlir
