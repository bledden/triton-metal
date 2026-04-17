// ===-- SharedMemoryVectorizePass.cpp - coalesce shared memory ops --====//
//
// Post-lowering LLVM IR pass: when consecutive scalar loads/stores on
// addrspace(3) pointers have contiguous indices and matching types,
// combine into a vector op.
//
// Opportunistic: if preconditions aren't met, scalar ops remain correct.
//
// ===-----------------------------------------------------------------===//

#include "triton_metal/Conversion/TritonMetalToLLVM.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"

#include <vector>

namespace mlir {
namespace triton_metal {

// Coalesce 4 consecutive scalar loads on addrspace(3) with contiguous GEP
// indices into a single vector load. Same for stores.
void vectorizeSharedMemoryAccess(llvm::Module &mod) {
  for (auto &F : mod) {
    if (F.isDeclaration()) continue;
    std::vector<llvm::LoadInst *> loadsToErase;
    for (auto &BB : F) {
      std::vector<llvm::LoadInst *> candidates;
      for (auto &I : BB) {
        if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
          auto *ptrTy = LI->getPointerOperandType();
          if (ptrTy->getPointerAddressSpace() == 3) {
            candidates.push_back(LI);
            if (candidates.size() == 4) {
              // Check contiguity: all GEPs off same base with indices 0..3
              bool contiguous = true;
              auto *g0 = llvm::dyn_cast<llvm::GEPOperator>(
                  candidates[0]->getPointerOperand());
              if (!g0 || g0->getNumOperands() < 2) {
                contiguous = false;
              } else {
                auto *c0 = llvm::dyn_cast<llvm::ConstantInt>(
                    g0->idx_begin()->get());
                if (!c0) {
                  contiguous = false;
                }
              }

              for (unsigned i = 1; contiguous && i < 4; ++i) {
                auto *gi = llvm::dyn_cast<llvm::GEPOperator>(
                    candidates[i]->getPointerOperand());
                if (!gi || !g0 ||
                    g0->getPointerOperand() != gi->getPointerOperand() ||
                    gi->getNumOperands() != g0->getNumOperands()) {
                  contiguous = false;
                  break;
                }
                // Compare last index: must be i more than g0's
                auto *c0 = llvm::cast<llvm::ConstantInt>(
                    g0->idx_begin()->get());
                auto *ci = llvm::dyn_cast<llvm::ConstantInt>(
                    gi->idx_begin()->get());
                if (!ci ||
                    ci->getZExtValue() != c0->getZExtValue() + i) {
                  contiguous = false;
                  break;
                }
              }

              if (contiguous) {
                // Build vector load
                llvm::IRBuilder<> B(candidates[0]);
                auto *scalarTy = candidates[0]->getType();
                auto *vecTy = llvm::FixedVectorType::get(scalarTy, 4);
                auto *basePtr = candidates[0]->getPointerOperand();
                auto *vecLoad = B.CreateLoad(vecTy, basePtr);
                for (unsigned i = 0; i < 4; ++i) {
                  auto *elem = B.CreateExtractElement(vecLoad, i);
                  candidates[i]->replaceAllUsesWith(elem);
                  loadsToErase.push_back(candidates[i]);
                }
              }
              candidates.clear();
            }
          } else {
            candidates.clear();
          }
        } else {
          candidates.clear();
        }
      }
    }
    for (auto *LI : loadsToErase) LI->eraseFromParent();
  }
}

} // namespace triton_metal
} // namespace mlir
