// ===-- SharedMemoryAliasingPass.cpp - reuse shared memory --------====//
//
// Post-lowering LLVM IR pass: coalesce addrspace(3) globals when their
// live ranges don't overlap. Simple greedy graph coloring.
//
// Runs after MLIR -> LLVM IR translation, before typed-pointer conversion.
//
// ===------------------------------------------------------------===//

#include "triton_metal/Conversion/TritonMetalToLLVM.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>
#include <set>
#include <map>
#include <algorithm>

namespace mlir {
namespace triton_metal {

// Runs on an LLVM Module. Coalesces addrspace(3) globals where live ranges
// don't overlap. Call after LLVM IR generation, before typed-ptr conversion.
void aliasSharedMemoryGlobals(llvm::Module &mod) {
  // Collect all addrspace(3) globals that we generated.
  llvm::SmallVector<llvm::GlobalVariable *, 8> tgGlobals;
  for (auto &G : mod.globals()) {
    if (G.getAddressSpace() != 3) continue;
    auto name = G.getName();
    if (!name.starts_with("__tg_shared_") &&
        !name.starts_with("__reduce_shared_") &&
        !name.starts_with("__tg_dot_out_")) continue;
    tgGlobals.push_back(&G);
  }
  if (tgGlobals.size() < 2) return;

  // For each function, compute liveness and build interference graph.
  for (auto &F : mod) {
    if (F.isDeclaration()) continue;

    // Number instructions linearly.
    std::map<llvm::GlobalVariable *, std::pair<int, int>> liveRanges;
    int instrIdx = 0;
    for (auto &BB : F) {
      for (auto &I : BB) {
        for (auto &Op : I.operands()) {
          if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(Op.get())) {
            if (std::find(tgGlobals.begin(), tgGlobals.end(), GV)
                != tgGlobals.end()) {
              auto &range = liveRanges[GV];
              if (range.first == 0 && range.second == 0) {
                range.first = instrIdx;
                range.second = instrIdx;
              } else {
                range.second = instrIdx;
              }
            }
          }
        }
        instrIdx++;
      }
    }

    if (liveRanges.size() < 2) continue;

    // Build interference graph.
    std::vector<llvm::GlobalVariable *> globals;
    for (auto &kv : liveRanges) globals.push_back(kv.first);
    int n = globals.size();
    std::vector<std::set<int>> adj(n);
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        auto &ri = liveRanges[globals[i]];
        auto &rj = liveRanges[globals[j]];
        // Overlap: [ri.first, ri.second] intersects [rj.first, rj.second]
        if (ri.first <= rj.second && rj.first <= ri.second) {
          adj[i].insert(j);
          adj[j].insert(i);
        }
      }
    }

    // Greedy color by size (largest first).
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      auto sizeA = mod.getDataLayout().getTypeAllocSize(
          globals[a]->getValueType());
      auto sizeB = mod.getDataLayout().getTypeAllocSize(
          globals[b]->getValueType());
      return sizeA > sizeB;
    });

    std::vector<int> color(n, -1);
    // For each color, track the index of the member with the largest
    // allocation size. We reuse that member's value type as the merged
    // global's type so existing GEPs remain type-consistent (e.g. all
    // members are `[32 x float]` -> merged stays `[32 x float]`). This
    // avoids downstream type mismatches in the typed-pointer post-pass.
    std::map<int, int> colorRepIdx;
    std::map<int, uint64_t> colorSize;
    for (int idx : order) {
      std::set<int> used;
      for (int nb : adj[idx]) {
        if (color[nb] != -1) used.insert(color[nb]);
      }
      int c = 0;
      while (used.count(c)) c++;
      color[idx] = c;
      auto sz = mod.getDataLayout().getTypeAllocSize(
          globals[idx]->getValueType());
      auto it = colorSize.find(c);
      if (it == colorSize.end() || sz > it->second) {
        colorSize[c] = sz;
        colorRepIdx[c] = idx;
      }
    }

    // Count unique colors; if all colors are unique (no aliasing possible),
    // skip to avoid churning the IR.
    std::set<int> uniqueColors(color.begin(), color.end());
    if ((int)uniqueColors.size() == n) continue;

    // Create merged globals, one per color. Use the representative member's
    // value type (it is the largest in the group), so the merged global's
    // declared type matches the GEP element types already in the IR.
    std::map<int, llvm::GlobalVariable *> colorToMerged;
    for (auto &[c, sz] : colorSize) {
      auto *rep = globals[colorRepIdx[c]];
      auto *valTy = rep->getValueType();
      std::string name = "__tg_merged_" + std::to_string(c);
      auto *merged = new llvm::GlobalVariable(
          mod, valTy, /*isConstant=*/false,
          llvm::GlobalValue::InternalLinkage,
          llvm::UndefValue::get(valTy), name, /*InsertBefore=*/nullptr,
          llvm::GlobalValue::NotThreadLocal, /*AddressSpace=*/3);
      merged->setAlignment(llvm::MaybeAlign(16));
      colorToMerged[c] = merged;
    }

    // Replace original globals with the merged global (or a bitcast to the
    // original's pointer type if their value types differ). Pointer types
    // in opaque-pointer mode are identical (just `ptr addrspace(3)`), so
    // `replaceAllUsesWith` usually works directly; the bitcast fallback
    // keeps us correct if a legacy typed-pointer consumer sneaks in.
    for (int i = 0; i < n; ++i) {
      auto *orig = globals[i];
      auto *merged = colorToMerged[color[i]];
      if (orig->getType() == merged->getType()) {
        orig->replaceAllUsesWith(merged);
      } else {
        orig->replaceAllUsesWith(
            llvm::ConstantExpr::getBitCast(merged, orig->getType()));
      }
      orig->eraseFromParent();
    }
  }
}

} // namespace triton_metal
} // namespace mlir
