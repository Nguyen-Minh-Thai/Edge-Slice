//===- LivenessRefinedDealloc.cpp - sub-block ownership deallocation ------===//
//
// `--ownership-based-buffer-deallocation` emits one merged
// `bufferization.dealloc` per block, at the block's exit. That is by design:
// `bufferization.dealloc` is set-valued and its ownership predicates are
// resolved per block, which is what makes it sound across arbitrary control
// flow.
//
// A bufferized DNN forward pass, however, is a single basic block. Block
// granularity plus one block means every buffer lives until the function
// returns, so the peak equals the sum of every allocation. Measured on
// ResNet-18 at 224x224 after one-shot bufferization: 58 allocations, zero
// `cf.br`/`cf.cond_br`, and exactly one `bufferization.dealloc` covering all
// 58.
//
// This pass refines that placement without leaving the ownership model. For
// each (memref, condition) pair of a merged dealloc it asks whether the pair
// can be discharged early, and extracts it only when every one of these holds:
//
//   1. the ownership condition is a statically known `true`;
//   2. the operand traces to a `memref.alloc` in the same block;
//   3. the complete forward alias closure of that allocation is derivable
//      through operations whose aliasing behaviour we model -- an unmodelled
//      one is a refusal, never an assumption;
//   4. no value in that closure is retained by the dealloc;
//   5. every use of every alias maps to an ancestor operation inside the
//      block, and the last of them is not the terminator;
//   6. no alias is already deallocated.
//
// Within a single block with no control flow there is no join point at which
// ownership could differ, so under 1-6 the buffer is dead after its last use
// on every path -- there is only one path. Everything else stays in the merged
// block-exit dealloc, so the pass degrades to today's behaviour exactly where
// today's behaviour is the reason the ownership model is sound.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// Why a (memref, condition) pair was left in the block-exit dealloc. Every
/// refusal is named so that a run can be audited pair by pair rather than
/// summarised as a count.
enum class Refusal {
  Sunk,
  ConditionNotStaticallyTrue,
  OperandIsNotAnAllocation,
  AllocationInAnotherBlock,
  RetainedAliasExists,
  AliasUsedOutsideBlock,
  UnmodelledAliasingUser,
  CallMayCaptureBuffer,
  LastUseIsTerminator,
  AlreadyDeallocated,
  DeallocPrecedesLastUse,
  UndeclaredMemoryEffects,
};

static StringRef refusalName(Refusal r) {
  switch (r) {
  case Refusal::Sunk:
    return "sunk";
  case Refusal::ConditionNotStaticallyTrue:
    return "condition-not-statically-true";
  case Refusal::OperandIsNotAnAllocation:
    return "operand-is-not-an-allocation";
  case Refusal::AllocationInAnotherBlock:
    return "allocation-in-another-block";
  case Refusal::RetainedAliasExists:
    return "retained-alias-exists";
  case Refusal::AliasUsedOutsideBlock:
    return "alias-used-outside-block";
  case Refusal::UnmodelledAliasingUser:
    return "unmodelled-aliasing-user";
  case Refusal::CallMayCaptureBuffer:
    return "call-may-capture-buffer";
  case Refusal::LastUseIsTerminator:
    return "last-use-is-terminator";
  case Refusal::AlreadyDeallocated:
    return "already-deallocated";
  case Refusal::DeallocPrecedesLastUse:
    return "dealloc-precedes-last-use";
  case Refusal::UndeclaredMemoryEffects:
    return "undeclared-memory-effects";
  }
  return "unknown";
}

/// Static byte size of a memref with a fully known shape, or 0 when any extent
/// or the element width is not known. Reporting 0 rather than a guess keeps an
/// unknown out of the totals instead of inventing one.
static int64_t staticByteSize(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref || !memref.hasStaticShape())
    return 0;
  Type elem = memref.getElementType();
  if (!elem.isIntOrFloat())
    return 0;
  int64_t bits = elem.getIntOrFloatBitWidth();
  if (bits % 8 != 0)
    return 0;
  int64_t bytes = bits / 8;
  for (int64_t dim : memref.getShape())
    bytes *= dim;
  return bytes;
}

/// Collects every value that may alias `root`, walking forward only through
/// operations whose aliasing behaviour is modelled here. An operation that
/// yields a memref we cannot attribute is a refusal, not an assumption.
///
/// `root` is always the result of a `memref.alloc`, which is a fresh
/// allocation and therefore terminal in the backward direction: the forward
/// closure is the whole alias set.
static bool collectAliases(Value root, Operation *ownerDealloc,
                           llvm::SetVector<Value> &out, Refusal &why) {
  SmallVector<Value> worklist{root};
  out.insert(root);
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();

      // A deallocation consumes the buffer and yields no alias. A *second*
      // merged dealloc naming the same buffer is another matter: discharging
      // the pair from one of them would leave the other still holding it, so
      // refuse rather than risk the double free.
      if (isa<bufferization::DeallocOp>(user)) {
        if (user != ownerDealloc) {
          why = Refusal::AlreadyDeallocated;
          return false;
        }
        continue;
      }
      if (isa<memref::DeallocOp>(user)) {
        // Placing a second free would be a double free.
        why = Refusal::AlreadyDeallocated;
        return false;
      }

      // Views: every memref result is an alias of the source.
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() == value) {
          for (Value result : user->getResults())
            if (isa<BaseMemRefType>(result.getType()) && out.insert(result))
              worklist.push_back(result);
          continue;
        }
      }

      // `memref.extract_strided_metadata` hands out the base buffer; the other
      // results are indices and carry no aliasing.
      if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(user)) {
        Value base = metadata.getBaseBuffer();
        if (out.insert(base))
          worklist.push_back(base);
        continue;
      }

      // A callee may store the pointer for later.
      if (isa<CallOpInterface>(user)) {
        why = Refusal::CallMayCaptureBuffer;
        return false;
      }

      // An operation that declares no memory effects at all tells us nothing
      // about what it does with the buffer -- it could store the descriptor
      // somewhere that outlives the block. Everything the bufferized payloads
      // here contain declares them, either directly or recursively, so this
      // costs nothing on them and refuses on anything unfamiliar.
      if (!isa<MemoryEffectOpInterface>(user) &&
          !user->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
        why = Refusal::UndeclaredMemoryEffects;
        return false;
      }

      // Anything else is a plain consumer only if it cannot pass the buffer
      // onward. `scf.if`/`scf.for` yielding a memref land here and refuse.
      for (Value result : user->getResults()) {
        if (isa<BaseMemRefType>(result.getType())) {
          why = Refusal::UnmodelledAliasingUser;
          return false;
        }
      }
    }
  }
  return true;
}

struct PairDecision {
  Refusal outcome;
  int64_t bytes = 0;
  /// Number of operations the free moved earlier; zero when refused.
  int64_t movedBy = 0;
};

struct LivenessRefinedDeallocPass
    : public PassWrapper<LivenessRefinedDeallocPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LivenessRefinedDeallocPass)

  LivenessRefinedDeallocPass() = default;
  LivenessRefinedDeallocPass(const LivenessRefinedDeallocPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final {
    return "edgeslide-liveness-refined-dealloc";
  }
  StringRef getDescription() const final {
    return "Discharge statically-owned buffers from the block-exit "
           "bufferization.dealloc at their last use";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, bufferization::BufferizationDialect,
                    arith::ArithDialect, func::FuncDialect>();
  }

  Option<std::string> reportFile{
      *this, "report",
      llvm::cl::desc("Write a per-pair JSON decision record to this path"),
      llvm::cl::init("")};

  void runOnOperation() override;
};

void LivenessRefinedDeallocPass::runOnOperation() {
  Operation *scope = getOperation();
  llvm::json::Array records;

  SmallVector<bufferization::DeallocOp> deallocs;
  scope->walk([&](bufferization::DeallocOp op) { deallocs.push_back(op); });

  for (bufferization::DeallocOp dealloc : deallocs) {
    Block *block = dealloc->getBlock();
    auto enclosing = dealloc->getParentOfType<func::FuncOp>();
    std::string where = enclosing ? enclosing.getName().str() : "<unknown>";
    SmallVector<Value> keptMemrefs, keptConditions;
    SmallVector<std::pair<Value, Operation *>> toSink;

    for (auto pair : llvm::zip(dealloc.getMemrefs(), dealloc.getConditions())) {
      Value memref = std::get<0>(pair);
      Value condition = std::get<1>(pair);
      PairDecision decision{Refusal::ConditionNotStaticallyTrue};

      auto keep = [&]() {
        keptMemrefs.push_back(memref);
        keptConditions.push_back(condition);
      };

      // (1) ownership must be statically held.
      if (!matchPattern(condition, m_One())) {
        records.push_back(llvm::json::Object{
            {"decision", refusalName(decision.outcome)}, {"bytes", 0}, {"function", where}});
        keep();
        continue;
      }

      // (2) the operand must trace to an allocation in this block.
      Value root = memref;
      if (auto metadata = memref.getDefiningOp<memref::ExtractStridedMetadataOp>())
        root = metadata.getSource();
      auto alloc = root.getDefiningOp<memref::AllocOp>();
      if (!alloc) {
        decision.outcome = Refusal::OperandIsNotAnAllocation;
        records.push_back(llvm::json::Object{
            {"decision", refusalName(decision.outcome)}, {"bytes", 0}, {"function", where}});
        keep();
        continue;
      }
      decision.bytes = staticByteSize(root.getType());
      if (alloc->getBlock() != block) {
        decision.outcome = Refusal::AllocationInAnotherBlock;
        records.push_back(
            llvm::json::Object{{"decision", refusalName(decision.outcome)},
                               {"bytes", decision.bytes},
                               {"function", where}});
        keep();
        continue;
      }

      // (3) the alias closure must be fully derivable.
      llvm::SetVector<Value> aliases;
      Refusal why = Refusal::Sunk;
      if (!collectAliases(root, dealloc, aliases, why)) {
        records.push_back(llvm::json::Object{{"decision", refusalName(why)},
                                             {"bytes", decision.bytes},
                                             {"function", where}});
        keep();
        continue;
      }

      // (4) nothing in the closure may be retained.
      bool retained = false;
      for (Value value : dealloc.getRetained())
        if (aliases.contains(value))
          retained = true;
      if (retained) {
        records.push_back(
            llvm::json::Object{{"decision", refusalName(Refusal::RetainedAliasExists)},
                               {"bytes", decision.bytes},
                               {"function", where}});
        keep();
        continue;
      }

      // (5) every use must live inside this block; find the last of them.
      Operation *lastUse = nullptr;
      bool escapes = false;
      for (Value value : aliases) {
        for (Operation *user : value.getUsers()) {
          if (user == dealloc.getOperation())
            continue;
          // An operation with no memory effects does not touch the buffer, so
          // it does not extend its liveness: `memref.subview`,
          // `memref.expand_shape` and `memref.extract_strided_metadata` are
          // arithmetic on the descriptor, not accesses through it. This
          // matters concretely -- the ownership pass emits an
          // `extract_strided_metadata` for every buffer immediately before the
          // block-exit dealloc, and treating those as uses would pin every
          // free back to the end of the block. The memrefs such an operation
          // derives are already in the alias closure, so any real access
          // through them is still counted.
          if (isMemoryEffectFree(user))
            continue;
          Operation *ancestor = block->findAncestorOpInBlock(*user);
          if (!ancestor) {
            escapes = true;
            break;
          }
          if (!lastUse || lastUse->isBeforeInBlock(ancestor))
            lastUse = ancestor;
        }
        if (escapes)
          break;
      }
      if (escapes) {
        records.push_back(
            llvm::json::Object{{"decision", refusalName(Refusal::AliasUsedOutsideBlock)},
                               {"bytes", decision.bytes},
                               {"function", where}});
        keep();
        continue;
      }
      // An allocation with no use at all is still freed, immediately.
      if (!lastUse)
        lastUse = alloc.getOperation();
      if (lastUse == block->getTerminator()) {
        records.push_back(
            llvm::json::Object{{"decision", refusalName(Refusal::LastUseIsTerminator)},
                               {"bytes", decision.bytes},
                               {"function", where}});
        keep();
        continue;
      }
      if (!lastUse->isBeforeInBlock(dealloc.getOperation())) {
        records.push_back(
            llvm::json::Object{{"decision", refusalName(Refusal::DeallocPrecedesLastUse)},
                               {"bytes", decision.bytes},
                               {"function", where}});
        keep();
        continue;
      }

      // Count how far the free travels, as evidence that it moved at all.
      int64_t distance = 0;
      for (Operation *op = lastUse->getNextNode(); op && op != dealloc.getOperation();
           op = op->getNextNode())
        ++distance;

      // Free the allocation itself rather than the base buffer: the base
      // buffer is produced by an `extract_strided_metadata` the ownership pass
      // placed next to the block-exit dealloc, so it does not dominate the
      // last use, while the allocation result dominates everything.
      toSink.push_back({root, lastUse});
      records.push_back(llvm::json::Object{{"decision", "sunk"},
                                           {"bytes", decision.bytes},
                                           {"moved_past_ops", distance},
                                           {"function", where}});
    }

    if (toSink.empty())
      continue;

    OpBuilder builder(dealloc);
    for (auto &entry : toSink) {
      builder.setInsertionPointAfter(entry.second);
      builder.create<memref::DeallocOp>(entry.first.getLoc(), entry.first);
    }

    // Rebuild the merged dealloc with the pairs that remain. The retained list
    // is unchanged, so the result count -- one updated condition per retained
    // memref -- is unchanged too, and no pair we removed aliased a retained
    // value, so none of those conditions changes value.
    builder.setInsertionPoint(dealloc);
    if (keptMemrefs.empty() && dealloc.getRetained().empty()) {
      dealloc.erase();
      continue;
    }
    auto replacement = builder.create<bufferization::DeallocOp>(
        dealloc.getLoc(), keptMemrefs, keptConditions, dealloc.getRetained());
    dealloc.replaceAllUsesWith(replacement.getUpdatedConditions());
    dealloc.erase();
  }

  if (!reportFile.empty()) {
    std::error_code error;
    // CD_CreateAlways: truncate. The default two-argument constructor writes
    // from offset zero without truncating, so a shorter second write leaves the
    // tail of the first behind and the file stops being JSON. That happens
    // whenever this pass is nested under `func.func(...)` in a module with more
    // than one function.
    llvm::raw_fd_ostream out(reportFile, error, llvm::sys::fs::CD_CreateAlways,
                             llvm::sys::fs::FA_Write, llvm::sys::fs::OF_Text);
    if (error) {
      scope->emitError() << "cannot write dealloc report to '" << reportFile
                         << "': " << error.message();
      return signalPassFailure();
    }
    llvm::json::Object root{{"pairs", std::move(records)}};
    out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
  }
}

} // namespace

extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "EdgeSlideDealloc", "0.1", []() {
            ::mlir::PassRegistration<LivenessRefinedDeallocPass>();
          }};
}
