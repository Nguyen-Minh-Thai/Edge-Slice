# Research Proposal

## Title

**EdgeSlice: SRAM-Resident Operator Scheduling for Fast Edge Inference via MLIR Transform Dialect**

---

## Abstract

Deep learning inference on edge accelerators is often limited by movement between DRAM and a small explicitly managed on-chip SRAM. This proposal introduces **EdgeSlice**, an MLIR scheduling pass whose hard constraint is that the complete working set of each tiled or fused compute region—inputs, weights, halos, accumulators, temporaries, and output tile—fits a declared SRAM capacity. Among capacity-feasible schedules, EdgeSlice minimizes estimated off-chip traffic and uses target measurements to select the fastest schedule. DAG liveness remains diagnostic but no longer gates tiling: an operation is a candidate whenever its local untiled working set exceeds SRAM, even if it is not at the graph's global memory peak. Standard MLIR Transform dialect operations express tiling and fusion; explicit promotion, memory-space assignment, copy scheduling, double buffering, and target-specific inner-kernel lowering make residency executable rather than merely annotated. Evaluation prioritizes measured latency and DRAM↔SRAM bytes, with peak RSS reported only as context.

**Keywords:** MLIR, Transform Dialect, SRAM, Scratchpad Memory, Tensor Tiling, Data Movement, Edge Inference, Compiler Optimization

---

## 1. Introduction & Problem Statement

### 1.1 Background

Deep learning models deployed on **edge devices** face a fundamental memory bottleneck. Modern CNNs produce large intermediate tensors during inference — activations, feature maps, and attention outputs — that can easily exceed the limited on-chip memory (L1/SRAM) available on edge hardware. When active working sets exceed cache capacity, execution increasingly depends on lower cache levels and **DRAM**, which are slower and more energy-intensive, causing severe performance degradation.

This problem is particularly acute in domains processing large inputs:
- **Autonomous driving:** Multi-camera streams at high resolution (e.g., 1600×900)
- **Edge vision:** Real-time object detection on high-resolution images
- **Multimodal inference:** Large batch-equivalent dimensions from sensor fusion

Two complementary compiler-level strategies address memory pressure:

1. **Tensor Slicing for Memory Reduction:** Splitting large tensors into smaller tiles that reduce the active cache working set. Sousa et al. (2023) demonstrated this with TSO for multicore NPUs (21.7% speedup). Ferrari et al. (2023) introduced SConv with Convolution Slicing Analysis (CSA) for cache-blocking (9–42% speedup over Im2Col+BLAS).

2. **Declarative Transform Dialect Rewriting:** Lücke et al. (2024) formalized the MLIR Transform dialect as a controllable IR-based transformation system, enabling fine-grained, composable compiler optimizations without rebuilding the compiler. Ferrari et al. (2025) extended this with SConvTransform.

### 1.2 Problem Statement

Despite these advances, a critical gap remains: **no existing approach combines DAG-level static analysis (liveness, peak memory estimation) with declarative MLIR Transform dialect rewriting to automatically slice large intermediate tensors for memory-constrained edge inference.**

Specifically:
- **SConv / SConvTransform** operate on individual convolution ops without global DAG context — no liveness analysis, no peak memory estimation across the full model graph.
- **TSO** is tightly coupled to TensorFlow XLA/LLVM and specific NPU architectures, limiting portability.
- **Neither approach** targets the frontend compiler scope: accepting PyTorch models, analyzing the DAG, and outputting optimized MLIR IR for further lowering.
- **The Transform Dialect** (Lücke et al., 2024) provides the infrastructure for composable transformations but has not been applied to automatic tensor slicing guided by DAG analysis.

### 1.3 Motivation & Scope

**Problem context:** When running deep learning models with large inputs on edge devices, intermediate working sets exceed cache capacity, increasing lower-cache/DRAM traffic and slowing execution.

**Goal:** Produce the fastest capacity-legal schedule for a target SRAM by minimizing off-chip data movement and exposing an efficient inner kernel.

**Scope:**
- Parse deep learning models into DAG graphs
- Optimize on the DAG by slicing large tensors and reassembling tiles at appropriate points
- Accept compile-time overhead in exchange for runtime performance improvement
- **Out of scope:** Pre-slicing graph optimizations (operator fusion, constant folding, etc.)

**Input/Output:**
- **Input:** PyTorch model file (`.pt` / `.pth`)
- **Output:** Optimized MLIR file (sliced IR, ready for further lowering passes)

### 1.4 Research Questions

- **RQ1:** Can a whole-working-set model guarantee that each scheduled compute region fits the target SRAM?
- **RQ2:** Which tile shape, dataflow, and fusion boundary minimize DRAM↔SRAM traffic under that capacity constraint?
- **RQ3:** Can MLIR tiling, fusion, promotion, memory spaces, and copy scheduling materialize the selected SRAM schedule without semantic loss?
- **RQ4:** How closely do predicted traffic and reuse explain measured target latency, stalls, and energy?

---

## 2. Literature Review

### 2.1 MLIR and the Transform Dialect

MLIR (Multi-Level Intermediate Representation) (Lattner et al., 2020) has emerged as the dominant compiler infrastructure for ML workloads, providing a hierarchy of dialects from high-level model representations down to target-specific code generation.

The **Transform Dialect** (Lücke et al., 2024; arXiv:2409.03864v2) formalizes a controllable IR-based transformation system that addresses a fundamental limitation of modern compilers: coarse-grained pass pipelines that do not allow fine-grained control over optimization decisions. Key contributions include:

- **Separation of concerns:** Transform scripts are expressed as first-class IR operations, separate from the payload program.
- **Handle-based chaining:** Each transform produces handles (SSA values) referencing payload IR operations, enabling safe chaining and static detection of handle invalidation ("use after free").
- **Pre- and post-conditions:** Transforms declare which operations they consume and produce, enabling static detection of phase ordering violations.
- **Extensibility:** New transform operations can be defined without modifying the compiler core.

The Transform Dialect has been evaluated across five case studies demonstrating low overhead, robustness, and integration with search methods. However, its application to automatic tensor slicing guided by DAG-level memory analysis remains unexplored.

### 2.2 SConv: Direct Convolution with Cache-Blocking Analysis

Ferrari et al. (2023; arXiv:2303.04739v1) proposed **SConv**, a direct-convolution algorithm based on MLIR/LLVM that replaces the traditional Im2Col+BLAS approach. Three key components:

- **Convolution Slicing Analysis (CSA):** A convolution-specific 3D cache-blocking analysis pass that determines tile sizes and scheduling strategy (Input Stationary vs. Weight Stationary) based on cache hierarchy constraints.
- **Convolution Slicing Optimization (CSO):** A code-generation pass that uses CSA parameters to generate a tiled direct-convolution macro-kernel.
- **Vector-Based Packing (VBP):** Architecture-specific input tensor packing using vector-register shift instructions for unitary-stride convolutions.

SConv achieves 9–25% speedup on Intel x86 and 10–42% on IBM POWER10 for end-to-end model inference. It outperforms BLAS GEMM on pointwise convolutions in >83% of 219 tested instances.

**Limitation:** SConv operates on individual convolutions without global DAG context. It does not perform liveness analysis or peak memory estimation across the full model graph.

### 2.3 SConvTransform: Declarative Sliced Convolution in MLIR

Ferrari et al. (2025; arXiv:2511.18222v1) ported SConv to the MLIR Transform dialect as **SConvTransform**, replacing imperative rewrite patterns with declarative `transform.structured.sconv` operations. Key improvements:

- Fully declarative transformation pipeline within the Transform dialect
- Modular stages: edge case handling, slicing, tiling, packing, and affine map rewriting
- Achieves 60% of peak performance on ARM SME and 67% on Intel AVX512

**Limitation:** Like SConv, operates per-op without DAG-level analysis. Does not address intermediate tensor memory pressure across the full model graph.

### 2.4 Tensor Slicing Optimization (TSO) for Multicore NPUs

Sousa et al. (2023; arXiv:2304.03013v1) proposed **TSO**, a TensorFlow XLA/LLVM compiler pass for multicore NPUs:

- **Burst-Aware Slicing:** Uses DRAM memory burst time estimates to guide tensor slicing.
- **Core Parallelism Maximization:** Distributes sliced computations across NPU cores.
- 21.7% speedup over non-burst-aware slicing; validated on 32-core RISC-V NMP and ported to Glow.

**Limitation:** Tightly coupled to TensorFlow XLA/LLVM and specific NPU architectures. Does not leverage MLIR's Transform dialect.

### 2.5 Comparison Summary

| Approach | Framework | Key Technique | Limitation for Edge Memory |
|----------|-----------|---------------|---------------------------|
| SConv (Ferrari et al., 2023) | MLIR/LLVM | CSA cache-blocking + CSO macro-kernel | Per-op; no DAG analysis |
| SConvTransform (Ferrari et al., 2025) | MLIR Transform | Declarative Transform dialect + CSA | Per-op; no global memory analysis |
| TSO (Sousa et al., 2023) | TF XLA/LLVM | Burst-aware slicing for NPUs | Architecture-specific; not MLIR |
| TVM AutoTiling (Chen et al., 2018) | TVM | Autotuning-based tile size search | Requires search; no static DAG analysis |
| Transform Dialect (Lücke et al., 2024) | MLIR | Fine-grained IR-based transformation control | General framework; no memory-specific application |

### 2.6 Research Gap

The literature reveals a clear gap: **no existing system combines (a) DAG-level liveness and peak memory analysis, (b) automatic dimension-aware slicing decisions, and (c) declarative MLIR Transform dialect rewriting into a single frontend compiler pass targeting memory-constrained edge inference.**

EdgeSlice fills this gap by:
1. Operating at the DAG level to identify intermediate tensors that exceed L1 capacity and contribute most to peak memory pressure.
2. Selecting the optimal slicing dimension using a heuristic that prioritizes batch → spatial → channel dimensions.
3. Generating the sliced schedule entirely within the MLIR Transform dialect, ensuring composability with downstream lowering passes.
4. Accepting compile-time overhead as a deliberate trade-off for runtime memory reduction in ahead-of-time edge deployment scenarios.

---

## 3. Research Objectives

### 3.1 Primary Objective

Design, implement, and evaluate **EdgeSlice**, an MLIR scheduling pass that guarantees per-region SRAM feasibility and minimizes measured inference latency by reducing off-chip data movement.

### 3.2 Specific Objectives

1. **O1:** Implement a DAG parser that ingests PyTorch models via torch-mlir and constructs an operation-level dataflow graph.
2. **O2:** Model the complete simultaneous working set of each operator and fused region against a declared SRAM capacity.
3. **O3:** Search legal 1D/2D tile shapes and fusion boundaries, ranking them by predicted off-chip traffic, vector alignment, and parallelism.
4. **O4:** Materialize tiling plus explicit SRAM allocation, promotion, copy scheduling, and optional double buffering in MLIR.
5. **O5:** Evaluate on target hardware using latency, DRAM↔SRAM bytes, stalls, utilization, energy, numerical equivalence, and capacity violations.

### 3.3 Scope Definition

**In scope:**
- Frontend compiler optimization (PyTorch → MLIR DAG → Sliced MLIR)
- Liveness analysis and peak memory estimation on the DAG
- Automatic tensor slicing for ops exceeding L1 threshold
- Declarative Transform dialect rewriting
- Compile-time overhead accepted as deliberate trade-off

**Out of scope:**
- Pre-slicing graph optimizations (operator fusion, constant folding, dead code elimination)
- Backend-specific code generation (target-dependent scheduling)
- Dynamic shape support (initial version restricted to statically-shaped models)

---

 ## 4. Proposed Methodology

### 4.1 System Architecture Overview
![System Architecture](conv2d_graph.png)




### 4.2 Stage 1: DAG Construction

**Input:** PyTorch model (`.pt` / `.pth`)

**Process:**
1. Parse the model using `torch-mlir` to obtain Torch dialect IR.
2. Lower Torch dialect to Linalg dialect via standard `torch-to-linalg` conversion passes.
3. Traverse the Linalg IR to construct a directed acyclic graph (DAG) where:
   - **Nodes** represent operations (`linalg.conv_2d`, `linalg.matmul`, `linalg.elementwise`, etc.)
   - **Edges** represent tensor dependencies (producer → consumer)
   - Each node stores: operation type, input/output tensor shapes, parameters (stride/padding/dilation for convolutions)

**Output:** Operation DAG with annotated tensor metadata.

### 4.3 Stage 2: SRAM Target & Local Working-Set Analysis

The target declares physical SRAM capacity, transfer alignment, vector width,
and whether ping-pong buffering is required. With double buffering, EdgeSlice
reserves two tile slots: one tile may use at most half the physical capacity.
CPU cache targets are modeled separately and cannot be reported as
SRAM-resident.

For each operation, EdgeSlice computes the simultaneous local working set:

```text
working_set(op, tile) = input tiles
                      + weight tiles
                      + other operand tiles
                      + output/accumulator tile
                      + halo and temporary storage
```

An operation becomes a candidate when its **untiled local working set** exceeds
the usable SRAM budget. This includes graph sinks and does not depend on the
global liveness peak. Graph liveness remains a diagnostic for whole-model
allocation behavior, but it is not the SRAM candidate gate.

### 4.4 Stage 3: Capacity-Constrained Schedule Search

**Tileability Check:**
```
is_tileable(op):
    return op.type in {conv2d, matmul, elementwise}
       AND all dimensions of output tensor are statically known
       AND output tensor rank >= 2
```

**Schedule search:**
```
legal = []
for tile_shape, dataflow, fusion_boundary in schedule_space(op):
    if complete_working_set(tile_shape, dataflow, fusion_boundary) <= usable_sram:
        legal.append(schedule)

rank(schedule) = (
    estimated_dram_sram_bytes,
    estimated_bank_conflicts,
    -vector_utilization,
    -parallelism,
)
shortlist = lowest_ranked(legal)
winner = minimum_measured_latency(shortlist, target)
```

The analytical model prunes illegal and obviously expensive schedules;
measured target latency selects the winner. Dimension order is only a search
hint. Fusion is accepted only while the combined live tiles remain within the
same SRAM budget and eliminate enough intermediate traffic to justify any
recomputation.

**Op-type refinement (evaluated as Ablation D, §5.4):** the priority order above
is stated for a *dense* convolution, where every output channel is a reduction
over *all* input channels, so slicing along C leaves the input tile unshrunk.
It is wrong for a depthwise/grouped convolution, where output channel *c*
depends only on input channel *c*: there, slicing along C requires no halo
*and* shrinks the input tile proportionally, so C should be ranked ahead of
H/W rather than used as a fallback. The implementation therefore branches on
op type — `dimension_selector.rank_dimension_indices` ranks channel first for
depthwise ops, and the tile-size cost model credits the corresponding input
shrink (`tile_size._estimate_input_tile_shape`).

This is a correction to a real defect, not a tuning knob: the dense-conv
assumption was previously applied to *every* conv, and on depthwise workloads
it selected spatial tiling that made locality strictly worse. Ablation D (§5.4)
isolates its effect; measured outcomes are reported there.

### 4.5 Stage 4: MLIR Tiling, SRAM Promotion & Execution

The rewrite pipeline uses standard Transform dialect decomposition via
`transform.structured.match` + `transform.structured.tile_using_forall`, then
promotes the selected operand and output tiles into the target SRAM memory
space. Tiling creates smaller views; it does **not** by itself place data in
SRAM. Explicit allocation, copy-in/copy-out, synchronization, and
memory-space-aware lowering are required.

**Pseudocode:**
```
for each SRAM candidate in DAG (topological order):
    schedule = search_capacity_legal_schedules(op, target)

        // Step 1: Slice the required input and output-init operands
        operand_slices = insert extract_slice(op.operands, dim, tile_size)

        // Step 2: Replace op with the selected tiled compute loop
        loop = insert scf.forall over sliced dimension
            with tile_size as step
            body:
                tile_result = apply op on dependence-complete operand_slices
                yield tile_result

        // Step 2b: Promote/copy the tile to the SRAM memory space.
        // A double-buffered target overlaps transfer of i+1 with compute of i.
        sram_operands = copy_in(operand_slices, memory_space=SRAM)
        tile_result = target_inner_kernel(sram_operands)
        copy_out(tile_result)

        // Step 3: Reassemble tiles via tensor.parallel_insert_slice inside scf.forall.in_parallel
        final_output = tensor.parallel_insert_slice(tile_result, destination, dim)

        // Step 4: Replace original op's uses with final_output
        replace_all_uses_with(op.output, final_output)
        erase op
```

**MLIR IR Example (conceptual):**
```mlir
// Before transformation:
%output = linalg.conv_2d_nchw_fchw
    ins(%input, %filter : tensor<1x32x152x152xf32>, tensor<256x32x3x3xf32>)
    outs(%init : tensor<1x256x75x75xf32>)
    -> tensor<1x256x75x75xf32>

// After EdgeSlice transformation (slicing on spatial dimension H):
%0 = scf.forall (%i) in (0 to 75 step 32) -> (tensor<1x256x32x75xf32>) {
  %slice_input = tensor.extract_slice %input[0, 0, %i*2, 0] [1, 32, 64, 152] [1, 1, 1, 1]
  %slice_init = tensor.extract_slice %init[0, 0, %i, 0] [1, 256, 32, 75] [1, 1, 1, 1]
  %tile = linalg.conv_2d_nchw_fchw ins(%slice_input, %filter : ...)
      outs(%slice_init : ...) -> ...
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %tile into %output_slice[%i] ...
  }
}
// scf.forall.in_parallel reassembles tiles with tensor.parallel_insert_slice
```

### 4.6 Transform Dialect Implementation Considerations

**Handle Management (Lücke et al., 2024):**
The Transform dialect uses SSA-based handles to track payload IR operations. Each transform invalidates its operand handles and returns new handles to results. EdgeSlice must carefully chain handles through the `extract_slice` → `scf.forall` → `tensor.parallel_insert_slice` sequence to avoid dangling references.

**Pre- and Post-Conditions:**
Following the Transform Dialect's pre/post-condition system, EdgeSlice declares:
- **Pre-condition:** Payload contains `linalg.*` operations with statically-known output shapes.
- **Post-condition:** Selected operations are replaced with `scf.forall` loop nests containing sliced `linalg.*` operations and `tensor.parallel_insert_slice` reassembly.

**Bufferization Interaction:**
EdgeSlice makes the SRAM decision at tensor level, then carries it across
bufferization with an explicit memory-space assignment. The backend must lower
promoted tiles to the target scratchpad allocation and DMA/copy mechanism.
`tensor.extract_slice` or `memref.subview` alone is only an aliasing view and
does not move data. The compiler verifies the post-bufferization SRAM
high-water mark against the declared capacity and rejects a schedule that
cannot preserve the planned placement.

### 4.7 Output Specification

**Output:** Optimized MLIR IR file containing:
- The original model structure with selected operations replaced by sliced equivalents
- Transform dialect annotations for downstream passes to understand the slicing decisions
- Ready for further lowering (bufferization, target-specific code generation)

The output file may contain either:
- **Single unified IR:** All operations in one module, with sliced and non-sliced ops coexisting
- **Multiple IR fragments:** Separate modules for sliced subgraphs (to be linked during lowering)

The exact format will be determined during implementation based on compatibility with downstream MLIR passes.

### 4.8 Correctness Argument

This section proves correctness of the analysis and rewrite under explicit
preconditions. It does not prove the empirical hypotheses in Section 6.

#### 4.8.1 Definitions and Preconditions

Let (G=(V,E)) be the operation DAG and let
\(\pi=(v_0,\ldots,v_{|V|-1})\) be the fixed topological execution order. For
each intermediate tensor (t), let (p(t)) be the index of its producer and
let (C(t)) be the indices of its consumers. Define

\[
\ell(t)=
\begin{cases}
\max C(t), & C(t)\neq\varnothing,\\
p(t), & C(t)=\varnothing,
\end{cases}
\qquad
L_i=\{t\mid p(t)\le i\le \ell(t)\}.
\]

The rewrite proof assumes:

1. (G) is acyclic and tensor shapes are static (or have valid conservative
   bounds).
2. The selected payload operation is pure and is legal for MLIR structured
   tiling; only parallel output dimensions are tiled. A reduction dimension is
   not distributed as an independent output tile.
3. Operand slices contain the compVlete dependence region for each output tile.
   For convolution this includes the halo
   \(h_{in}=(h_{out}-1)s_h+d_h(k_h-1)+1\), and analogously for width.
4. Tile size (T\) is a positive integer, the destination has the same shape
   and element type as the original result, and each output position is written
   by exactly one `tensor.parallel_insert_slice`.
5. Operations have no observable side effects other than producing their
   tensor results. Floating-point equivalence is interpreted using the same
   reduction order, or otherwise with the stated numerical tolerance.

#### 4.8.2 Correctness of Liveness and Peak Estimation

**Lemma 1 (exact live set for the fixed schedule).** At program point (i),
the analysis reports exactly (L_i).

**Proof.** If (t\in L_i), then its producer has executed and its last consumer
has not executed before point (i); therefore its value may still be needed and
must be live. Conversely, if (t\notin L_i), either (i<p(t)), so the tensor
does not yet exist, or (i>\ell(t)), so every consumer has executed and the
tensor can be released. Hence the reported interval test
\(p(t)\le i\le\ell(t)\) is both sound and complete. \(\square\)

**Corollary 1 (frontend peak).** If (s(t)) is the statically computed byte
size of (t), then

\[
M_i=\sum_{t\in L_i}s(t),\qquad M_{peak}=\max_i M_i
\]

is the exact peak of the proposal's pre-bufferization model for schedule
\(\pi\), where the sum runs over distinct *buffers* rather than distinct SSA
values.

That distinction is load-bearing rather than pedantic. Frontend IR is in
destination-passing style: `tensor.empty` produces an uninitialized destination
placeholder, and a `linalg` or `tensor.insert_slice` result is written *into*
its destination operand, so one activation appears as a chain of several SSA
values that bufferization will map onto a single allocation. Counting those
values independently inflates \(M_i\) — on U-Net at 512×512 it overstated the
peak by 30%, and it also let a single CSE'd `tensor.empty`, shared as the
destination of nine `linalg.generic` ops, hold a full-size buffer live across
143 program points. The estimator therefore coalesces a destination-passing
result with its destination when that destination has exactly one consumer;
where a destination is shared, bufferization must copy for all but one use, so
the buffers stay distinct and are counted separately. `tensor.empty` itself
contributes no bytes, since its allocation is attributed to the op that
materializes it.

This is not automatically the physical runtime peak: buffer reuse, alignment,
allocator metadata, constants, and backend workspaces may change it. When
bufferized MLIR is available, allocation-lifetime analysis and hardware
measurement are required to validate that external quantity.

#### 4.8.3 Coverage and Semantic Preservation of Tiling

Let the sliced output dimension have extent (D>0), and let (T>0). Define
\(q=\lceil D/T\rceil\) and

\[
I_j=[jT,\min((j+1)T,D)),\qquad 0\le j<q.
\]

**Lemma 2 (tile partition).** The intervals (I_0,\ldots,I_{q-1}) are pairwise
disjoint and their union is ([0,D)).

**Proof.** Consecutive intervals meet only at the boundary ((j+1)T), and the
half-open convention excludes that boundary from the left interval. For any
index (x\in[0,D)), (j=\lfloor x/T\rfloor) satisfies (0\le j<q) and
\(x\in I_j). Thus every output index belongs to exactly one tile. \(\square\)

**Theorem 1 (single-operation semantic preservation).** Under Preconditions
2-5, replacing a selected operation by `tile_using_forall`, sliced structured
computations, and `tensor.parallel_insert_slice` produces the same result tensor
as the original operation.

**Proof.** Choose an arbitrary output index (x). By Lemma 2, exactly one tile
contains (x). Structured tiling preserves the scalar operation at (x), and
Precondition 3 supplies every input element on which that scalar result depends;
therefore the tile computes the original value at (x). Preconditions 4 and
Lemma 2 ensure that the value is inserted once at the same destination index and
cannot be overwritten by another tile. Since this holds for every output index,
the reassembled tensor is elementwise equal to the original result. \(\square\)

**Theorem 2 (whole-DAG semantic preservation).** If every selected operation
satisfies Theorem 1 and rewrites are applied in topological order, the transformed
DAG has the same observable outputs as the original DAG.

**Proof.** Use induction over \(\pi\). Before the first operation, model inputs
are identical. Assume all operands of (v_i) equal their original values. If
\(v_i\) is unchanged, its result is unchanged; if it is tiled, Theorem 1 gives
the same result. Thus the induction hypothesis holds for (v_{i+1}), and in
particular for every graph output. \(\square\)

#### 4.8.4 Cache-Budget Property

Let (B\) be the usable cache budget and (F(T)) the estimated bytes for the
input tile, output tile, weights, alignment, and convolution halo. Assume
\(F(T)\) is monotone nondecreasing and (F(1)\le B\).

**Theorem 3 (tile fits the modeled budget).** Binary search for the largest
feasible tile followed by rounding down returns (T^*\ge1) with
\(F(T^*)\le B\).

**Proof.** The invariant is that every discarded candidate below the current
lower bound is feasible or has already been dominated by a larger feasible
candidate, while every candidate above the current upper bound is infeasible.
Monotonicity makes each binary-search update valid. On termination, `best` is
feasible. Power-of-two/vector rounding does not increase `best`; hence
monotonicity gives (F(T^*)\le F(\text{best})\le B\). \(\square\)

If (F(1)>B\), no positive tile can satisfy the modeled budget. The
implementation may still return tile size 1 to make progress, so it must report
this case rather than claim cache residency. Also, fitting the model does not
guarantee physical L1 residency in the presence of conflict misses or unrelated
runtime data.

For the frontend estimate only, replacing a full output contribution (S) by
a tile contribution (S_T\le S) changes the modeled memory at that point to
\(M'_i=M_i-S+S_T\le M_i\). This proves non-increase in that estimator, not a
30-50% physical-memory reduction; the magnitude remains an experimental result.

#### 4.8.5 Termination and Complexity

Kahn's algorithm removes one vertex per iteration and therefore terminates after
\(|V|\) removals or rejects a cycle. Liveness scans finite tensors and program
points. Tile search strictly shrinks a finite integer interval, and each
`scf.forall` has \(\lceil D/T\rceil\) iterations for (D,T>0). Thus every stage
terminates under the stated preconditions.

With (K) distinct intermediate tensors, the current interval-based liveness
implementation costs (O(|V|+|E|+|V|K)) time and (O(|V|K)) worst-case live-set
storage. An event-sweep representation can reduce the analysis term to
\(O(|V|+|E|+K)\) when full live sets are not materialized. A working-set tile
search costs (O(\log D)) footprint evaluations per selected operation.

#### 4.8.6 What Is and Is Not Proven

The arguments above establish conditional functional correctness, termination,
and correctness of the stated analytical estimators. H1-H4, runtime speedup,
hardware cache misses, physical peak memory, and numerical tolerance across
different lowering/reassociation choices remain empirical questions and must be
established by the evaluation in Section 5.

---

## 5. Data Analysis Plan

### 5.1 Evaluation Workloads

**Model Selection:** Representative CNN models chosen for both large intermediate
tensors and architectural diversity in the per-operation dimension-selection
heuristic (§4.4): dense convolutions (ResNet-50) consume every input channel
per output channel, whereas depthwise-separable convolutions (MobileNetV2,
EfficientNet-B0) do not, which changes whether tiling along the channel axis
or the spatial (H/W) axis dominates for a given operator — see the analysis
accompanying Ablation D (§5.4).
- **ResNet-50:** Deep stack of dense convolutions with large early-stage feature maps
- **MobileNetV2:** Depthwise-separable convolutions (inverted residuals) throughout —
  the architectural counterpoint to ResNet-50's dense convs for the channel-vs-spatial
  tiling question
- **EfficientNet-B0:** Depthwise-separable convolutions plus a per-block
  Squeeze-and-Excitation gate, whose global average pool cannot be tiled along
  H/W in isolation — a case dimension-selection alone does not cover
- **U-Net (4-stage encoder-decoder):** Large-input segmentation model (§5.1
  resolutions up to 1600×900) whose multi-scale skip connections keep encoder
  feature maps live across the entire decoder pass, exercising liveness-driven
  slicing rather than dimension selection

An earlier revision of this plan used a YOLOv8-style detector for the
multi-scale/large-spatial-input role. It was dropped: the only implementation
available in the constrained `.venv-mlir` environment was a random-initialization
stand-in (`ultralytics` does not import there), so its accuracy as a benchmark
workload could not be verified against real YOLOv8 weights or topology.

**Input Configurations:**
- Varying input resolutions: 512×512, 1024×1024, 1600×900
- Batch sizes: 1, 2, 4
- Target L1 sizes: 256 KB, 512 KB, 1 MB

### 5.2 Metrics

| Metric | Description |
|--------|-------------|
| **Peak Memory Reduction (%)** | Reduction in maximum live tensor memory vs. baseline |
| **Compile-Time Overhead (s)** | Additional time for DAG analysis + Transform rewrite |
| **Runtime Speedup (%)** | Execution time improvement on target hardware |
| **Memory Traffic (GB)** | Total bytes transferred between L1/L2/DRAM |
| **Numerical Equivalence** | `numpy.allclose` comparison between sliced MLIR output and PyTorch baseline |
| **Hardware Cache Counters** | Linux `perf stat` L1/LLC cache references and misses where permissions allow |

### 5.3 Comparison Baselines

1. **Baseline 1:** No slicing (standard torch-mlir → MLIR lowering)
2. **Baseline 2:** Uniform tiling (fixed tile size across all ops)
3. **Baseline 3:** SConv-style PyTorch spatial-halo baseline (per-op slicing, no DAG analysis); official SConvTransform comparison is future work unless the upstream implementation is integrated
4. **EdgeSlice:** Proposed method (DAG-aware, dimension-aware slicing)

### 5.4 Ablation Study

1. **Ablation A:** DAG analysis only (no dimension heuristic → uniform tiling)
2. **Ablation B:** Dimension heuristic only (no DAG analysis → slice all ops)
3. **Ablation C:** Full EdgeSlice (DAG analysis + dimension heuristic)
4. **Ablation D (implemented; measurement in progress):** op-type-aware
   dimension selection — the dense-conv-only priority order vs. the
   depthwise/grouped-conv-aware variant of the §4.4 heuristic and its
   tile-size cost model, isolating how much of the MobileNetV2/EfficientNet-B0
   memory-reduction result is attributable to the channel-vs-spatial priority
   order rather than to liveness analysis. See the §4.4 refinement note and
   §5.1's rationale for those two workloads.

   Status of the evidence. (i) *Cache locality — positive, on one
   microbenchmark:* on the `depthwise_conv` model in
   `benchmarks/hardware_profiler.py`, the dense-conv-only order tiled
   spatially and made D1 misses 42% worse and wall clock 79% slower than not
   tiling at all; the depthwise-aware order reverses the sign. This is a
   single synthetic op shape, not a model-level claim, and the wall-clock half
   is measured on a virtualized host whose noise floor does not support
   quoting a speedup (§5.5).

   (ii) *Peak memory — no effect.* A direct A/B at 512×512 / L1 = 256 KB, with
   the depthwise-aware selection enabled and disabled and everything else held
   fixed, gives an identical modeled peak either way: MobileNetV2 5.54% and
   EfficientNet-B0 25.59% reduction in both arms. The reason is visible in the
   estimate: on MobileNetV2 the post-slicing peak is pinned by a `tensor.pad`
   that the dimension heuristic does not tile, so changing which axis the
   depthwise convs are tiled along cannot move it. Ablation D is therefore
   justified as a *locality* refinement, not a peak-memory one, and this
   proposal does not claim peak-memory benefit for it.

   An earlier draft of this section reported 26.7% / 20.6% here and read them
   as Ablation D's effect. Both figures were artifacts of an estimator defect —
   the baseline counted each destination-passing chain
   (`tensor.empty` → `linalg.fill` → `tensor.insert_slice`) as several live
   copies of one activation, while the sliced arm was partly credited for that
   same duplication through fused-producer accounting, so the two arms were not
   measured under the same memory model. §4.8.2 records the corrected model.
   The A/B above was run after that fix and supersedes those numbers.

### 5.5 Statistical Analysis

- **Primary metric:** Peak memory reduction (%) compared to baseline
- **Statistical tests:** Wilcoxon signed-rank test on paired timings, with medians and 95% bootstrap confidence intervals reported per model/configuration
- **Effect size:** Median paired difference and rank-biserial effect size for Wilcoxon comparisons
- **Confidence intervals:** 95% bootstrap CI for median execution time and paired speedups

---

## 6. Expected Results & Impact

### 6.1 Expected Results

Based on the performance of SConv (9–42% speedup), TSO (21.7% speedup), and the theoretical benefits of DAG-aware slicing:

- **H1:** EdgeSlice will reduce peak memory by 30–50% compared to no-slicing baseline, by selectively slicing only the tensors that contribute most to memory pressure.
- **H2:** EdgeSlice will achieve 10–20% execution time speedup on edge devices by reducing active working set size and improving spatial locality rather than guaranteeing perfect L1 residency.
- **H3:** EdgeSlice will improve locality relative to SConv-style PyTorch baselines on models with deep convolution stacks, by leveraging DAG-level analysis to avoid redundant slicing.
- **H4:** The compile-time overhead will be acceptable (< 2× baseline compile time) for ahead-of-time edge deployment scenarios.

### 6.2 Impact

- **Academic:** First work to combine DAG-based liveness analysis with MLIR Transform dialect rewriting for automatic tensor slicing in edge inference.
- **Practical:** Reusable MLIR frontend pass that can be integrated into any PyTorch-to-MLIR compilation flow.
- **Domain-Specific:** Addresses a real-world deployment challenge for edge devices processing large-input models.

### 6.3 Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MLIR Transform dialect limitations for complex slicing patterns | Medium | High | Fallback to SCF dialect if Transform dialect cannot express certain patterns |
| torch-mlir compatibility issues with target models | Medium | Medium | Audit torch-mlir op coverage early; plan custom lowering for unsupported ops |
| Dimension heuristic suboptimal for certain op types | Medium | Medium | Extend heuristic with cost model in future work |
| Dynamic shapes not supported | High | Low | Restrict initial evaluation to statically-shaped models; dynamic support as future work |
| Evaluation hardware access limited | Medium | Medium | Use QEMU-based simulation or focus on accessible targets (ARM SME, AVX512) |

---

## 7. Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Design & Prototyping** | Months 1–2 | DAG parser, liveness analysis prototype, dimension heuristic design |
| **Phase 2: MLIR Implementation** | Months 3–5 | Transform dialect extension, rewrite pipeline, integration with torch-mlir |
| **Phase 3: Evaluation Setup** | Months 6–7 | Model selection, input preparation, baseline implementation |
| **Phase 4: Experiments & Analysis** | Months 8–9 | Experiments, ablation study, statistical analysis |
| **Phase 5: Writing & Publication** | Months 10–11 | Paper drafting, revision, submission to target venue |

**Target Venues:** CGO (Code Generation and Optimization), ASPLOS, or MLSys.

---

## 8. References

### Primary Sources

1. Lücke, M. P., Zinenko, O., Moses, W. S., Steuwer, M., & Cohen, A. (2024). The MLIR Transform Dialect: Your compiler is more powerful than you think. *arXiv preprint arXiv:2409.03864v2*.

2. Ferrari, V., Pereira, M., Alvarenga, L., Leite, G., & Araujo, G. (2025). Using MLIR Transform to Design Sliced Convolution Algorithm. *arXiv preprint arXiv:2511.18222v1*.

3. Ferrari, V., Sousa, R., Pereira, M., Carvalho, J. P. L., Amaral, J. N., Moreira, J., & Araujo, G. (2023). Advancing Direct Convolution using Convolution Slicing Optimization and ISA Extensions. *arXiv preprint arXiv:2303.04739v1*.

4. Sousa, R., Pereira, M., Kwon, Y., Kim, T., Jung, N., Kim, C. S., Frank, M., & Araujo, G. (2023). Tensor Slicing and Optimization for Multicore NPUs. *Journal of Parallel and Distributed Computing*, 175, 66–79. https://doi.org/10.1016/j.jpdc.2022.12.008

### MLIR & Compiler Infrastructure

5. Lattner, C., et al. (2020). MLIR: Scaling Compiler Infrastructure for Domain Specific Computation. *Proceedings of CGO*.

6. Vasilache, N., et al. (2022). Composable and Modular Code Generation in MLIR: A Structured and Retargetable Approach to Tensor Compiler Construction. *arXiv preprint arXiv:2202.03293*.

7. Chen, T., et al. (2018). TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. *Proceedings of OSDI*.

### Hardware-Aware Compilation

8. Jouppi, N. P., et al. (2017). In-Datacenter Performance Analysis of a Tensor Processing Unit. *Proceedings of ISCA*.

9. Chen, Y.-H., et al. (2016). Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks. *IEEE Journal of Solid-State Circuits*, 52(1), 127–138.
