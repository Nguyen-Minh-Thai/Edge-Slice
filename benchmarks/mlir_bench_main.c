/*
 * Minimal MLIR payload driver for Cachegrind / native benchmarking.
 *
 * Builds ranked memref descriptors for the payload's input and output, invokes
 * the lowered MLIR C interface entry (e.g. _mlir_ciface_forward), and prints a
 * checksum so the compiler cannot delete the compute as dead code.
 *
 * Shapes are supplied by benchmarks/hardware_profiler.py at compile time as
 *   -DIN_RANK=4 -DIN_SIZES='{1,3,224,224}' -DOUT_RANK=2 -DOUT_SIZES='{1,1000}'
 * together with -DMLIR_ENTRY_NAME.
 *
 * Rank is a macro rather than a fixed 4 because the descriptor layout the
 * _mlir_ciface_ ABI expects is {allocated, aligned, offset, sizes[rank],
 * strides[rank]} — the strides begin immediately after the last size, so a
 * fixed-capacity array would put them at the wrong offset for any smaller
 * rank. Hard-coding rank 4 excluded every classifier in the evaluation matrix
 * (ResNet-50, MobileNetV2 and EfficientNet all end in a rank-2 logits tensor),
 * which meant no RSS or cachegrind number could be collected for them at all.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef IN_RANK
#define IN_RANK 4
#endif
#ifndef IN_SIZES
#define IN_SIZES {1, 3, 32, 32}
#endif
#ifndef OUT_RANK
#define OUT_RANK 4
#endif
#ifndef OUT_SIZES
#define OUT_SIZES {1, 8, 32, 32}
#endif
#ifndef NUM_ITERS
#define NUM_ITERS 8
#endif
#ifndef MLIR_ENTRY_NAME
#define MLIR_ENTRY_NAME forward
#endif

#define MLIR_GLUE(a, b) a##b
#define MLIR_XGLUE(a, b) MLIR_GLUE(a, b)
#define MLIR_CIFACE_FUNC MLIR_XGLUE(_mlir_ciface_, MLIR_ENTRY_NAME)

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[IN_RANK];
  int64_t strides[IN_RANK];
} MemRefIn;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[OUT_RANK];
  int64_t strides[OUT_RANK];
} MemRefOut;

extern void MLIR_CIFACE_FUNC(void *result, void *input);

static const int64_t kInSizes[IN_RANK] = IN_SIZES;
static const int64_t kOutSizes[OUT_RANK] = OUT_SIZES;

static int64_t elem_count(const int64_t *sizes, int rank) {
  int64_t count = 1;
  for (int i = 0; i < rank; ++i) {
    count *= sizes[i];
  }
  return count;
}

static void fill_dummy_input(float *data, size_t count) {
  /* Deterministic pattern — not random to keep binaries reproducible. */
  for (size_t i = 0; i < count; ++i) {
    data[i] = (float)((i % 97) * 0.001f - 0.5f);
  }
}

static void init_input(MemRefIn *desc, float *storage) {
  desc->allocated = storage;
  desc->aligned = storage;
  desc->offset = 0;
  int64_t stride = 1;
  for (int i = IN_RANK - 1; i >= 0; --i) {
    desc->sizes[i] = kInSizes[i];
    desc->strides[i] = stride;
    stride *= kInSizes[i];
  }
}

/*
 * Sum the callee's result through its own descriptor. The _mlir_ciface_
 * convention has the CALLEE allocate the result buffer and overwrite every
 * field of the result descriptor, so the caller must read data through
 * aligned/offset/sizes/strides — never through a caller-allocated buffer.
 *
 * Walks the index space odometer-style so it is correct for any rank and for
 * a non-contiguous result; the dense case short-circuits to one flat pass the
 * compiler can vectorize.
 */
static double checksum_result(const MemRefOut *desc) {
  double sum = 0.0;
  const float *base = desc->aligned + desc->offset;
  const int64_t count = elem_count(desc->sizes, OUT_RANK);

  int dense = 1;
  int64_t expected = 1;
  for (int i = OUT_RANK - 1; i >= 0; --i) {
    if (desc->strides[i] != expected) {
      dense = 0;
      break;
    }
    expected *= desc->sizes[i];
  }
  if (dense) {
    for (int64_t i = 0; i < count; ++i) {
      sum += (double)base[i];
    }
    return sum;
  }

  int64_t index[OUT_RANK];
  memset(index, 0, sizeof(index));
  for (int64_t visited = 0; visited < count; ++visited) {
    int64_t offset = 0;
    for (int i = 0; i < OUT_RANK; ++i) {
      offset += index[i] * desc->strides[i];
    }
    sum += (double)base[offset];
    for (int i = OUT_RANK - 1; i >= 0; --i) {
      if (++index[i] < desc->sizes[i]) {
        break;
      }
      index[i] = 0;
    }
  }
  return sum;
}

/*
 * Free the callee-allocated result buffer, guarding against a result that
 * aliases the harness's input buffer: in-place bufferization may legally
 * return the input memref, and freeing it mid-run would corrupt the heap
 * (use-after-free in later iterations, double free at exit). A result backed
 * by non-heap storage other than the input (e.g. memref.global) cannot be
 * detected here; the strict-IR gate upstream is what rules those payloads out.
 */
static void free_result(MemRefOut *desc, const float *input_storage) {
  if (desc->allocated != NULL && desc->allocated != input_storage) {
    free(desc->allocated);
  }
  desc->allocated = NULL;
}

int main(int argc, char **argv) {
  /* Iteration count is runtime-overridable so wall-clock callers can amortize
   * process startup over many kernel invocations while cachegrind keeps the
   * compile-time default. */
  int iters = NUM_ITERS;
  if (argc > 1) {
    iters = atoi(argv[1]);
    if (iters < 1) {
      iters = 1;
    }
  }

  const size_t in_elems = (size_t)elem_count(kInSizes, IN_RANK);
  float *input_storage = (float *)malloc(in_elems * sizeof(float));
  if (!input_storage) {
    fprintf(stderr, "allocation failed\n");
    return 1;
  }
  fill_dummy_input(input_storage, in_elems);

  MemRefIn input_desc;
  MemRefOut output_desc;
  init_input(&input_desc, input_storage);
  memset(&output_desc, 0, sizeof(output_desc));

  for (int iter = 0; iter < iters; ++iter) {
    /* Each call allocates a fresh result buffer; release the previous one so
     * long wall-clock runs do not grow without bound. */
    free_result(&output_desc, input_storage);
    MLIR_CIFACE_FUNC(&output_desc, &input_desc);
  }

  /* Callers watch only the exit code (stderr is discarded on success), so a
   * broken kernel contract must fail loudly, not warn: timings of a kernel
   * that computed nothing (or the wrong shape) are not measurements. */
  if (output_desc.aligned == NULL) {
    fprintf(stderr, "error: kernel did not populate the result descriptor\n");
    free(input_storage);
    return 1;
  }
  for (int i = 0; i < OUT_RANK; ++i) {
    if (output_desc.sizes[i] != kOutSizes[i]) {
      fprintf(stderr,
              "error: result dim %d is %" PRId64 ", expected %" PRId64 "\n", i,
              output_desc.sizes[i], kOutSizes[i]);
      free_result(&output_desc, input_storage);
      free(input_storage);
      return 1;
    }
  }

  double checksum = checksum_result(&output_desc);
  /* Printed to stdout — consumed by scripts; also prevents DCE. */
  printf("mlir_bench_checksum=%.6f iters=%d\n", checksum, iters);
  fflush(stdout);

  free_result(&output_desc, input_storage);
  free(input_storage);
  return 0;
}
