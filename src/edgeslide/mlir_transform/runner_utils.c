/* MLIR runner utilities for ExecutionEngine JIT (memrefCopy, malloc). */

#include <alloca.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int64_t rank;
  void *descriptor;
} UnrankedMemRefType;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[1];
} RankedMemRefDescriptor;

static void dynamicMemRefCopy(int64_t elemSize, const UnrankedMemRefType *srcArg,
                              const UnrankedMemRefType *dstArg) {
  int64_t rank = srcArg->rank;
  RankedMemRefDescriptor *srcDesc =
      (RankedMemRefDescriptor *)srcArg->descriptor;
  RankedMemRefDescriptor *dstDesc =
      (RankedMemRefDescriptor *)dstArg->descriptor;

  int64_t *srcSizes = srcDesc->sizes;
  int64_t *srcStrides = srcSizes + rank;
  int64_t *dstSizes = dstDesc->sizes;
  int64_t *dstStrides = dstSizes + rank;

  for (int64_t r = 0; r < rank; ++r) {
    if (srcSizes[r] == 0) {
      return;
    }
  }

  char *srcPtr = (char *)srcDesc->aligned + srcDesc->offset * elemSize;
  char *dstPtr = (char *)dstDesc->aligned + dstDesc->offset * elemSize;

  if (rank == 0) {
    memcpy(dstPtr, srcPtr, (size_t)elemSize);
    return;
  }

  int64_t *indices = (int64_t *)alloca(sizeof(int64_t) * (size_t)rank);
  int64_t *srcStrideBytes =
      (int64_t *)alloca(sizeof(int64_t) * (size_t)rank);
  int64_t *dstStrideBytes =
      (int64_t *)alloca(sizeof(int64_t) * (size_t)rank);

  for (int64_t r = 0; r < rank; ++r) {
    indices[r] = 0;
    srcStrideBytes[r] = srcStrides[r] * elemSize;
    dstStrideBytes[r] = dstStrides[r] * elemSize;
  }

  int64_t readIndex = 0;
  int64_t writeIndex = 0;
  for (;;) {
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, (size_t)elemSize);
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      int64_t newIndex = ++indices[axis];
      readIndex += srcStrideBytes[axis];
      writeIndex += dstStrideBytes[axis];
      if (srcSizes[axis] != newIndex) {
        break;
      }
      if (axis == 0) {
        return;
      }
      indices[axis] = 0;
      readIndex -= srcSizes[axis] * srcStrideBytes[axis];
      writeIndex -= dstSizes[axis] * dstStrideBytes[axis];
    }
  }
}

void memrefCopy(int64_t elemSize, UnrankedMemRefType *src,
                UnrankedMemRefType *dst) {
  dynamicMemRefCopy(elemSize, src, dst);
}

void _mlir_memrefCopy(int64_t elemSize, UnrankedMemRefType *src,
                      UnrankedMemRefType *dst) {
  memrefCopy(elemSize, src, dst);
}

void *_mlir_malloc(int64_t size) { return malloc((size_t)size); }