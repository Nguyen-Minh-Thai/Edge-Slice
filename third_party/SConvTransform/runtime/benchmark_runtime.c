#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#if defined(EDGESLIDE_USE_CBLAS)
#include <cblas.h>
#endif

#if defined(EDGESLIDE_USE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
} MemRef4F32;

typedef struct {
  int64_t rank;
  void *descriptor;
} UnrankedMemRef;

double rtclock(void) {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

void printFlops(double flops) {
  fprintf(stderr, "flops_per_s=%.6f\n", flops);
}

void printChecksumF32(int64_t rank, void *ptr) {
  if (rank != 4 || !ptr) {
    fprintf(stderr, "checksum=nan rank=%ld\n", (long)rank);
    return;
  }
  const MemRef4F32 *m = (const MemRef4F32 *)ptr;
  const float *data = (const float *)m->aligned;
  double sum = 0.0;
  double sumsq = 0.0;
  int64_t count = m->sizes[0] * m->sizes[1] * m->sizes[2] * m->sizes[3];
  for (int64_t i0 = 0; i0 < m->sizes[0]; ++i0)
    for (int64_t i1 = 0; i1 < m->sizes[1]; ++i1)
      for (int64_t i2 = 0; i2 < m->sizes[2]; ++i2)
        for (int64_t i3 = 0; i3 < m->sizes[3]; ++i3) {
          int64_t index = m->offset + i0 * m->strides[0] + i1 * m->strides[1] +
                          i2 * m->strides[2] + i3 * m->strides[3];
          double value = data[index];
          sum += value;
          sumsq += value * value;
        }
  fprintf(stderr, "checksum=%.17g sumsq=%.17g elements=%ld\n", sum, sumsq,
          (long)count);
}

/* Functions declared with llvm.emit_c_interface are called through this
 * prefixed ABI after MLIR lowering. */
void _mlir_ciface_printChecksumF32(void *unranked) {
  if (!unranked) {
    fprintf(stderr, "checksum=nan rank=0\n");
    return;
  }
  const UnrankedMemRef *u = (const UnrankedMemRef *)unranked;
  printChecksumF32(u->rank, u->descriptor);
}

/* ABI emitted by transform.lower.to_blas. A and B are KxM/KxN packed panels;
 * C is an N-row output tile with lda elements between rows. */
int sgemm_blas_kernel(int64_t m, int64_t n, int64_t k, float alpha,
                      const float *a, const float *b, float *c, int64_t lda) {
  if (!a || !b || !c || m < 0 || n < 0 || k < 0 || lda < m)
    return -1;
#ifdef EDGESLIDE_USE_CBLAS
  /* B^T (n x k) * A (k x m) -> C (n x m), matching SConv's packed panels. */
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              (int)n, (int)m, (int)k, alpha,
              b, (int)n, a, (int)m, 1.0f, c, (int)lda);
  return 0;
#elif defined(EDGESLIDE_USE_NEON) && defined(__aarch64__)
  /* SConv's normal microkernel is mK=[16,8].  Keep edge tiles on the scalar
   * path; CSA emits full tiles whenever the convolution dimensions allow it. */
  if (m == 16 && n == 8) {
    float32x4_t acc[8][4];
    for (int j = 0; j < 8; ++j) {
      const float *row = c + j * lda;
      acc[j][0] = vld1q_f32(row);
      acc[j][1] = vld1q_f32(row + 4);
      acc[j][2] = vld1q_f32(row + 8);
      acc[j][3] = vld1q_f32(row + 12);
    }
    for (int64_t p = 0; p < k; ++p) {
      const float *arow = a + p * 16;
      const float *brow = b + p * 8;
      float32x4_t av0 = vld1q_f32(arow);
      float32x4_t av1 = vld1q_f32(arow + 4);
      float32x4_t av2 = vld1q_f32(arow + 8);
      float32x4_t av3 = vld1q_f32(arow + 12);
      for (int j = 0; j < 8; ++j) {
        float scale = alpha * brow[j];
        acc[j][0] = vmlaq_n_f32(acc[j][0], av0, scale);
        acc[j][1] = vmlaq_n_f32(acc[j][1], av1, scale);
        acc[j][2] = vmlaq_n_f32(acc[j][2], av2, scale);
        acc[j][3] = vmlaq_n_f32(acc[j][3], av3, scale);
      }
    }
    for (int j = 0; j < 8; ++j) {
      float *row = c + j * lda;
      vst1q_f32(row, acc[j][0]);
      vst1q_f32(row + 4, acc[j][1]);
      vst1q_f32(row + 8, acc[j][2]);
      vst1q_f32(row + 12, acc[j][3]);
    }
    return 0;
  }
#endif
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int64_t p = 0; p < k; ++p)
        sum += a[p * m + i] * b[p * n + j];
      c[j * lda + i] += alpha * sum;
    }
  return 0;
}
