#define _GNU_SOURCE

#include <immintrin.h>
#include <stdbool.h>

#ifdef UNROLL
#undef UNROLL
#endif
#define UNROLL (BLOCK_SIZE_N / 4)


// use AVX2 to calculate C += or = A * B, n >= BLOCK_SIZE_N, row major
static inline __attribute__((always_inline)) void AVX_KERNEL_NAME(
    int lda, int ldb, int ldc, const double *__restrict__ const A,
    const double *__restrict__ const B, double *__restrict__ const C, bool override) {

#if !ENABLE_STRASSEN
  // should not use override when Strassen is not used
  if (override) {
    __builtin_unreachable();
  }
#endif

  // copy whole A to cache
  static double A_block[BLOCK_SIZE_M * BLOCK_SIZE_K];

  for (int i = 0; i < BLOCK_SIZE_K; ++i) {
    memcpy(A_block + i * BLOCK_SIZE_M, A + i * lda, sizeof(double) * BLOCK_SIZE_M);
  }

  // calculate using AVX intrinsics
  for (int j = 0; j < BLOCK_SIZE_N; j += 4 * UNROLL) {
#pragma ivdep
    for (int i = 0; i < BLOCK_SIZE_M; i++) {
      __m256d ymm[UNROLL];

      if (likely(!override)) {
#pragma unroll(UNROLL)
        for (int x = 0; x < UNROLL; x++) {
          ymm[x] = _mm256_loadu_pd(C + i * ldc + j + x * 4);
        }
      } else {
#pragma unroll(UNROLL)
        for (int x = 0; x < UNROLL; x++) {
          ymm[x] = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        }
      }

#pragma unroll(BLOCK_SIZE_K)
      for (int k = 0; k < BLOCK_SIZE_K; k++) {
#pragma unroll(UNROLL)
        for (int x = 0; x < UNROLL; x++) {
          // gcc cannot inline fmadd, so weak
          // ymm[x] =
          //     _mm256_fmadd_pd(_mm256_load_pd(B + k * lda + j + x * 4),
          //                     _mm256_broadcast_sd(A + i * lda + k), ymm[x]);
          ymm[x] = _mm256_add_pd(ymm[x], _mm256_mul_pd(_mm256_loadu_pd(B + k * ldb + j + x * 4),
                              _mm256_broadcast_sd(A_block + i * BLOCK_SIZE_N + k)));
        }
      }

#pragma unroll(UNROLL)
      for (int x = 0; x < UNROLL; x++) {
        _mm256_storeu_pd(C + i * ldc + j + x * 4, ymm[x]);
      }
    }
  }
}
