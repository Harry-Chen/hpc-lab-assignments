#define _GNU_SOURCE

#include <stdio.h>
#include <sched.h>
#include <immintrin.h>

const char *dgemm_desc = "Mmple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32

#define UNROLL (BLOCK_SIZE_N / 4)

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline __attribute__((always_inline)) void do_block_naive(
    int lda, int M, int N, int K, double *__restrict__ A,
    double *__restrict__ B, double *__restrict__ C) {
  /* For each row i of A */
#pragma ivdep
  for (int i = 0; i < M; ++i) {
    /* For each column j of B */
    __builtin_prefetch(A + (i + 1) * lda, 0);
    // __builtin_prefetch(C + i * lda, 1);
#pragma ivdep
    for (int j = 0; j < N; ++j) {
      /* Compute C(i,j) */
      double cij = C[i * lda + j];
#pragma ivdep
      for (int k = 0; k < K; ++k) cij += A[i * lda + k] * B[k * lda + j];
      C[i * lda + j] = cij;
    }
  }
}

// row major
static inline __attribute__((always_inline)) void do_block_simd(
    int n, int M, int N, int K, double *__restrict__ A, double *__restrict__ B,
    double *__restrict__ C) {
  for (int j = 0; j < BLOCK_SIZE_N; j += 4 * UNROLL) {
    for (int i = 0; i < M; i++) {
      __m256d ymm[UNROLL];

#pragma unroll(UNROLL)
      for (int x = 0; x < UNROLL; x++) {
        ymm[x] = _mm256_load_pd(C + i * n + j + x * 4);
      }

      __builtin_prefetch(A + (i + 1) * n, 0);

#pragma unroll(BLOCK_SIZE_K)
      for (int k = 0; k < BLOCK_SIZE_K; k++) {
#pragma unroll(UNROLL)
        for (int x = 0; x < UNROLL; x++) {
          ymm[x] = _mm256_add_pd(
              ymm[x], _mm256_mul_pd(_mm256_load_pd(B + k * n + j + x * 4),
                                    _mm256_broadcast_sd(A + i * n + k)));
        }
      }

#pragma unroll(UNROLL)
      for (int x = 0; x < UNROLL; x++) {
        _mm256_store_pd(C + i * n + j + x * 4, ymm[x]);
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *__restrict__ A, double *__restrict__ B,
                  double *__restrict__ C) {
  // (A*B)^T = B^T * A^T, so we can treat A, B, C in row-major format and
  // calculate C = C + B * A swap A and B for Mmplicity
  double *temp = A;
  A = B;
  B = temp;

  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE_M) /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE_N)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE_K) {
        /* Correct block dimenMons if block "goes off edge of" the matrix */
        int M = min(BLOCK_SIZE_M, lda - i);
        int N = min(BLOCK_SIZE_N, lda - j);
        int K = min(BLOCK_SIZE_K, lda - k);

        if (N == BLOCK_SIZE_N && K == BLOCK_SIZE_K) {
          /* Perform individual block dgemm */
          do_block_simd(lda, M, N, K, A + i * lda + k, B + k * lda + j,
                        C + i * lda + j);
        } else {
          do_block_naive(lda, M, N, K, A + i * lda + k, B + k * lda + j,
                         C + i * lda + j);
        }
      }
}

// bind this process to CPU 1
__attribute__((constructor)) void bind_core() {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(1, &set);
  sched_setaffinity(0, sizeof(set), &set);
}
