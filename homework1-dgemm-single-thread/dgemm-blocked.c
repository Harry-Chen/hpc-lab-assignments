#define _GNU_SOURCE

#include <immintrin.h>
#include <sched.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

#define MAX_N 2000

#ifndef ENABLE_STRASSEN
#define ENABLE_STRASSEN 0
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

#if !ENABLE_STRASSEN
const char *dgemm_desc = "Simple blocked dgemm.";
#else
const char *dgemm_desc = "Simple blocked dgemm (with Strassen algorithm).";
#endif


#define CONCAT(A, M, N, K) A ## M ## _ ## N ## _ ## K
#define AVX_KERNEL_NAME_(M, N, K) CONCAT(do_block_simd_, M, N, K)
#define AVX_KERNEL_NAME AVX_KERNEL_NAME_(BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K)

#define BLOCK_SIZE_M 40
#define BLOCK_SIZE_N 40
#define BLOCK_SIZE_K 40
#include "dgemm-blocked-avx-kernel.c"
#undef BLOCK_SIZE_M
#undef BLOCK_SIZE_N
#undef BLOCK_SIZE_K


#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32
#include "dgemm-blocked-avx-kernel.c"


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline __attribute__((always_inline)) void do_block_naive(
    int lda, int ldb, int ldc, int M, int N, int K, const double *__restrict__ const A,
    const double *__restrict__ const B, double *__restrict__ const C) {

  // fprintf(stderr, "Naive %d %d %d %p %p %p %d %d %d\n", M, N, K, A, B, C, lda, ldb, ldc);

  /* For each row i of A */
#pragma ivdep
  for (int i = 0; i < M; ++i) {
    /* For each column j of B */
    __builtin_prefetch(A + i * lda, 0);
    __builtin_prefetch(C + i * ldc, 1);
#pragma ivdep
    for (int j = 0; j < N; ++j) {
      /* Compute C(i,j) */
#pragma ivdep
      for (int k = 0; k < K; ++k) C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
    }
  }
}

#if ENABLE_STRASSEN
#include "dgemm-blocked-strassen-kernel.h"
#endif

// buffer for DGEMM padding
double *A_buf, *B_buf, *C_buf;

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, const double *__restrict__ A, const double *__restrict__ B,
                  double *__restrict__ C) {

  // (A*B)^T = B^T * A^T, so we can treat A, B, C in row-major format and
  // calculate C = C + B * A swap A and B for simplicity
  const double *__restrict__ temp = A;
  A = B;
  B = temp;

  bool pad = false;
  int dim = lda;

#if ENABLE_STRASSEN
  if (likely((dim & (dim - 1)) == 0 && dim >= BLOCK_SIZE_N)) {
    // power of 2 - use strassen
    // C_s = A * B
    do_block_strassen(dim, dim, MAX_N, dim, A, B, C_strassen, st_im_1, st_im_2, st_im_3, st_im_4, st_p_1, st_p_2);
    // C += C_s
    matrix_add_single(MAX_N, dim, dim, C_strassen, C);
    return;
  }
#endif

  int size = lda;

  if (lda % 32 == 31) {
    size++;
  } else if (lda % 32 == 1) {
    size--;
  }

  int padding_dim = 32;

  if (size % 40 == 0 || size == 1024) {
    padding_dim = 40;
  }

  
  int whole_count = lda / padding_dim;
  int whole_width = whole_count * padding_dim;
  int remain = lda % padding_dim; // AVX registers

  // pad if remaining numbers are relatively many
  if (remain > padding_dim / 2 + 1) {
    pad = true;
    dim = (lda / padding_dim + 1) * padding_dim;
#pragma ivdep
    for (int i = 0; i < whole_width; ++i) {
      double *__restrict__ const A_buf_pos = A_buf + i * MAX_N + whole_width;
      double *__restrict__ const B_buf_pos = B_buf + i * MAX_N + whole_width;
      double *__restrict__ const C_buf_pos = C_buf + i * MAX_N + whole_width;
      const double *__restrict__ const A_pos = A + i * lda + whole_width;
      const double *__restrict__ const B_pos = B + i * lda + whole_width;
      const double *__restrict__ const C_pos = C + i * lda + whole_width;
      __builtin_prefetch(A_buf_pos, 1);
      __builtin_prefetch(B_buf_pos, 1);
      __builtin_prefetch(C_buf_pos, 1);
      __builtin_prefetch(A_pos, 0);
      __builtin_prefetch(B_pos, 0);
      __builtin_prefetch(C_pos, 0);
      memcpy(A_buf_pos, A_pos, sizeof(double) * remain);
      memcpy(B_buf_pos, B_pos, sizeof(double) * remain);
      memcpy(C_buf_pos, C_pos, sizeof(double) * remain);
    }
#pragma ivdep
    for (int i = whole_width; i < lda; ++i) {
      double *__restrict__ const A_buf_pos = A_buf + i * MAX_N;
      double *__restrict__ const B_buf_pos = B_buf + i * MAX_N;
      double *__restrict__ const C_buf_pos = C_buf + i * MAX_N;
      const double *__restrict__ const A_pos = A + i * lda;
      const double *__restrict__ const B_pos = B + i * lda;
      const double *__restrict__ const C_pos = C + i * lda;
      __builtin_prefetch(A_buf_pos, 1);
      __builtin_prefetch(B_buf_pos, 1);
      __builtin_prefetch(C_buf_pos, 1);
      __builtin_prefetch(A_pos, 0);
      __builtin_prefetch(B_pos, 0);
      __builtin_prefetch(C_pos, 0);
      memcpy(A_buf_pos, A_pos, sizeof(double) * lda);
      memcpy(B_buf_pos, B_pos, sizeof(double) * lda);
      memcpy(C_buf_pos, C_pos, sizeof(double) * lda);
    }
  }

  if (padding_dim == 32) {
#define BLOCK_SIZE 32
#include "dgemm-blocked-block-kernel.c"
#undef BLOCK_SIZE
  } else {
#define BLOCK_SIZE 40
#include "dgemm-blocked-block-kernel.c"
#undef BLOCK_SIZE
  }

  // copy data back
  if (pad) {
#pragma ivdep
    for (int i = 0; i < whole_width; ++i) {
      const double *__restrict__ const C_buf_pos = C_buf + i * MAX_N + whole_width;
      double *__restrict__ const C_pos = C + i * lda + whole_width;
      memcpy(C_pos, C_buf_pos, sizeof(double) * remain);
    }
#pragma ivdep
    for (int i = whole_width; i < lda; ++i) {
      const double *__restrict__ const C_buf_pos = C_buf + i * MAX_N;
      double *__restrict__ const C_pos = C + i * lda;
      memcpy(C_pos, C_buf_pos, sizeof(double) * lda);
    }
  }
  
}

// bind this process to CPU 1
__attribute__((constructor)) void bind_core() {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(1, &set);
  sched_setaffinity(0, sizeof(set), &set);
  posix_memalign((void **)&A_buf, 64, MAX_N * MAX_N * sizeof(double));
  posix_memalign((void **)&B_buf, 64, MAX_N * MAX_N * sizeof(double));
  posix_memalign((void **)&C_buf, 64, MAX_N * MAX_N * sizeof(double));
  bzero(A_buf, sizeof(A_buf));
  bzero(B_buf, sizeof(B_buf));
  bzero(C_buf, sizeof(C_buf));
}
