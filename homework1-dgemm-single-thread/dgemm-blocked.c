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
      double cij = C[i * ldc + j];
#pragma ivdep
      for (int k = 0; k < K; ++k) cij += A[i * lda + k] * B[k * ldb + j];
      C[i * ldc + j] = cij;
    }
  }
}

#if ENABLE_STRASSEN

// B += A
static inline __attribute__((always_inline)) void matrix_add_single(int lda, int ldb, int n, double *__restrict__ A, double *__restrict__ B) {
  const int ADD_UNROLL = 8;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; j += 4 * ADD_UNROLL) {
#pragma unroll(ADD_UNROLL)
      for (int x = 0; x < ADD_UNROLL; x++) {
        _mm256_store_pd(B + i * ldb + j + x * 4, _mm256_add_pd(_mm256_load_pd(A + i * lda + j + x * 4), _mm256_load_pd(B + i * ldb + j + x * 4)));
      }
    }
  }
}

// C = A +/- B, n >= BLOCK_SIZE_N
static inline __attribute__((always_inline)) void matrix_add(bool add, 
  int lda, int ldb, int ldc, int n, double *__restrict__ A, double *__restrict__ B,
    double *__restrict__ C
) {
  const int ADD_UNROLL = 8;
  if (add) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; j += 4 * ADD_UNROLL) {
  #pragma unroll(ADD_UNROLL)
        for (int x = 0; x < ADD_UNROLL; x++) {
          _mm256_store_pd(C + i * ldc + j + x * 4, _mm256_add_pd(_mm256_load_pd(A + i * lda + j + x * 4), _mm256_load_pd(B + i * ldb + j + x * 4)));
        }
      }
    }
  } else {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; j += 4 * ADD_UNROLL) {
  #pragma unroll(ADD_UNROLL)
        for (int x = 0; x < ADD_UNROLL; x++) {
          _mm256_store_pd(C + i * ldc + j + x * 4, _mm256_sub_pd(_mm256_load_pd(A + i * lda + j + x * 4), _mm256_load_pd(B + i * ldb + j + x * 4)));
        }
      }
    }
  }
}

// C += A +/- B, n >= BLOCK_SIZE_N
static inline __attribute__((always_inline)) void matrix_add_to(bool add, 
  int lda, int ldb, int ldc, int n, double *__restrict__ A, double *__restrict__ B,
    double *__restrict__ C
) {
  const int ADD_UNROLL = 8;
  if (add) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; j += 4 * ADD_UNROLL) {
  #pragma unroll(ADD_UNROLL)
        for (int x = 0; x < ADD_UNROLL; x++) {
          _mm256_store_pd(C + i * ldc + j + x * 4, _mm256_add_pd(_mm256_load_pd(C + i * ldc + j + x * 4), _mm256_add_pd(_mm256_load_pd(A + i * lda + j + x * 4), _mm256_load_pd(B + i * ldb + j + x * 4))));
        }
      }
    }
  } else {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; j += 4 * ADD_UNROLL) {
  #pragma unroll(ADD_UNROLL)
        for (int x = 0; x < ADD_UNROLL; x++) {
          _mm256_store_pd(C + i * ldc + j + x * 4, _mm256_add_pd(_mm256_load_pd(C + i * ldc + j + x * 4), _mm256_sub_pd(_mm256_load_pd(A + i * lda + j + x * 4), _mm256_load_pd(B + i * ldb + j + x * 4))));
        }
      }
    }
  }
}

// buffers for Strassen algorithm
static double st_im_1[MAX_N * MAX_N], st_im_2[MAX_N * MAX_N],
    st_im_3[MAX_N * MAX_N], st_im_4[MAX_N * MAX_N], st_p_1[MAX_N * MAX_N], st_p_2[MAX_N * MAX_N];
static double C_strassen[MAX_N * MAX_N];


// calculate C = A * B by Strassen algorithm
static inline void do_block_strassen(
    int lda, int ldb, int ldc, int n, double *__restrict__ A, double *__restrict__ B,
    double *__restrict__ C, double *__restrict__ im_1,
    double *__restrict__ im_2, double *__restrict__ im_3, double *__restrict__ im_4,
    double *__restrict__ p_1, double *__restrict__ p_2) {

  // fprintf(stderr, "Strassen %d %p %p %p %d %d %d\n", n, A, B, C, lda, ldb, ldc);

  // n is small enough now
  if (unlikely(n < BLOCK_SIZE_N)) {
    __builtin_unreachable();
  } else if (n == BLOCK_SIZE_N) {
    do_block_simd_32_32_32(lda, ldb, ldc, A, B, C, true);
    return;
  }

  int m = n / 2;
  
  // split sub-matricies
  double *__restrict__ A11 = A;
  double *__restrict__ A12 = A + m;
  double *__restrict__ A21 = A + m * lda;
  double *__restrict__ A22 = A + m * lda + m;
  double *__restrict__ B11 = B;
  double *__restrict__ B12 = B + m;
  double *__restrict__ B21 = B + m * ldb;
  double *__restrict__ B22 = B + m * ldb + m;
  double *__restrict__ C11 = C;
  double *__restrict__ C12 = C + m;
  double *__restrict__ C21 = C + m * ldc;
  double *__restrict__ C22 = C + m * ldc + m;
  // intermediate values offset
  // variables with trailing underscore means it would be used in recursion
  // im_1: B12m22, A11p12, A21p22, im_1_
  // im_2: B21m11, A11p22, B11p22, im_2_
  // im_3: A12m22, B21p22, A21m11, im_3_
  // im_4: B11p12, p_1_, EMPTY, im_4_
  // p_1: P1, P2, P3, P4
  // p_2: P5, P6, P7, p_2_
  double *__restrict__ B12m22 = im_1;
  double *__restrict__ A11p12 = im_1 + m;
  double *__restrict__ A21p22 = im_1 + m * MAX_N;
  double *__restrict__ B21m11 = im_2;
  double *__restrict__ A11p22 = im_2 + m;
  double *__restrict__ B11p22 = im_2 + m * MAX_N;
  double *__restrict__ A12m22 = im_3;
  double *__restrict__ B21p22 = im_3 + m;
  double *__restrict__ A21m11 = im_3 + m * MAX_N;
  double *__restrict__ B11p12 = im_4;
  double *__restrict__ P1 = p_1;
  double *__restrict__ P2 = p_1 + m;
  double *__restrict__ P3 = p_1 + m * MAX_N;
  double *__restrict__ P4 = p_1 + m * MAX_N + m;
  double *__restrict__ P5 = p_2;
  double *__restrict__ P6 = p_2 + m;
  double *__restrict__ P7 = p_2 + m * MAX_N;
  // smaller matricies
  double *__restrict__ _p_1_ = im_4 + m;
  double *__restrict__ _p_2_ = p_2 + m * MAX_N + m;
  double *__restrict__ _im_1_ = im_1 + m * MAX_N + m;
  double *__restrict__ _im_2_ = im_2 + m * MAX_N + m;
  double *__restrict__ _im_3_ = im_3 + m * MAX_N + m;
  double *__restrict__ _im_4_ = im_4 + m * MAX_N + m;

  // calculate intermediate values
  matrix_add(false, ldb, ldb, MAX_N, m, B12, B22, B12m22);
  matrix_add(true,  lda, lda, MAX_N, m, A11, A12, A11p12);
  matrix_add(true,  lda, lda, MAX_N, m, A21, A22, A21p22);
  matrix_add(false, ldb, ldb, MAX_N, m, B21, B11, B21m11);
  matrix_add(true,  lda, lda, MAX_N, m, A11, A22, A11p22);
  matrix_add(true,  ldb, ldb, MAX_N, m, B11, B22, B11p22);
  matrix_add(false, lda, lda, MAX_N, m, A12, A22, A12m22);
  matrix_add(true,  ldb, ldb, MAX_N, m, B21, B22, B21p22);
  matrix_add(false, lda, lda, MAX_N, m, A21, A11, A21m11);
  matrix_add(true,  ldb, ldb, MAX_N, m, B11, B12, B11p12);

  // P1 = Strassen(A11,B12 − B22)
  do_block_strassen(lda, MAX_N, MAX_N, m, A11, B12m22, P1, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P2 = Strassen(A11 + A12,B22)
  do_block_strassen(MAX_N, ldb, MAX_N, m, A11p12, B22, P2, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P3 = Strassen(A21 + A22,B11)
  do_block_strassen(MAX_N, ldb, MAX_N, m, A21p22, B11, P3, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P4 = Strassen(A22,B21 − B11)
  do_block_strassen(lda, MAX_N, MAX_N, m, A22, B21m11, P4, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P5 = Strassen(A11 + A22,B11 + B22)
  do_block_strassen(MAX_N, MAX_N, MAX_N, m, A11p22, B11p22, P5, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P6 = Strassen(A12 − A22,B21 + B22)
  do_block_strassen(MAX_N, MAX_N, MAX_N, m, A12m22, B21p22, P6, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P7 = Strassen(A21 − A11,B11 + B12)
  do_block_strassen(MAX_N, MAX_N, MAX_N, m, A21m11, B11p12, P7, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);

  // C11 = (P5 + P4) + (P6 - P2)
  matrix_add(true, MAX_N, MAX_N, ldc, m, P5, P4, C11);
  matrix_add_to(false, MAX_N, MAX_N, ldc, m, P6, P2, C11);
  // C12 = P1 + P2
  matrix_add(true, MAX_N, MAX_N, ldc, m, P1, P2, C12);
  // C21 = P3 + P4
  matrix_add(true, MAX_N, MAX_N, ldc, m, P3, P4, C21);
  // C22 = (P1 + P7) + (P5 - P3)
  matrix_add(true, MAX_N, MAX_N, ldc, m, P1, P7, C22);
  matrix_add_to(false, MAX_N, MAX_N, ldc, m, P5, P3, C22);

}
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

  if (size % 40 == 0) {
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
