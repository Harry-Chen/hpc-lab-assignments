#define _GNU_SOURCE

#include <assert.h>
#include <immintrin.h>
#include <sched.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

const char *dgemm_desc = "Mmple blocked dgemm.";

const int BLOCK_SIZE_M = 32;
const int BLOCK_SIZE_N = 32;
const int BLOCK_SIZE_K = 32;

const int UNROLL = BLOCK_SIZE_N / 4;
#define MAX_N 2048

#define ENABLE_STRASSEN true

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
    __builtin_prefetch(A + i * lda, 0);
    __builtin_prefetch(C + i * lda, 1);
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
    int lda, int ldb, int ldc, int M, int N, int K, double *__restrict__ A,
    double *__restrict__ B, double *__restrict__ C) {
  for (int j = 0; j < BLOCK_SIZE_N; j += 4 * UNROLL) {
    for (int i = 0; i < M; i++) {
      __m256d ymm[UNROLL];

#pragma unroll(UNROLL)
      for (int x = 0; x < UNROLL; x++) {
        ymm[x] = _mm256_load_pd(C + i * ldc + j + x * 4);
      }

      __builtin_prefetch(A + i * lda, 0);

#pragma unroll(BLOCK_SIZE_K)
      for (int k = 0; k < BLOCK_SIZE_K; k++) {
#pragma unroll(UNROLL)
        for (int x = 0; x < UNROLL; x++) {
          // gcc cannot inline fmadd, so weak
          // ymm[x] =
          //     _mm256_fmadd_pd(_mm256_load_pd(B + k * lda + j + x * 4),
          //                     _mm256_broadcast_sd(A + i * lda + k), ymm[x]);
          ymm[x] = _mm256_add_pd(ymm[x], _mm256_mul_pd(_mm256_load_pd(B + k * ldb + j + x * 4),
                              _mm256_broadcast_sd(A + i * lda + k)));
        }
      }

#pragma unroll(UNROLL)
      for (int x = 0; x < UNROLL; x++) {
        _mm256_store_pd(C + i * ldc + j + x * 4, ymm[x]);
      }
    }
  }
}



// C = A +/- B
static inline __attribute__((always_inline)) void matrix_add(bool add, 
  int lda, int ldb, int ldc, int n, double *__restrict__ A, double *__restrict__ B,
    double *__restrict__ C
) {
  const int ADD_UNROLL = 8;
  if (n >= 4 * ADD_UNROLL) {
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
  } else {
    if (add) {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          C[i * ldc + j] = A[i * lda + j] + B[i * ldb + j];
        }
      }
    } else {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          C[i * ldc + j] = A[i * lda + j] - B[i * ldb + j];
        }
      }
    }
  }
}


// C += A +/- B
static inline __attribute__((always_inline)) void matrix_add_to(bool add, 
  int lda, int ldb, int ldc, int n, double *__restrict__ A, double *__restrict__ B,
    double *__restrict__ C
) {
  const int ADD_UNROLL = 8;
  if (n >= 4 * ADD_UNROLL) {
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
  } else {
    if (add) {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          C[i * ldc + j] += A[i * lda + j] + B[i * ldb + j];
        }
      }
    } else {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          C[i * ldc + j] += A[i * lda + j] - B[i * ldb + j];
        }
      }
    }
  }
}

static double st_im_1[MAX_N * MAX_N], st_im_2[MAX_N * MAX_N],
    st_im_3[MAX_N * MAX_N], st_im_4[MAX_N * MAX_N], st_p_1[MAX_N * MAX_N], st_p_2[MAX_N * MAX_N];

static inline void do_block_strassen(
    int lda, int ldb, int ldc, int n, double *__restrict__ A, double *__restrict__ B,
    double *__restrict__ C, double *__restrict__ im_1,
    double *__restrict__ im_2, double *__restrict__ im_3, double *__restrict__ im_4,
    double *__restrict__ p_1, double *__restrict__ p_2) {

  // fprintf(stderr, "Strassen %d %p %p %p %d %d %d\n", n, A, B, C, lda, ldb, ldc);

  // n is small enough now
  if (n < BLOCK_SIZE_N) {
    __builtin_unreachable();
  }
  if (n == BLOCK_SIZE_N) {
    do_block_simd(lda, ldb, ldc, n, n, n, A, B, C);
    return;
  }

  // if (n == 1) {
  //   C[0] = A[0] * B[0];
  //   return;
  // }

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

static double A_buf[MAX_N * MAX_N], B_buf[MAX_N * MAX_N], C_buf[MAX_N * MAX_N];

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *__restrict__ A, double *__restrict__ B,
                  double *__restrict__ C) {

  // (A*B)^T = B^T * A^T, so we can treat A, B, C in row-major format and
  // calculate C = C + B * A swap A and B for simplicity
  double *temp = A;
  A = B;
  B = temp;

  bool pad = false;
  int dim = lda;

  int remain = lda % BLOCK_SIZE_N;
  // do some padding
  if (remain > BLOCK_SIZE_N / 2) {
    pad = true;
    dim = (lda / BLOCK_SIZE_N + 1) * BLOCK_SIZE_N;
    for (int i = 0; i < lda; ++i) {
      __builtin_prefetch(A_buf + i * MAX_N, 1);
      __builtin_prefetch(B_buf + i * MAX_N, 1);
      __builtin_prefetch(C_buf + i * MAX_N, 1);
      __builtin_prefetch(A + i * lda, 0);
      __builtin_prefetch(B + i * lda, 0);
      __builtin_prefetch(C + i * lda, 0);
      memcpy(A_buf + i * MAX_N, A + i * lda, sizeof(double) * dim);
      memcpy(B_buf + i * MAX_N, B + i * lda, sizeof(double) * dim);
      memcpy(C_buf + i * MAX_N, C + i * lda, sizeof(double) * dim);
    }
  }

  double *__restrict__ _A = pad ? A_buf : A;
  double *__restrict__ _B = pad ? B_buf : B;
  double *__restrict__ _C = pad ? C_buf : C;
  int stride = pad ? MAX_N : lda;

#if ENABLE_STRASSEN
  if ((dim & (dim - 1)) == 0 && dim >= 32) {
    // power of 2 - use strassen
    do_block_strassen(stride, stride, stride, dim, _A, _B, _C, st_im_1, st_im_2, st_im_3, st_im_4, st_p_1, st_p_2);
  } else {
#endif
    /* For each block-row of A */
    for (int i = 0; i < dim; i += BLOCK_SIZE_M) {
      /* For each block-column of B */
      for (int j = 0; j < dim; j += BLOCK_SIZE_N) {
        /* Accumulate block dgemms into block of C */
        for (int k = 0; k < dim; k += BLOCK_SIZE_K) {
          /* Correct block dimenMons if block "goes off edge of" the matrix */
          int M = min(BLOCK_SIZE_M, dim - i);
          int N = min(BLOCK_SIZE_N, dim - j);
          int K = min(BLOCK_SIZE_K, dim - k);

          if (N == BLOCK_SIZE_N && K == BLOCK_SIZE_K) {
            /* Perform individual block dgemm */
            do_block_simd(stride, stride, stride, M, N, K, _A + i * stride + k,
                          _B + k * stride + j, _C + i * stride + j);
          } else {
            do_block_naive(lda, M, N, K, A + i * lda + k, B + k * lda + j,
                          C + i * lda + j);
          }
        }
      }
    }
#if ENABLE_STRASSEN
  }
#endif

  // copy data back
  if (pad) {
    for (int i = 0; i < lda; ++i) {
      __builtin_prefetch(A_buf + i * MAX_N, 0);
      __builtin_prefetch(B_buf + i * MAX_N, 0);
      __builtin_prefetch(C_buf + i * MAX_N, 0);
      __builtin_prefetch(A + i * lda, 1);
      __builtin_prefetch(B + i * lda, 1);
      __builtin_prefetch(C + i * lda, 1);
      memcpy(A + i * lda, A_buf + i * MAX_N, sizeof(double) * lda);
      memcpy(B + i * lda, B_buf + i * MAX_N, sizeof(double) * lda);
      memcpy(C + i * lda, C_buf + i * MAX_N, sizeof(double) * lda);
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
