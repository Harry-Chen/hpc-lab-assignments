#pragma once

#if ENABLE_STRASSEN

#include <immintrin.h>

#ifndef MAX_N
#define MAX_N 2000
#endif

// B += A
template<int ADD_UNROLL = 8>
static inline __attribute__((always_inline)) void matrix_add_single(int lda, int ldb, int n, const double *__restrict__ const A, double *__restrict__ const B) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; j += 4 * ADD_UNROLL) {
#pragma unroll(ADD_UNROLL)
      for (int x = 0; x < ADD_UNROLL; x++) {
        _mm256_store_pd(B + i * ldb + j + x * 4, _mm256_add_pd(_mm256_load_pd(A + i * lda + j + x * 4), _mm256_load_pd(B + i * ldb + j + x * 4)));
      }
    }
  }
}

// C =/+= A +/- B, n >= BLOCK_SIZE_N
template<bool add, bool override = true, int ADD_UNROLL = 8>
static inline __attribute__((always_inline)) void matrix_add(
  int lda, int ldb, int ldc, int n, const double *__restrict__ const A, const double *__restrict__ const B, double *__restrict__ const C
) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; j += 4 * ADD_UNROLL) {
#pragma unroll(ADD_UNROLL)
      for (int x = 0; x < ADD_UNROLL; x++) {
        if constexpr (!override) {
          if constexpr (add) {
            _mm256_store_pd(C + i * ldc + j + x * 4, _mm256_add_pd(_mm256_load_pd(C + i * ldc + j + x * 4), _mm256_add_pd(_mm256_load_pd(A + i * lda + j + x * 4), _mm256_load_pd(B + i * ldb + j + x * 4))));
          } else {
            _mm256_store_pd(C + i * ldc + j + x * 4, _mm256_add_pd(_mm256_load_pd(C + i * ldc + j + x * 4), _mm256_sub_pd(_mm256_load_pd(A + i * lda + j + x * 4), _mm256_load_pd(B + i * ldb + j + x * 4))));
          }
        } else {
          if constexpr (add) {
            _mm256_store_pd(C + i * ldc + j + x * 4, _mm256_add_pd(_mm256_load_pd(A + i * lda + j + x * 4), _mm256_load_pd(B + i * ldb + j + x * 4)));
          } else {
            _mm256_store_pd(C + i * ldc + j + x * 4, _mm256_sub_pd(_mm256_load_pd(A + i * lda + j + x * 4), _mm256_load_pd(B + i * ldb + j + x * 4)));
          }
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
template<int BLOCK_SIZE_N>
static inline __attribute__((always_inline)) void do_block_strassen(
    int lda, int ldb, int ldc, int n, const double *__restrict__ const A, const double *__restrict__ const B,
    double *__restrict__ const C, double *__restrict__ const im_1,
    double *__restrict__ const im_2, double *__restrict__ const im_3, double *__restrict__ const im_4,
    double *__restrict__ const p_1, double *__restrict__ const p_2) {

  // fprintf(stderr, "Strassen %d %p %p %p %d %d %d\n", n, A, B, C, lda, ldb, ldc);

  // n is small enough now
  if (unlikely(n < BLOCK_SIZE_N)) {
    __builtin_unreachable();
  } else if (n == BLOCK_SIZE_N) {
    avx_kernel<true, true, BLOCK_SIZE_N>(lda, ldb, ldc, A, B, C);
    return;
  }

  int m = n / 2;
  
  // split sub-matricies
  const double *__restrict__ const A11 = A;
  const double *__restrict__ const A12 = A + m;
  const double *__restrict__ const A21 = A + m * lda;
  const double *__restrict__ const A22 = A + m * lda + m;
  const double *__restrict__ const B11 = B;
  const double *__restrict__ const B12 = B + m;
  const double *__restrict__ const B21 = B + m * ldb;
  const double *__restrict__ const B22 = B + m * ldb + m;
  double *__restrict__ const C11 = C;
  double *__restrict__ const C12 = C + m;
  double *__restrict__ const C21 = C + m * ldc;
  double *__restrict__ const C22 = C + m * ldc + m;
  // intermediate values offset
  // variables with trailing underscore means it would be used in recursion
  // im_1: B12m22, A11p12, A21p22, im_1_
  // im_2: B21m11, A11p22, B11p22, im_2_
  // im_3: A12m22, B21p22, A21m11, im_3_
  // im_4: B11p12, p_1_, EMPTY, im_4_
  // p_1: P1, P2, P3, P4
  // p_2: P5, P6, P7, p_2_
  double *__restrict__ const B12m22 = im_1;
  double *__restrict__ const A11p12 = im_1 + m;
  double *__restrict__ const A21p22 = im_1 + m * MAX_N;
  double *__restrict__ const B21m11 = im_2;
  double *__restrict__ const A11p22 = im_2 + m;
  double *__restrict__ const B11p22 = im_2 + m * MAX_N;
  double *__restrict__ const A12m22 = im_3;
  double *__restrict__ const B21p22 = im_3 + m;
  double *__restrict__ const A21m11 = im_3 + m * MAX_N;
  double *__restrict__ const B11p12 = im_4;
  double *__restrict__ const P1 = p_1;
  double *__restrict__ const P2 = p_1 + m;
  double *__restrict__ const P3 = p_1 + m * MAX_N;
  double *__restrict__ const P4 = p_1 + m * MAX_N + m;
  double *__restrict__ const P5 = p_2;
  double *__restrict__ const P6 = p_2 + m;
  double *__restrict__ const P7 = p_2 + m * MAX_N;
  // smaller matricies
  double *__restrict__ _p_1_ = im_4 + m;
  double *__restrict__ _p_2_ = p_2 + m * MAX_N + m;
  double *__restrict__ _im_1_ = im_1 + m * MAX_N + m;
  double *__restrict__ _im_2_ = im_2 + m * MAX_N + m;
  double *__restrict__ _im_3_ = im_3 + m * MAX_N + m;
  double *__restrict__ _im_4_ = im_4 + m * MAX_N + m;

  // calculate intermediate values
  matrix_add<false>(ldb, ldb, MAX_N, m, B12, B22, B12m22);
  matrix_add<true >(lda, lda, MAX_N, m, A11, A12, A11p12);
  matrix_add<true >(lda, lda, MAX_N, m, A21, A22, A21p22);
  matrix_add<false>(ldb, ldb, MAX_N, m, B21, B11, B21m11);
  matrix_add<true >(lda, lda, MAX_N, m, A11, A22, A11p22);
  matrix_add<true >(ldb, ldb, MAX_N, m, B11, B22, B11p22);
  matrix_add<false>(lda, lda, MAX_N, m, A12, A22, A12m22);
  matrix_add<true >(ldb, ldb, MAX_N, m, B21, B22, B21p22);
  matrix_add<false>(lda, lda, MAX_N, m, A21, A11, A21m11);
  matrix_add<true >(ldb, ldb, MAX_N, m, B11, B12, B11p12);

  // P1 = Strassen(A11,B12 − B22)
  do_block_strassen<BLOCK_SIZE_N>(lda, MAX_N, MAX_N, m, A11, B12m22, P1, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P2 = Strassen(A11 + A12,B22)
  do_block_strassen<BLOCK_SIZE_N>(MAX_N, ldb, MAX_N, m, A11p12, B22, P2, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P3 = Strassen(A21 + A22,B11)
  do_block_strassen<BLOCK_SIZE_N>(MAX_N, ldb, MAX_N, m, A21p22, B11, P3, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P4 = Strassen(A22,B21 − B11)
  do_block_strassen<BLOCK_SIZE_N>(lda, MAX_N, MAX_N, m, A22, B21m11, P4, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P5 = Strassen(A11 + A22,B11 + B22)
  do_block_strassen<BLOCK_SIZE_N>(MAX_N, MAX_N, MAX_N, m, A11p22, B11p22, P5, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P6 = Strassen(A12 − A22,B21 + B22)
  do_block_strassen<BLOCK_SIZE_N>(MAX_N, MAX_N, MAX_N, m, A12m22, B21p22, P6, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);
  // P7 = Strassen(A21 − A11,B11 + B12)
  do_block_strassen<BLOCK_SIZE_N>(MAX_N, MAX_N, MAX_N, m, A21m11, B11p12, P7, _im_1_, _im_2_, _im_3_, _im_4_, _p_1_, _p_2_);

  // C11 = (P5 + P4) + (P6 - P2)
  matrix_add<true>(MAX_N, MAX_N, ldc, m, P5, P4, C11);
  matrix_add<false, false>(MAX_N, MAX_N, ldc, m, P6, P2, C11);
  // C12 = P1 + P2
  matrix_add<true>(MAX_N, MAX_N, ldc, m, P1, P2, C12);
  // C21 = P3 + P4
  matrix_add<true>(MAX_N, MAX_N, ldc, m, P3, P4, C21);
  // C22 = (P1 + P7) + (P5 - P3)
  matrix_add<true>(MAX_N, MAX_N, ldc, m, P1, P7, C22);
  matrix_add<false, false>(MAX_N, MAX_N, ldc, m, P5, P3, C22);

}

#endif
