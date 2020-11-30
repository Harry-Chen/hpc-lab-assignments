#ifndef STENCIL_COMMON_HH
#define STENCIL_COMMON_HH

#include <cstring>
#include <cstdio>
#include <cassert>
#include <immintrin.h>

#include "common.h"

#define REQUIRE(COND, MSG) assert(((void) MSG, COND))

// OMP blocking
#if !defined BX
#define BX 16
#endif
#if !defined BY
#define BY 16
#endif
#if !defined BZ
#define BZ 16
#endif
#if !defined BT
#define BT 3
#endif

#define BUF_DIM_X (BX + 2 * BT)
#define BUF_DIM_Y (BY + 2 * BT)
#define BUF_DIM_Z (BZ + 2 * BT)
#define BUF_SIZE (BUF_DIM_X * BUF_DIM_Y * BUF_DIM_Z)

using std::max;
using std::min;

// copy data from buffer to origin matrix
static inline __attribute__((always_inline)) void copy_from_buffer(
    ptr_t a, ptr_t b, ptr_t c,
    cptr_t a_buf, cptr_t b_buf, cptr_t c_buf,
    int x, int y, int z,
    int x_start, int y_start, int z_start,
    int ldx, int ldy, int ldz
) {
    for (int zz = z; zz < z + BZ; ++zz) {
        for (int yy = y; yy < y + BY; ++yy) {
            int buf_z = (zz - z_start) % BZ + BT;
            int buf_y = (yy - y_start) % BY + BT;
            int buf_off = INDEX(BT, buf_y, buf_z, BUF_DIM_X, BUF_DIM_Y);
            int src_off = INDEX(x, yy, zz, ldx, ldy);
            memcpy(a + src_off, a_buf + buf_off, sizeof(data_t) * BX);
            memcpy(b + src_off, b_buf + buf_off, sizeof(data_t) * BX);
            memcpy(c + src_off, c_buf + buf_off, sizeof(data_t) * BX);
        }
    }
}


// print thread local buffer, debug purpose
inline void debug_buffer(ptr_t buf, int dim, int lda, int start = 0) {
    fprintf(stderr, "\n\nstart ");
    for (int z = start; z < start + dim; ++z) {
        for (int y = start; y < start + dim; ++y) {
            for (int x = start; x < start + dim; ++x) {
                fprintf(stderr, "%.3lf ", buf[INDEX(x, y, z, lda, lda)]);
            }
        }
    }
    fprintf(stderr, "end\n\n");
    fflush(stderr);
}


template <int BUFFER_CLEAR_EXT = 3>
static inline __attribute__((always_inline)) void clear_buffers(ptr_t a) {
    for (int zz = BT - BUFFER_CLEAR_EXT; zz < BZ + BT + BUFFER_CLEAR_EXT; ++zz) {
        for (int yy = BT - BUFFER_CLEAR_EXT; yy < BY + BT + BUFFER_CLEAR_EXT; ++yy) {
            bzero(a + INDEX(BT - BUFFER_CLEAR_EXT, yy, zz, BUF_DIM_X, BUF_DIM_Y), sizeof(data_t) * (BX + 2 * BUFFER_CLEAR_EXT));
        }
    }
}


// stencil coefficients
__m256d al[7], bl[7], cl[7];

void inline __attribute__((always_inline)) load_stencil_coefficients() {
    al[0] = _mm256_set1_pd((double)ALPHA_ZZZ);
    al[1] = _mm256_set1_pd((double)ALPHA_NZZ);
    al[2] = _mm256_set1_pd((double)ALPHA_PZZ);
    al[3] = _mm256_set1_pd((double)ALPHA_ZNZ);
    al[4] = _mm256_set1_pd((double)ALPHA_ZPZ);
    al[5] = _mm256_set1_pd((double)ALPHA_ZZN);
    al[6] = _mm256_set1_pd((double)ALPHA_ZZP);

    bl[0] = _mm256_set1_pd((double)ALPHA_PNZ);
    bl[1] = _mm256_set1_pd((double)ALPHA_NPZ);
    bl[2] = _mm256_set1_pd((double)ALPHA_PPZ);
    bl[3] = _mm256_set1_pd((double)ALPHA_NZN);
    bl[4] = _mm256_set1_pd((double)ALPHA_PZN);
    bl[5] = _mm256_set1_pd((double)ALPHA_PZP);
    bl[6] = _mm256_set1_pd((double)ALPHA_NZP);

    cl[0] = _mm256_set1_pd((double)ALPHA_PNN);
    cl[1] = _mm256_set1_pd((double)ALPHA_PPN);
    cl[2] = _mm256_set1_pd((double)ALPHA_PPN);
    cl[3] = _mm256_set1_pd((double)ALPHA_NNP);
    cl[4] = _mm256_set1_pd((double)ALPHA_PNP);
    cl[5] = _mm256_set1_pd((double)ALPHA_NPP);
    cl[6] = _mm256_set1_pd((double)ALPHA_PPP);
}


// actual computation of stencil
template<bool USE_SIMD = true>
void inline __attribute__((always_inline)) stencil_inner_loop(cptr_t a0, ptr_t a1, cptr_t b0, ptr_t b1, cptr_t c0, ptr_t c1, int x_start, int x_end, int y, int z, int ldx, int ldy, int ldz) {

    // SIMD rounds
    int ks = (x_end - x_start) / 4;

    if constexpr (USE_SIMD) {
        for (int k = 0, x = x_start; k < ks; k++, x += 4) {
            // load a0
            __m256d res = _mm256_setzero_pd();
            __m256d in1 = _mm256_loadu_pd(a0 + INDEX(x, y, z, ldx, ldy));
            __m256d in2 = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z, ldx, ldy));
            __m256d in3 = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z, ldx, ldy));
            __m256d in4 = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z, ldx, ldy));
            __m256d in5 = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z, ldx, ldy));
            __m256d in6 = _mm256_loadu_pd(a0 + INDEX(x, y, z - 1, ldx, ldy));
            __m256d in7 = _mm256_loadu_pd(a0 + INDEX(x, y, z + 1, ldx, ldy));

            // a7
            res = _mm256_mul_pd  (al[0], in1);
            res = _mm256_fmadd_pd(al[1], in2, res);
            res = _mm256_fmadd_pd(al[2], in3, res);
            res = _mm256_fmadd_pd(al[3], in4, res);
            res = _mm256_fmadd_pd(al[4], in5, res);
            res = _mm256_fmadd_pd(al[5], in6, res);
            res = _mm256_fmadd_pd(al[6], in7, res);

            // load b0
            __m256d res2 = _mm256_setzero_pd();
            in1 = _mm256_loadu_pd(b0 + INDEX(x, y, z, ldx, ldy));
            in2 = _mm256_loadu_pd(b0 + INDEX(x - 1, y, z, ldx, ldy));
            in3 = _mm256_loadu_pd(b0 + INDEX(x + 1, y, z, ldx, ldy));
            in4 = _mm256_loadu_pd(b0 + INDEX(x, y - 1, z, ldx, ldy));
            in5 = _mm256_loadu_pd(b0 + INDEX(x, y + 1, z, ldx, ldy));
            in6 = _mm256_loadu_pd(b0 + INDEX(x, y, z - 1, ldx, ldy));
            in7 = _mm256_loadu_pd(b0 + INDEX(x, y, z + 1, ldx, ldy));

            // b7
            res2 = _mm256_mul_pd  (bl[0], in1);
            res2 = _mm256_fmadd_pd(bl[1], in2, res2);
            res2 = _mm256_fmadd_pd(bl[2], in3, res2);
            res2 = _mm256_fmadd_pd(bl[3], in4, res2);
            res2 = _mm256_fmadd_pd(bl[4], in5, res2);
            res2 = _mm256_fmadd_pd(bl[5], in6, res2);
            res2 = _mm256_fmadd_pd(bl[6], in7, res2);

            // c0
            __m256d res3 = _mm256_setzero_pd();
            in1 = _mm256_loadu_pd(c0 + INDEX(x, y, z, ldx, ldy));
            in2 = _mm256_loadu_pd(c0 + INDEX(x - 1, y, z, ldx, ldy));
            in3 = _mm256_loadu_pd(c0 + INDEX(x + 1, y, z, ldx, ldy));
            in4 = _mm256_loadu_pd(c0 + INDEX(x, y - 1, z, ldx, ldy));
            in5 = _mm256_loadu_pd(c0 + INDEX(x, y + 1, z, ldx, ldy));
            in6 = _mm256_loadu_pd(c0 + INDEX(x, y, z - 1, ldx, ldy));
            in7 = _mm256_loadu_pd(c0 + INDEX(x, y, z + 1, ldx, ldy));

            // c7
            res3 = _mm256_mul_pd  (cl[0], in1);
            res3 = _mm256_fmadd_pd(cl[1], in2, res3);
            res3 = _mm256_fmadd_pd(cl[2], in3, res3);
            res3 = _mm256_fmadd_pd(cl[3], in4, res3);
            res3 = _mm256_fmadd_pd(cl[4], in5, res3);
            res3 = _mm256_fmadd_pd(cl[5], in6, res3);
            res3 = _mm256_fmadd_pd(cl[6], in7, res3);

            // a1
            __m256d res4 = _mm256_mul_pd(res2, res3);
            __m256d res5 = _mm256_add_pd(res2, res3);
            res5 = _mm256_div_pd(res4, res5);
            __m256d outres =  _mm256_add_pd(res, res5);
            _mm256_storeu_pd(a1 + INDEX(x, y, z, ldx, ldy), outres);

            // b1
            res4 = _mm256_mul_pd(res, res3);
            res5 = _mm256_add_pd(res, res3);
            res5 = _mm256_div_pd(res4, res5);
            __m256d outres2 =  _mm256_add_pd(res2, res5);
            _mm256_storeu_pd(b1 + INDEX(x, y, z, ldx, ldy), outres2);

            // c1
            res4 = _mm256_mul_pd(res, res2);
            res5 = _mm256_add_pd(res, res2);
            res5 = _mm256_div_pd(res4, res5);
            __m256d outres3 =  _mm256_add_pd(res3, res5);
            _mm256_storeu_pd(c1 + INDEX(x, y, z, ldx, ldy), outres3);
        }
    } else {
        ks = 0;
    }

    // remaining rounds (e.g. on halo, or SIMD disabled)
    for (int x = x_start + 4 * ks; x < x_end; x ++) {
        data_t a7, b7, c7;
        a7 \
            = ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] \
            + ALPHA_NZZ * a0[INDEX(x-1, y, z, ldx, ldy)] \
            + ALPHA_PZZ * a0[INDEX(x+1, y, z, ldx, ldy)] \
            + ALPHA_ZNZ * a0[INDEX(x, y-1, z, ldx, ldy)] \
            + ALPHA_ZPZ * a0[INDEX(x, y+1, z, ldx, ldy)] \
            + ALPHA_ZZN * a0[INDEX(x, y, z-1, ldx, ldy)] \
            + ALPHA_ZZP * a0[INDEX(x, y, z+1, ldx, ldy)];
        b7 \
            = ALPHA_PNZ * b0[INDEX(x, y, z, ldx, ldy)] \
            + ALPHA_NPZ * b0[INDEX(x-1, y, z, ldx, ldy)] \
            + ALPHA_PPZ * b0[INDEX(x+1, y, z, ldx, ldy)] \
            + ALPHA_NZN * b0[INDEX(x, y-1, z, ldx, ldy)] \
            + ALPHA_PZN * b0[INDEX(x, y+1, z, ldx, ldy)] \
            + ALPHA_PZP * b0[INDEX(x, y, z-1, ldx, ldy)] \
            + ALPHA_NZP * b0[INDEX(x, y, z+1, ldx, ldy)];
        c7 \
            = ALPHA_PNN * c0[INDEX(x, y, z, ldx, ldy)] \
            + ALPHA_PPN * c0[INDEX(x-1, y, z, ldx, ldy)] \
            + ALPHA_PPN * c0[INDEX(x+1, y, z, ldx, ldy)] \
            + ALPHA_NNP * c0[INDEX(x, y-1, z, ldx, ldy)] \
            + ALPHA_PNP * c0[INDEX(x, y+1, z, ldx, ldy)] \
            + ALPHA_NPP * c0[INDEX(x, y, z-1, ldx, ldy)] \
            + ALPHA_PPP * c0[INDEX(x, y, z+1, ldx, ldy)];
        a1[INDEX(x, y, z, ldx, ldy)] = a7  +  (b7 * c7) / (b7 + c7); //sqrt
        b1[INDEX(x, y, z, ldx, ldy)] = b7  +  (a7 * c7) / (a7 + c7); //sqrt
        c1[INDEX(x, y, z, ldx, ldy)] = c7  +  (a7 * b7) / (a7 + b7); //sqrt
    }
};

struct stencil_neighbour_t {
    int rank = -1;
    int recv_type = -1;
    int send_type = -1;
};

#define NEIGHBOUR_NUM 2


// parameters for tiling
#define TX 16
#define TY 8
#define TZ 16
#define TT 16


struct do_nothing_t {
    void operator() (...) {}
};

// time skew blocking
// reference: https://github.com/shoaibkamil/stencilprobe/blob/master/probe_heat_timeskew.c
// reference: https://people.csail.mit.edu/skamil/projects/stencilprobe/
template<bool USE_SIMD = true, typename F = do_nothing_t>
ptr_t inline __attribute__((always_inline)) stencil_time_skew(
    int x_start, int x_end, int y_start, int y_end, int z_start, int z_end, 
    int nt, int ldx, int ldy, int ldz,
    ptr_t bufferx[2], ptr_t buffery[2], ptr_t bufferz[2],
    F&& mpi_callback = do_nothing_t()
) {

    // blocking on t dimension
    for (int t = 0; t < nt; t += TT) {

        mpi_callback(bufferx[t % 2], buffery[t % 2], bufferz[t % 2]);

        // blocking on y dimension
        for (int y = y_start; y < y_end; y += TY) {

            int neg_y_slope = y == 1 ? 0 : 1;
            int pos_y_slope = y == y_end - TY ? 0 : -1;

            // do actual stencil
            for (int tt = t; tt < min(t + TT, nt); tt++) {
                int y_begin = max(y_start, y - tt * neg_y_slope);
                int y_stop = max(y_start, y + TY + tt * pos_y_slope);

                cptr_t a0 = bufferx[tt % 2];
                ptr_t a1 = bufferx[(tt + 1) % 2];

                cptr_t b0 = buffery[tt % 2];
                ptr_t b1 = buffery[(tt + 1) % 2];

                cptr_t c0 = bufferz[tt % 2];
                ptr_t c1 = bufferz[(tt + 1) % 2];

#pragma omp parallel for collapse(1) schedule(static)
                for (int zz = z_start; zz < z_end; zz++) {
                    for (int yy = y_begin; yy < y_stop; yy++) {
                        stencil_inner_loop<USE_SIMD>(a0, a1, b0, b1, c0, c1, x_start, x_end, yy, zz, ldx, ldy, ldz);
                    }
                }
            }
        }
    }

    return bufferx[nt % 2];
}


#endif
