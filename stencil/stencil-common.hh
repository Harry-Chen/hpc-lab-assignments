#ifndef STENCIL_COMMON_HH
#define STENCIL_COMMON_HH

#include <cstring>
#include <cstdio>
#include <cassert>

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

#endif