#include <cmath>
#include <omp.h>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <cstdio>

// #include "../../dbg-macro/dbg.h"

#include "common.h"

const char* version_name = "Optimized version (OpenMP)";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    if(grid_info->p_id == 0) {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    } else {
        grid_info->local_size_x = 0;
        grid_info->local_size_y = 0;
        grid_info->local_size_z = 0;
    }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

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
#define BT 4
#endif

#define BUF_DIM_X (BX + 2 * BT)
#define BUF_DIM_Y (BY + 2 * BT)
#define BUF_DIM_Z (BZ + 2 * BT)
#define BUF_SIZE (BUF_DIM_X * BUF_DIM_Y * BUF_DIM_Z)


static inline __attribute__((always_inline)) void stencil_7_kernel(
    ptr_t a0, ptr_t a1, ptr_t b0, ptr_t b1, ptr_t c0, ptr_t c1, 
    int x_start, int x_end,
    int y_start, int y_end,
    int z_start, int z_end
) {

    int ldx = BUF_DIM_X;
    int ldy = BUF_DIM_Y;
    int ldz = BUF_DIM_Z;

    for (int z = z_start; z < z_end; ++z) {
        for (int y = y_start; y < y_end; ++y) {
#pragma omp simd
            for (int x = x_start; x < x_end; ++x) {
                    data_t a7,b7,c7;

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
        }
    }
}


// // run stencil for BT steps on the buffer of each thread
// // each round the dimension thrinks by 1
// static inline __attribute__((always_inline)) void do_stencil_7_block(ptr_t A0, ptr_t A1, ptr_t B0, ptr_t B1, ptr_t C0, ptr_t C1) {
//     for (int t = BT - 1; t >= 0; --t) {
//         int x_dim = BX + 2 * t, y_dim = BY + 2 * t, z_dim = BZ + 2 * t;
//         int x_start = BT - t, y_start = BT - t, z_start = BT - t;
//         stencil_7_kernel(A0, A1, B0, B1, C0, C1, x_start, x_start + x_dim, y_start, y_start + y_dim, z_start, z_start + z_dim);
//         std::swap(A0, A1);
//         std::swap(B0, B1);
//         std::swap(C0, C1);
//     }
// }


// // copy data needed by a stencil kernel of BT steps to local buffer
// static inline __attribute__((always_inline)) void copy_to_buffer(
//     cptr_t a, cptr_t b, cptr_t c,
//     ptr_t a_buf, ptr_t b_buf, ptr_t c_buf,
//     int x_off, int y_off, int z_off,
//     int x_start, int y_start, int z_start,
//     int x_end, int y_end, int z_end,
//     int ldx, int ldy, int ldz
// ) {
//     // z_off: 0, 16, ..., 224, 240
//     // z_begin: 0, 12, ..., 220, 236
//     // z_stop: 20, 36, ..., 244, 256
//     // buf_z_start: 4, 0, ..., 0, 0
//     using std::min;
//     using std::max;
//     // begin index of x, y, z on src
//     int z_begin = max(z_off - BT, 0), z_stop = min(z_off + BZ + BT, z_end - z_start), buf_z_start = BT - (z_off - z_begin);
//     int y_begin = max(y_off - BT, 0), y_stop = min(y_off + BY + BT, y_end - y_start), buf_y_start = BT - (y_off - y_begin);
//     int x_begin = max(x_off - BT, 0), x_stop = min(x_off + BX + BT, x_end - x_start), buf_x_start = BT - (x_off - x_begin);
//     // data needed to be copied in each loop
//     size_t copy_size = sizeof(data_t) * (x_stop - x_begin);

//     for (int z = z_begin; z < z_stop; ++z) {
//         int src_z = z + z_start;
//         int buf_z = z - z_off + BT;
//         for (int y = y_begin; y < y_stop; ++y) {
//             int src_y = y + y_start;
//             int src_x = x_begin + x_start;
//             int buf_y = y - y_off + BT;
//             // dbg(buf_x, buf_y, buf_z, src_x, src_y, src_z);
//             int buf_off = INDEX(buf_x_start, buf_y, buf_z, BUF_DIM_X, BUF_DIM_Y);
//             int src_off = INDEX(src_x, src_y, src_z, ldx, ldy);
//             memcpy(a_buf + buf_off, a + src_off, copy_size);
//             memcpy(b_buf + buf_off, b + src_off, copy_size);
//             memcpy(c_buf + buf_off, c + src_off, copy_size);
//         }
//     }
// }

// // copy data from buffer to origin matrix
// static inline __attribute__((always_inline)) void copy_from_buffer(
//     ptr_t a, ptr_t b, ptr_t c,
//     cptr_t a_buf, cptr_t b_buf, cptr_t c_buf,
//     int x, int y, int z,
//     int x_start, int y_start, int z_start,
//     int ldx, int ldy, int ldz
// ) {
//     for (int z_ = z; z_ < z + BZ; ++z_) {
//         for (int y_ = y; y_ < y + BY; ++y_) {
//             int buf_z = (z_ - z_start) % BZ + BT;
//             int buf_y = (y_ - y_start) % BY + BT;
//             int buf_off = INDEX(BT, buf_y, buf_z, BUF_DIM_X, BUF_DIM_Y);
//             int src_off = INDEX(x, y_, z_, ldx, ldy);
//             memcpy(a + src_off, a_buf + buf_off, sizeof(data_t) * BX);
//             memcpy(b + src_off, b_buf + buf_off, sizeof(data_t) * BX);
//             memcpy(c + src_off, c_buf + buf_off, sizeof(data_t) * BX);
//         }
//     }
// }

void debug_buffer(ptr_t buf) {
    for (int z = 0; z < BUF_DIM_Z; ++z) {
        for (int y = 0; y < BUF_DIM_Y; ++y) {
            for (int x = 0; x < BUF_DIM_Z; ++x) {
                fprintf(stderr, "%.3lf ", buf[INDEX(x, y, z, BUF_DIM_X, BUF_DIM_Y)]);
            }
        }
    }
    fprintf(stderr, "\n");
}

ptr_t stencil_7(ptr_t A0, ptr_t A1, ptr_t B0, ptr_t B1, ptr_t C0, ptr_t C1, const dist_grid_info_t *grid_info, int nt) {
    ptr_t bufferx[2] = {A0, A1};
    ptr_t buffery[2] = {B0, B1};
    ptr_t bufferz[2] = {C0, C1};

    // check constraints
    assert(grid_info->local_size_x % BX == 0);
    assert(grid_info->local_size_y % BY == 0);
    assert(grid_info->local_size_z % BZ == 0);


    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    int t, k;
    ptr_t ret = A0;
    for (t = 0, k = 0; t < nt; t += BT, k++) {

        ptr_t a0 = bufferx[k % 2];
        ptr_t a1 = bufferx[(k + 1) % 2];

        ptr_t b0 = buffery[k % 2];
        ptr_t b1 = buffery[(k + 1) % 2];

        ptr_t c0 = bufferz[k % 2];
        ptr_t c1 = bufferz[(k + 1) % 2];
        
#pragma omp parallel for collapse(2) schedule(static)
        for (int z = z_start; z < z_end; z += BZ) {
            for (int y = y_start; y < y_end; y += BY) {
                for (int x = x_start; x < x_end; x += BX) {
                    int z_off = z - z_start, y_off = y - y_start, x_off = x - x_start;
                    using std::min;
                    using std::max;
                    int z_begin = max(z_off - BT, 0), z_stop = min(z_off + BZ + BT, z_end - z_start), buf_z_start = BT - (z_off - z_begin), buf_z_end = BUF_DIM_Z - (z_off + BZ + BT - z_stop);
                    int y_begin = max(y_off - BT, 0), y_stop = min(y_off + BY + BT, y_end - y_start), buf_y_start = BT - (y_off - y_begin), buf_y_end = BUF_DIM_Y - (y_off + BY + BT - y_stop);
                    int x_begin = max(x_off - BT, 0), x_stop = min(x_off + BX + BT, x_end - x_start), buf_x_start = BT - (x_off - x_begin), buf_x_end = BUF_DIM_X - (x_off + BX + BT - x_stop);
                    // allocate contiguous buffer
                    data_t a0_buf[BUF_SIZE] = {}, b0_buf[BUF_SIZE] = {}, c0_buf[BUF_SIZE] = {}; // make sure it is zero initialized
                    data_t a1_buf[BUF_SIZE], b1_buf[BUF_SIZE], c1_buf[BUF_SIZE];
                    // data needed to be copied in each loop
                    size_t copy_size = sizeof(data_t) * (x_stop - x_begin);
                    // pack a0, b0, c0 (and BT level of neighbours) to buffer
                    for (int z_ = z_begin; z_ < z_stop; ++z_) {
                        int src_z = z_ + z_start;
                        int buf_z = z_ - z_off + BT;
                        for (int y_ = y_begin; y_ < y_stop; ++y_) {
                            int src_y = y_ + y_start;
                            int src_x = x_begin + x_start;
                            int buf_y = y_ - y_off + BT;
                            // dbg(buf_x, buf_y, buf_z, src_x, src_y, src_z);
                            int buf_off = INDEX(buf_x_start, buf_y, buf_z, BUF_DIM_X, BUF_DIM_Y);
                            int src_off = INDEX(src_x, src_y, src_z, ldx, ldy);
                            memcpy(a0_buf + buf_off, a0 + src_off, copy_size);
                            memcpy(b0_buf + buf_off, b0 + src_off, copy_size);
                            memcpy(c0_buf + buf_off, c0 + src_off, copy_size);
                        }
                    }
                    // run stencil kernel on current block for BT rounds
                    // a0_buf -> a1_buf -> a0_buf -> a1_buf -> a0_buf
                    ptr_t A0 = a0, A1 = a1, B0 = b0, B1 = b1, C0 = c0, C1 = c1;
                    for (int t = BT - 1; t >= 0; --t) {
                        int x_start = max(BT - t, buf_x_start), y_start = max(BT - t, buf_y_start), z_start = max(BT - t, buf_z_start);
                        int x_end = min(BX + 2 * t, buf_x_end), y_end = min(BY + 2 * t, buf_y_end), z_end = min(BZ + 2 * t, buf_z_end);
                        stencil_7_kernel(A0, A1, B0, B1, C0, C1, x_start, x_end, y_start, y_end, z_start, z_end);
                        std::swap(A0, A1);
                        std::swap(B0, B1);
                        std::swap(C0, C1);
                    }
                    // do_stencil_7_block(a0_buf, a1_buf, b0_buf, b1_buf, c0_buf, c1_buf);
                    // debug_buffer(a0_buf);
                    // debug_buffer(b0_buf);
                    // debug_buffer(c0_buf);
                    // copy back from buffer to a1, b1, c1
                    for (int z_ = z; z_ < z + BZ; ++z_) {
                        for (int y_ = y; y_ < y + BY; ++y_) {
                            int buf_z = (z_ - z_start) % BZ + BT;
                            int buf_y = (y_ - y_start) % BY + BT;
                            int buf_off = INDEX(BT, buf_y, buf_z, BUF_DIM_X, BUF_DIM_Y);
                            int src_off = INDEX(x, y_, z_, ldx, ldy);
                            memcpy(a1 + src_off, a0_buf + buf_off, sizeof(data_t) * BX);
                            memcpy(b1 + src_off, b0_buf + buf_off, sizeof(data_t) * BX);
                            memcpy(c1 + src_off, c0_buf + buf_off, sizeof(data_t) * BX);
                        }
                    }
                    ret = a1;
                }
            }
        }
    }

    // deal with remaining steps
    if (unlikely(t < nt)) {
        for (int t_ = k; t_ < nt - t + k; ++t_) {
            cptr_t a0 = bufferx[t_ % 2];
            ptr_t a1 = bufferx[(t_ + 1) % 2];

            cptr_t b0 = buffery[t_ % 2];
            ptr_t b1 = buffery[(t_ + 1) % 2];

            cptr_t c0 = bufferz[t_ % 2];
            ptr_t c1 = bufferz[(t_ + 1) % 2];
            
            for (int z = z_start; z < z_end; ++z) {
                for (int y = y_start; y < y_end; ++y) {
                    for (int x = x_start; x < x_end; ++x) {

                        data_t a7,b7,c7;

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
                }
            }
            ret = a1;
        }
    }

    return ret;
}
