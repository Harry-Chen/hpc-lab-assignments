#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <numa.h>
#include <omp.h>

#include <algorithm>

#include "stencil-common.hh"

extern "C" const char* version_name = "Optimized version (MPI + OpenMP)";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    
    int ranks = grid_info->p_num, rank = grid_info->p_id;

    // retrieve local rank and do some checking
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    int local_size, local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    int nodes = numa_num_task_nodes();
    int cpus = numa_num_task_cpus();
    int threads = omp_get_max_threads();

    if (rank == 0) {
        REQUIRE(grid_info->global_size_z % ranks == 0, "Rank number must divide z dimension");
        if (local_size != nodes) {
            fprintf(stderr, "[WARNING] rank num %d on each node is not optimal, use %d for best performance.\n", local_size, nodes);
        }
        if (threads > cpus) {
            fprintf(stderr, "[WARNING] thread %d for each rank is not optimal, use %d for best performance.\n", threads, cpus);
        }
    }

    // bind process to NUMA node (OpenMP threads binds automatically)
    if (ranks >= 2) {
        auto cpu_mask = numa_allocate_nodemask();
        numa_bitmask_setbit(cpu_mask, local_rank % nodes);
        numa_bind(cpu_mask);
    }

    // slice array along z axis
    int local_size_x = grid_info->global_size_x;
    int local_size_y = grid_info->global_size_y;
    int local_size_z = grid_info->global_size_z / ranks;
    int offset_z = rank * local_size_z;

    int halo_size_x = 1;
    int halo_size_y = 1;
    int halo_size_z = grid_info->global_size_z < TRIVIAL_METHOD_THRESHOLD_MPI ? get_t_block_size(grid_info->global_size_x) : BT + 1;

    // config grid info
    grid_info->local_size_x = local_size_x;
    grid_info->local_size_y = local_size_y;
    grid_info->local_size_z = local_size_z;

    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = offset_z;

    grid_info->halo_size_x = halo_size_x;
    grid_info->halo_size_y = halo_size_y;
    grid_info->halo_size_z = halo_size_z;

    // fprintf(stderr, "Rank %d offset x %d y %d z %d\n", rank, grid_info->offset_x, grid_info->offset_y, grid_info->offset_z);
    load_stencil_coefficients();
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

static MPI_Request send_upper_req[3], recv_lower_req[3], send_lower_req[3], recv_upper_req[3];

inline void exchange_data(cptr_t A, cptr_t B, cptr_t C, int z_start, int z_end, int ldx, int ldy, int height, const dist_grid_info_t *grid_info) {
    // send last several elements
    int rank = grid_info->p_id;
    int upper_rank = grid_info->p_id + 1;
    int lower_rank = grid_info->p_id - 1;

    // receive from lower
    if (rank != 0) {
        MPI_Irecv((void*)(A + INDEX(0, 0, 0, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, lower_rank, 0, MPI_COMM_WORLD, recv_lower_req);
        MPI_Irecv((void*)(B + INDEX(0, 0, 0, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, lower_rank, 1, MPI_COMM_WORLD, recv_lower_req + 1);
        MPI_Irecv((void*)(C + INDEX(0, 0, 0, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, lower_rank, 2, MPI_COMM_WORLD, recv_lower_req + 2);
    } 
    
    // send to upper
    if (rank != grid_info->p_num - 1) {
        MPI_Isend((void*)(A + INDEX(0, 0, z_end - height, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, upper_rank, 0, MPI_COMM_WORLD, send_upper_req);
        MPI_Isend((void*)(B + INDEX(0, 0, z_end - height, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, upper_rank, 1, MPI_COMM_WORLD, send_upper_req + 1);
        MPI_Isend((void*)(C + INDEX(0, 0, z_end - height, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, upper_rank, 2, MPI_COMM_WORLD, send_upper_req + 2);
    }

    // receive from upper
    if (rank != grid_info->p_num - 1) {
        MPI_Irecv((void*)(A + INDEX(0, 0, z_end, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, upper_rank, 3, MPI_COMM_WORLD, recv_upper_req);
        MPI_Irecv((void*)(B + INDEX(0, 0, z_end, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, upper_rank, 4, MPI_COMM_WORLD, recv_upper_req + 1);
        MPI_Irecv((void*)(C + INDEX(0, 0, z_end, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, upper_rank, 5, MPI_COMM_WORLD, recv_upper_req + 2);
    } 
    
    // send to lower
    if (rank != 0) {
        MPI_Isend((void*)(A + INDEX(0, 0, z_start, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, lower_rank, 3, MPI_COMM_WORLD, send_lower_req);
        MPI_Isend((void*)(B + INDEX(0, 0, z_start, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, lower_rank, 4, MPI_COMM_WORLD, send_lower_req + 1);
        MPI_Isend((void*)(C + INDEX(0, 0, z_start, ldx, ldy)), ldx * ldy * height, MPI_DOUBLE, lower_rank, 5, MPI_COMM_WORLD, send_lower_req + 2);
    }

    // wait for lower communication
    if (rank != 0) {
        MPI_Waitall(3, recv_lower_req, MPI_STATUSES_IGNORE);
        MPI_Waitall(3, send_lower_req, MPI_STATUSES_IGNORE);
    }
    
    // wait for upper communication
    if (rank != grid_info->p_num - 1) {
        MPI_Waitall(3, recv_upper_req, MPI_STATUSES_IGNORE);
        MPI_Waitall(3, send_upper_req, MPI_STATUSES_IGNORE);
    }
}


ptr_t inline __attribute__((always_inline)) stencil_trivial(
    int x_start, int x_end, int y_start, int y_end, int z_start, int z_end, 
    int nt, int ldx, int ldy, int ldz,
    ptr_t bufferx[2], ptr_t buffery[2], ptr_t bufferz[2],
    bool has_down, bool has_up, const dist_grid_info_t *grid_info
) {
    
    // return array
    ptr_t ret = bufferx[0];
    // fused rounds (after each round, a1 will store BT rounds of stencil on a0, and a0 will be garbage)
    int t_fused = nt / BT;

    // main stencil loop
    for (int k = 0, t = 0; k < nt; k += BT, t += 1) {

        ptr_t a0 = bufferx[t % 2];
        ptr_t a1 = bufferx[(t + 1) % 2];

        ptr_t b0 = buffery[t % 2];
        ptr_t b1 = buffery[(t + 1) % 2];

        ptr_t c0 = bufferz[t % 2];
        ptr_t c1 = bufferz[(t + 1) % 2];
        
        exchange_data(a0, b0, c0, z_start, z_end, ldx, ldy, BT + 1, grid_info);
        
#pragma omp parallel for collapse(2) schedule(static)
        for (int z = z_start; z < z_end; z += BZ) {
            for (int y = y_start; y < y_end; y += BY) {
                for (int x = x_start; x < x_end; x += BX) {
                    int z_off = z - z_start, y_off = y - y_start, x_off = x - x_start;
                    int z_begin = max(z_off - BT, 0), z_stop = min(z + BZ + BT, z_end) - z_start;
                    int y_begin = max(y_off - BT, 0), y_stop = min(y + BY + BT, y_end) - y_start;
                    int x_begin = max(x_off - BT, 0), x_stop = min(x + BX + BT, x_end) - x_start;
                    // check whether halo contains data from other ranks
                    if (has_down && z_begin == 0) {
                        z_begin = -BT; // use halo data
                    }
                    if (has_up && z_stop == z_end - z_start) {
                        z_stop += BT; // use halo data
                    }
                    int buf_z_start = BT - (z_off - z_begin), buf_z_end = BUF_DIM_Z - (z_off + BZ + BT - z_stop);
                    int buf_y_start = BT - (y_off - y_begin), buf_y_end = BUF_DIM_Y - (y_off + BY + BT - y_stop);
                    int buf_x_start = BT - (x_off - x_begin), buf_x_end = BUF_DIM_X - (x_off + BX + BT - x_stop);
                    // allocate contiguous buffer
                    data_t a_buf_0[BUF_SIZE] = {}, b_buf_0[BUF_SIZE] = {}, c_buf_0[BUF_SIZE] = {};
                    data_t a_buf_1[BUF_SIZE] = {}, b_buf_1[BUF_SIZE] = {}, c_buf_1[BUF_SIZE] = {};
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
                            memcpy(a_buf_0 + buf_off, a0 + src_off, copy_size);
                            memcpy(b_buf_0 + buf_off, b0 + src_off, copy_size);
                            memcpy(c_buf_0 + buf_off, c0 + src_off, copy_size);
                        }
                    }
                    // run stencil kernel on block buffer for BT rounds
                    // in each round, the dimension shrinks by 1
                    // a_buf_0 -> a_buf_1 -> a_buf_0 -> ...
                    ptr_t A0 = a_buf_0, A1 = a_buf_1, B0 = b_buf_0, B1 = b_buf_1, C0 = c_buf_0, C1 = c_buf_1;
                    if (likely(t < t_fused)) {
#pragma unroll(BT)
                        for (int t = BT - 1; t >= 0; --t) {
#define STENCIL_FUSED_LOOP \
                            int xx_start = max(BT - t, buf_x_start), yy_start = max(BT - t, buf_y_start), zz_start = max(BT - t, buf_z_start); \
                            int xx_end = min(BT + BX + t, buf_x_end), yy_end = min(BT + BY + t, buf_y_end), zz_end = min(BT + BZ + t, buf_z_end); \
                            for (int zz = zz_start; zz < zz_end; ++zz) { \
                                for (int yy = yy_start; yy < yy_end; ++yy) { \
                                    stencil_inner_loop(A0, A1, B0, B1, C0, C1, xx_start, xx_end, yy, zz, BUF_DIM_X, BUF_DIM_Y, BUF_DIM_Z); \
                                } \
                            } \
                            std::swap(A0, A1); \
                            std::swap(B0, B1); \
                            std::swap(C0, C1);
                            STENCIL_FUSED_LOOP
                        }
                    } else {
                        // remaining steps
                        for (int t = (nt - k) - 1; t >= 0; --t) {
                            STENCIL_FUSED_LOOP
#undef STENCIL_FUSED_LOOP
                        }
                    }
                    // copy back from buffer to a1, b1, c1
                    copy_from_buffer(a1, b1, c1, A0, B0, C0, x, y, z, x_start, y_start, z_start, ldx, ldy, ldz);
                    ret = a1;
                }
            }
        }
    }

    return ret;
}


// benchmark entrypoint
ptr_t stencil_7(ptr_t A0, ptr_t A1, ptr_t B0, ptr_t B1, ptr_t C0, ptr_t C1, const dist_grid_info_t *grid_info, int nt) {
  
    ptr_t bufferx[2] = {A0, A1};
    ptr_t buffery[2] = {B0, B1};
    ptr_t bufferz[2] = {C0, C1};

    // calculate local size
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;
    bool has_up = grid_info->p_id < grid_info->p_num - 1, has_down = grid_info->p_id > 0;

    int TT = get_t_block_size(grid_info->global_size_x);

    if (grid_info->global_size_x < TRIVIAL_METHOD_THRESHOLD_MPI) {
        return stencil_time_skew<true>(
            x_start, x_end, y_start, y_end, z_start, z_end, nt, ldx, ldy, ldz,
            bufferx, buffery, bufferz, grid_info->global_size_x, has_up, has_down,
            [=](auto a, auto b, auto c)
                __attribute__((always_inline)) {
                    exchange_data(a, b, c, z_start, z_end, ldx, ldy, TT, grid_info);
                });
    } else {
        return stencil_trivial(x_start, x_end, y_start, y_end, z_start, z_end, nt, ldx, ldy, ldz, bufferx, buffery, bufferz, has_down, has_up, grid_info);
    }

}
