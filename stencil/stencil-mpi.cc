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
    int halo_size_z = TT + 1;

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

    // size of local data
    int local_sizes[3] = {local_size_x + 2 * halo_size_x, local_size_y + 2 * halo_size_y, local_size_z + 2 * halo_size_z};
    int sub_sizes[3] = {local_size_x, local_size_y, halo_size_z};

    auto neighbours = new stencil_neighbour_t[NEIGHBOUR_NUM]();
    grid_info->additional_info = neighbours;

    // calculate neighbour information along z axis
    if (rank < ranks - 1) {
        neighbours[0].rank = rank + 1;
        int starts[3] = {halo_size_x, halo_size_y, local_size_z + halo_size_z};
        int comm_type;
        // data to receive (from remote data to local halo)
        MPI_Type_create_subarray(3, local_sizes, sub_sizes, starts, MPI_ORDER_FORTRAN, MPI_DOUBLE, &comm_type);
        MPI_Type_commit(&comm_type);
        neighbours[0].recv_type = comm_type;
        // data to receive (from local data to remote halo)
        starts[2] = local_size_z;
        MPI_Type_create_subarray(3, local_sizes, sub_sizes, starts, MPI_ORDER_FORTRAN, MPI_DOUBLE, &comm_type);
        MPI_Type_commit(&comm_type);
        neighbours[0].send_type = comm_type;
    }
    if (rank > 0) {
        neighbours[1].rank = rank - 1;
        int starts[3] = {halo_size_x, halo_size_y, 0};
        int comm_type;
        // data to receive (from remote data to local halo)
        MPI_Type_create_subarray(3, local_sizes, sub_sizes, starts, MPI_ORDER_FORTRAN, MPI_DOUBLE, &comm_type);
        MPI_Type_commit(&comm_type);
        neighbours[1].recv_type = comm_type;
        // data to receive (from local data to remote halo)
        starts[2] = halo_size_z;
        MPI_Type_create_subarray(3, local_sizes, sub_sizes, starts, MPI_ORDER_FORTRAN, MPI_DOUBLE, &comm_type);
        MPI_Type_commit(&comm_type);
        neighbours[1].send_type = comm_type;
    }

    load_stencil_coefficients();
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

void inline __attribute__((always_inline)) exchange_data(ptr_t a0, ptr_t b0, ptr_t c0, stencil_neighbour_t *neighbours) {
        // exchange data between neighbours to current buffer
        MPI_Request requests[NEIGHBOUR_NUM][3 * 2];
#pragma unroll(NEIGHBOUR_NUM)
        for (int i = 0; i < NEIGHBOUR_NUM; ++i) {
            if (neighbours[i].rank != -1) {
                MPI_Isend(a0, 1, neighbours[i].send_type, neighbours[i].rank, 0, MPI_COMM_WORLD, &requests[i][0]);
                MPI_Isend(b0, 1, neighbours[i].send_type, neighbours[i].rank, 0, MPI_COMM_WORLD, &requests[i][1]);
                MPI_Isend(c0, 1, neighbours[i].send_type, neighbours[i].rank, 0, MPI_COMM_WORLD, &requests[i][2]);
                MPI_Irecv(a0, 1, neighbours[i].recv_type, neighbours[i].rank, 0, MPI_COMM_WORLD, &requests[i][3]);
                MPI_Irecv(b0, 1, neighbours[i].recv_type, neighbours[i].rank, 0, MPI_COMM_WORLD, &requests[i][4]);
                MPI_Irecv(c0, 1, neighbours[i].recv_type, neighbours[i].rank, 0, MPI_COMM_WORLD, &requests[i][5]);
            }
        }

        // wait for all communication to finish
#pragma unroll(NEIGHBOUR_NUM)
        for (int i = 0; i < NEIGHBOUR_NUM; ++i) {
            if (neighbours[i].rank != -1) {
                MPI_Waitall(6, requests[i], MPI_STATUSES_IGNORE);
            }
        }
}


ptr_t inline __attribute__((always_inline)) stencil_trivial(
    int x_start, int x_end, int y_start, int y_end, int z_start, int z_end, 
    int nt, int ldx, int ldy, int ldz,
    ptr_t bufferx[2], ptr_t buffery[2], ptr_t bufferz[2],
    bool has_down, bool has_up, stencil_neighbour_t *neighbours
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
        
        exchange_data(a0, b0, c0, neighbours);
        
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
  
    auto neighbours = (stencil_neighbour_t *) grid_info->additional_info;

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

    if (grid_info->global_size_x < 768) {
        if (has_up) z_end += TT;
        if (has_down) z_start -= TT;
        return stencil_time_skew<false>(
            x_start, x_end, y_start, y_end, z_start, z_end, nt, ldx, ldy, ldz, bufferx, buffery, bufferz,
            [neighbours] (auto a, auto b, auto c) __attribute__((always_inline)) {
                exchange_data(a, b, c, neighbours);
            }
        );
    } else {
        return stencil_trivial(x_start, x_end, y_start, y_end, z_start, z_end, nt, ldx, ldy, ldz, bufferx, buffery, bufferz, has_down, has_up, neighbours);
    }

}
