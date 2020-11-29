#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <numa.h>
#include <omp.h>

#include "stencil-common.hh"

extern "C" const char* version_name = "Optimized version (MPI + OpenMP)";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    
    // split data array
    int ranks = grid_info->p_num;
    int slice_z = 1 + (ranks & 2 != 0), slice_y = 1 + (ranks & 4 != 0), slice_x = 1 + (ranks & 8 != 0);

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

    if (grid_info->p_id == 0) {
        REQUIRE(ranks == 1 || ranks == 2 || ranks == 4 || ranks == 8, "MPI rank number not allowed");
        REQUIRE(grid_info->global_size_x % slice_x == 0 && grid_info->global_size_y % slice_y == 0 && grid_info->global_size_z % slice_z == 0, "Dimensions must divide number of slices");
        if (local_size != nodes) {
            fprintf(stderr, "[WARNING] rank num %d on each node is not optimal, use %d for best performance.\n", local_size, nodes);
        }
        if (threads > cpus) {
            fprintf(stderr, "[WARNING] thread %d for each rank is not optimal, use %d for best performance.\n", threads, cpus);
        }
    }

    // bind process to NUMA node (OpenMP threads binds automatically)
    if (grid_info->p_num >= 2) {
        auto cpu_mask = numa_allocate_nodemask();
        numa_bitmask_setbit(cpu_mask, local_rank % nodes);
        numa_bind(cpu_mask);
    }

    grid_info->slice_x = slice_x;
    grid_info->slice_y = slice_y;
    grid_info->slice_z = slice_z;
    grid_info->local_size_x = grid_info->global_size_x / slice_x;
    grid_info->local_size_y = grid_info->global_size_y / slice_y;
    grid_info->local_size_z = grid_info->global_size_z / slice_z;

    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;

    grid_info->halo_size_x = BT + 1;
    grid_info->halo_size_y = BT + 1;
    grid_info->halo_size_z = BT + 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

ptr_t stencil_7(ptr_t A0, ptr_t A1,ptr_t B0, ptr_t B1,ptr_t C0, ptr_t C1, const dist_grid_info_t *info, int nt) {
    fprintf(stderr, "unimplemented\n");
    exit(1);
    return A0;
}

