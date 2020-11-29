#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "stencil-common.hh"

extern "C" const char* version_name = "Optimized version (MPI + OpenMP)";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    
    // split data array
    int ranks = grid_info->p_num;
    int slice_z = 1 + (ranks & 2 != 0), slice_y = 1 + (ranks & 4 != 0), slice_x = 1 + (ranks & 8 != 0);

    // do some checking before start
    if (grid_info->p_id == 0) {
        REQUIRE(ranks == 1 || ranks == 2 || ranks == 4 || ranks == 8, "MPI rank number not allowed");
        REQUIRE(grid_info->global_size_x % slice_x == 0 && grid_info->global_size_y % slice_y == 0 && grid_info->global_size_z % slice_z == 0, "Dimensions must divide number of slices");
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

    grid_info->halo_size_x = MPI_HALO_WIDTH;
    grid_info->halo_size_y = MPI_HALO_WIDTH;
    grid_info->halo_size_z = MPI_HALO_WIDTH;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

ptr_t stencil_7(ptr_t A0, ptr_t A1,ptr_t B0, ptr_t B1,ptr_t C0, ptr_t C1, const dist_grid_info_t *info, int nt) {
    fprintf(stderr, "unimplemented\n");
    exit(1);
    return A0;
}

