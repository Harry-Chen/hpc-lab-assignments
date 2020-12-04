#include <float.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "common.h"
#include "utils.h"

#define EPS_TOL 1e-10

#define CHECK_AND_SET(cond, state) if(cond) {state;}
#define CHECK_AND_BREAK(cond, state) if(cond) {state;break;}

typedef FILE *file_t;

int clean(int ret, void *p) {
    free(p);
    return ret;
}
                
int clean_file(int ret, file_t file) {
    fclose(file);
    return ret;
}

void gpu_free(void *p) {
    cudaFree(p);
}

int read_matrix_default(dist_matrix_t *mat, const char* filename) {
    file_t file;
    int global_m, global_nnz;
    int ret, count;
    index_t *r_pos;
    index_t *c_idx;
    data_t *values;
    index_t *gpu_r_pos;
    index_t *gpu_c_idx;
    data_t *gpu_values;

    file = fopen(filename, "rb");
    CHECK(file == NULL, IO_ERR)

    count = fread(&global_m, sizeof(index_t), 1, file);
    CHECK(count != 1, IO_ERR)

    r_pos = (index_t*)malloc(sizeof(index_t) * (global_m + 1));
    CHECK(r_pos == NULL, NO_MEM)

    count = fread(r_pos, sizeof(index_t), global_m + 1, file);
    CHECK(count != global_m + 1, IO_ERR)
    global_nnz = r_pos[global_m];

    c_idx = (index_t*)malloc(sizeof(index_t) * global_nnz);
    CHECK(c_idx == NULL, NO_MEM)
    values = (data_t*)malloc(sizeof(data_t) * global_nnz);
    CHECK(values == NULL, NO_MEM)

    count = fread(c_idx, sizeof(index_t), global_nnz, file);
    CHECK(count != global_nnz, IO_ERR)
    count = fread(values, sizeof(data_t), global_nnz, file);
    CHECK(count != global_nnz, IO_ERR)

    fclose(file);

    ret = cudaMalloc(&gpu_r_pos, sizeof(index_t) * (global_m + 1));
    CHECK_ERROR(ret, ret)
    ret = cudaMalloc(&gpu_c_idx, sizeof(index_t) * global_nnz);
    CHECK_ERROR(ret, ret)
    ret = cudaMalloc(&gpu_values, sizeof(data_t) * global_nnz);
    CHECK_ERROR(ret, ret)
    
    cudaMemcpy(gpu_r_pos, r_pos, sizeof(index_t) * (global_m + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c_idx, c_idx, sizeof(index_t) * global_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_values, values, sizeof(data_t)  * global_nnz, cudaMemcpyHostToDevice);

    mat->perm = NULL;
    mat->global_m = global_m;
    mat->global_nnz = global_nnz;
    mat->r_pos = r_pos;
    mat->c_idx = c_idx;
    mat->values = values;
    mat->gpu_r_pos = gpu_r_pos;
    mat->gpu_c_idx = gpu_c_idx;
    mat->gpu_values = gpu_values;
    mat->additional_info = NULL;
    mat->CPU_free = free;
    mat->GPU_free = gpu_free;
    return SUCCESS;
}

char *cancat_name(const char *a, const char *b) {
    int l1 = strlen(a) - 4, l2 = strlen(b);
    char *c = (char*)malloc(sizeof(char) * (l1 + l2 + 1));
    if(c != NULL) {
        memcpy(c, a, l1 * sizeof(char));
        memcpy(c + l1, b, l2 * sizeof(char));
        c[l1 + l2] = '\0';
    }
    return c;
}

static void apply_perm(int n, const int *perm, const data_t *x, data_t *Px) {
    for(int i = 0; i < n; ++i) {
        Px[i] = x[perm[i]];
    }
}

int read_vector(dist_matrix_t *mat, const char* filename, const char* suffix, \
    int n, data_t* x, data_t *buffer) {
    char *new_name;
    file_t file;
    int count, i, m = mat->global_m;
    new_name = cancat_name(filename, suffix);
    CHECK(new_name == NULL, NO_MEM)

    file = fopen(new_name, "rb");
    CHECK(new_name == NULL, clean(IO_ERR, new_name))

    for(i = 0; i < n; ++i) {
        if(mat->perm == NULL) {
            count = fread(x + m * i, sizeof(data_t), m, file);
        } else {
            count = fread(buffer, sizeof(data_t), m, file);
            apply_perm(m, mat->perm, buffer, x + m * i);
        }
        CHECK(count != m, clean(clean_file(IO_ERR, file), new_name))
    }
    return clean(clean_file(SUCCESS, file), new_name);
}

int check_answer(dist_matrix_t *mat, const char* filename, data_t* y) {
    int ret, i, mi, m = mat->global_m;
    data_t maxerr = 0;
    data_t *ref = (data_t*) malloc(sizeof(data_t) * m);
    
    CHECK(ref == NULL, NO_MEM)
    ret = read_vector(mat, filename, "_x.vec", 1, ref, y + m);
    CHECK_ERROR(ret, clean(ret, ref))

    for(i = 0; i < m; ++i) {
        data_t err = abs(y[i] - ref[i]);
        if(err > maxerr) {
            maxerr = err;
            mi = i;
        }
    }
    if(maxerr > EPS_TOL) {
        fprintf(stderr, "x[%d] is %e. It should be %e. Error = %e\n", \
                mi, y[mi], ref[mi], y[mi] - ref[mi]);
        return clean(5, ref);
    } else {
        printf("    Error = %e\n", maxerr);
    }
    return clean(0, ref);
}


void destroy_dist_matrix(dist_matrix_t *mat) {
    if(mat->additional_info != NULL){
        destroy_additional_info(mat->additional_info);
        mat->additional_info = NULL;
    }
    if(mat->CPU_free != NULL) {
        if(mat->r_pos != NULL){
            mat->CPU_free(mat->r_pos);
            mat->r_pos = NULL;
        }
        if(mat->c_idx != NULL){
            mat->CPU_free(mat->c_idx);
            mat->c_idx = NULL;
        }
        if(mat->values != NULL){
            mat->CPU_free(mat->values);
            mat->values = NULL;
        }
        if(mat->perm != NULL){
            mat->CPU_free(mat->perm);
            mat->perm = NULL;
        }
    }
    if(mat->GPU_free != NULL) {
        if(mat->gpu_r_pos != NULL){
            mat->GPU_free(mat->gpu_r_pos);
            mat->gpu_r_pos = NULL;
        }
        if(mat->gpu_c_idx != NULL){
            mat->GPU_free(mat->gpu_c_idx);
            mat->gpu_c_idx = NULL;
        }
        if(mat->gpu_values != NULL){
            mat->GPU_free(mat->gpu_values);
            mat->gpu_values = NULL;
        }
    }
}

