#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED 1

#include <stdio.h>
#include <assert.h>

#include "common.h"

#define CHECK(err, err_code) if(err) { return err_code; }
#define CHECK_ERROR(ret, err_code) CHECK(ret != 0, err_code)
#define CHECK_NULL(ret, err_code) CHECK(ret == NULL, err_code)

#ifdef __cplusplus
extern "C" {
#endif

int fatal_error(int code);

int read_matrix_default(dist_matrix_t *mat, const char* filename);
void destroy_dist_matrix(dist_matrix_t *mat);

int read_vector(dist_matrix_t *mat, const char* filename, const char* suffix, int n, data_t* x);
int check_answer(dist_matrix_t *mat, const char* filename, data_t* y);

#ifdef __cplusplus
}
#endif

#define SUCCESS cudaSuccess
#define NO_MEM cudaErrorMemoryAllocation
#define IO_ERR 3

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = (condition); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d: %s\n", error, cudaGetErrorString(error)); \
        assert(error == cudaSuccess); \
    } \
  } while (0)

#endif

#ifdef __NVCC__

#if __CUDA_ARCH__ < 600
#define ATOMIC_ADD_DOUBLE atomicAddDouble
__device__ inline double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#else
#define ATOMIC_ADD_DOUBLE atomicAdd
#endif

template<typename G>
__device__ __inline__ uint get_peers(G key) {
    uint peers=0;
    bool is_peer;

    // in the beginning, all lanes are available
    uint unclaimed=0xffffffff;

    do {
        // fetch key of first unclaimed lane and compare with this key
        is_peer = (key == __shfl(key, __ffs(unclaimed) - 1));

        // determine which lanes had a match
        peers = __ballot(is_peer);

        // remove lanes with matching keys from the pool
        unclaimed ^= peers;

        // quit if we had a match
    } while (!is_peer);

    return peers;
}

template <typename F>
__device__ __inline__ F reduce_peers(uint peers, F &x) {
    int lane = threadIdx.x & 31;

    // find the peer with lowest lane index
    int first = __ffs(peers)-1;

    // calculate own relative position among peers
    int rel_pos = __popc(peers << (32 - lane));

    // ignore peers with lower (or same) lane index
    peers &= (0xfffffffe << lane);

    while(__any(peers)) {
        // find next-highest remaining peer
        int next = __ffs(peers);

        // __shfl() only works if both threads participate, so we always do.
        F t = __shfl(x, next - 1);

        // only add if there was anything to add
        if (next) x += t;

        // all lanes with their least significant index bit set are done
        uint done = rel_pos & 1;

        // remove all peers that are already done
        peers &= ~__ballot(done);

        // abuse relative position as iteration counter
        rel_pos >>= 1;
    }

    // distribute final result to all peers (optional)
    F res = __shfl(x, first);

    return res;
}

#endif