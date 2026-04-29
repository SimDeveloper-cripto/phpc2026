#include "kernels.cuh"

__host__ __device__ float f(float x) { return x * x; }

// 1: Shared Memory (Tree-structured sum)
__global__ void trap_shared_tree(float a, float h, int n, float* total_sum) {
    extern __shared__ float sdata[];

    int tid       = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    float my_val  = 0.0f;

    if (global_id > 0 && global_id < n) {
        my_val = f(a + global_id * h);
    }

    sdata[tid] = my_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads(); 
    }

    if (tid == 0) {
        atomicAdd(total_sum, sdata[0]);
    }
}

// 2: Shared Memory (Dissemination sum)
__global__ void trap_shared_dissemination(float a, float h, int n, float* total_sum) {
    extern __shared__ float sdata[];

    int tid       = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    float my_val  = 0.0f;

    if (global_id > 0 && global_id < n) {
        my_val = f(a + global_id * h);
    }

    sdata[tid] = my_val;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float partner_val = sdata[(tid + offset) % blockDim.x];
        __syncthreads(); 

        sdata[tid] += partner_val;
        __syncthreads(); 
    }

    if (tid == 0) {
        atomicAdd(total_sum, sdata[0]); 
    }
}

// 3: Warp Shuffle (Per blocchi con k * warpSize thread)
__global__ void trap_warp_shuffle_k(float a, float h, int n, float* total_sum) {
    extern __shared__ float warp_sums[]; 

    int tid       = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    float my_val  = 0.0f;

    int lane_id = tid % warpSize;
    int warp_id = tid / warpSize;

    if (global_id > 0 && global_id < n) {
        my_val = f(a + global_id * h);
    }

    // FASE 1
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        my_val += __shfl_down_sync(0xffffffff, my_val, offset);
    }

    // FASE 2
    if (lane_id == 0) {
        warp_sums[warp_id] = my_val;
    }
    __syncthreads(); 

    // FASE 3
    if (warp_id == 0) {
        int num_warps   = blockDim.x / warpSize;
        float block_val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_val += __shfl_down_sync(0xffffffff, block_val, offset);
        }

        if (lane_id == 0) {
            atomicAdd(total_sum, block_val);
        }
    }
}