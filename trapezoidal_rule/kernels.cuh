#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__host__ __device__ float f(float x);

__global__ void trap_shared_tree         (float a, float h, int n, float* total_sum);
__global__ void trap_shared_dissemination(float a, float h, int n, float* total_sum);
__global__ void trap_warp_shuffle_k      (float a, float h, int n, float* total_sum);