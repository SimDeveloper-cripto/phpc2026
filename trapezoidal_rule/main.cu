#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "kernels.cuh"

#define GET_TIME(t) { \
   t = std::chrono::high_resolution_clock::now(); \
}

void run_integration(int method, float a, float b, int n, int block_size) {
    float *trap_p;
    cudaMallocManaged(&trap_p, sizeof(float));
    
    float h = (b - a) / n;
    int num_blocks = (n + block_size - 1) / block_size;
    
    *trap_p = 0.5f * (f(a) + f(b));

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    
    GET_TIME(start);
    
    size_t shared_mem_size = block_size * sizeof(float);
    
    if (method == 1) {
        trap_shared_tree<<<num_blocks, block_size, shared_mem_size>>>(a, h, n, trap_p);
    } else if (method == 2) {
        trap_shared_dissemination<<<num_blocks, block_size, shared_mem_size>>>(a, h, n, trap_p);
    } else if (method == 3) {
        size_t warp_shared_size = (block_size / 32) * sizeof(float);
        trap_warp_shuffle_k<<<num_blocks, block_size, warp_shared_size>>>(a, h, n, trap_p);
    }
    
    cudaDeviceSynchronize();
    
    GET_TIME(end);
    std::chrono::duration<double> diff = end - start;
    double time = diff.count();
    
    *trap_p = (*trap_p) * h;
    
    printf("Metodo %d | Risultato: %.3f | Tempo: %f sec\n", method, *trap_p, time);
    
    cudaFree(trap_p);
}

int main() {
    float a = 0.0f;
    float b = 10.0f;

    int n          = 10000000; 
    int block_size = 256; 

    printf("Integrazione di f(x)=x^2 da %f a %f con %d trapezi.\n", a, b, n);
    printf("Block size: %d threads\n\n", block_size);

    run_integration(1, a, b, n, block_size);
    run_integration(2, a, b, n, block_size);
    run_integration(3, a, b, n, block_size);

    return 0;
}