#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_BLKSZ 1024
#define WARPSZ 32

__device__ __host__ float f(float x) { return x * x; }

__device__ float Shared_mem_sum(float* shared_vals, int my_lane) {
    for (int offset = WARPSZ / 2; offset > 0; offset /= 2) {
        if (my_lane < offset) {
            shared_vals[my_lane] += shared_vals[my_lane + offset];
        }
        __syncwarp();
    }
    return shared_vals[0];
}

__global__ void trapezoidal_kernel(float a, float b, int n, float h, float* block_sums) {
    __shared__ float thread_calcs[MAX_BLKSZ]; 
    __shared__ float warp_sum_arr[WARPSZ];    

    int tid     = threadIdx.x + blockIdx.x * blockDim.x;
    int w       = threadIdx.x / WARPSZ;
    int my_lane = threadIdx.x % WARPSZ;

    // 1) Ogni thread calcola il suo contributo
    float my_x       = a + tid * h;
    float my_contrib = 0.0f;

    if (tid < n) {
        my_contrib = f(my_x);

        // Applica i pesi della regola del trapezio (metÃ  per gli estremi)
        if (tid == 0 || tid == n - 1) {
            my_contrib /= 2.0f;
        }
        my_contrib *= h;
    }

    // I thread del warp salvano i calcoli in un sotto-array dedicato
    float* shared_vals = thread_calcs + w * WARPSZ; 
    shared_vals[my_lane] = my_contrib; 

    __syncwarp();

    // Ogni warp somma i suoi contributi parziali
    float my_result = Shared_mem_sum(shared_vals, my_lane);

    // Il Lane 0 salva la somma del warp nell'array condiviso contiguo
    if (my_lane == 0) {
        warp_sum_arr[w] = my_result; 
    }

    __syncthreads(); 

    // Il Warp 0 somma i risultati di tutti i warp
    if (w == 0) {
        int num_warps = blockDim.x / WARPSZ;

        // Inizializza a 0 i pad finali se ci sono meno di 32 warp
        if (my_lane >= num_warps && my_lane < WARPSZ) {
            warp_sum_arr[my_lane] = 0.0f;
        }
        __syncwarp();

        // Somma finale all'interno del Warp 0
        float final_block_sum = Shared_mem_sum(warp_sum_arr, my_lane);

        // Il thread 0 scrive il risultato del blocco in memoria globale
        if (my_lane == 0) {
            block_sums[blockIdx.x] = final_block_sum;
        }
    }
}

float trapezoidal_cpu(float a, float b, int n, float h) {
    float sum = (f(a) + f(b)) / 2.0f;

    for (int i = 1; i < n - 1; ++i) {
        float x = a + i * h;
        sum += f(x);
    }
    return sum * h;
}

int main() {
    float a = 0.0f;
    float b = 10.0f;
    int n   = 1024 * 1024; // 1 mln intervals
    float h = (b - a) / (float)n;

    // Impostazione di k > 1 (es. k = 8 implica blocchi da 256 threads)
    int k               = 8;
    int threadsPerBlock = k * WARPSZ; 
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

    float *d_block_sums, *h_block_sums;

    h_block_sums = (float*)malloc(blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&d_block_sums, blocksPerGrid * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    trapezoidal_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, n, h, d_block_sums);
    cudaMemcpy(h_block_sums, d_block_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    
    float gpu_result = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        gpu_result += h_block_sums[i];
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    clock_t cpu_start = clock();
    float cpu_result  = trapezoidal_cpu(a, b, n, h);
    clock_t cpu_stop  = clock();
    float cpu_time    = 1000.0f * (float)(cpu_stop - cpu_start) / CLOCKS_PER_SEC;

    float speedup = cpu_time / gpu_time;

    printf("--- Risultati dell'Integrazione ---\n");
    printf("Risultato CPU: %f (Tempo: %f ms)\n", cpu_result, cpu_time);
    printf("Risultato GPU: %f (Tempo: %f ms)\n", gpu_result, gpu_time);
    printf("Speed-up ottenuto: %.2fx\n", speedup);

    cudaFree(d_block_sums);
    free(h_block_sums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}