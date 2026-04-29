// !nvcc coalescence_and_alignment/main_v2.cu -o exercise && ./exercise
// %%writefile coalescence_and_alignment/main_v2.cu

// [OBJ] matrix_mul_optimization

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

// --- 1. KERNEL NAIVE (NON COALESCENTE) ---
__global__ void MatMulGPU_naive(float *A, float *B, float *C, int M, int N, int K) {
    int indexRow = threadIdx.x + blockIdx.x * blockDim.x;
    int indexCol = threadIdx.y + blockIdx.y * blockDim.y;

    if (indexRow < M && indexCol < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[indexRow * N + i] * B[i * K + indexCol];
        }
        C[indexRow * K + indexCol] = sum;
    }
}

// --- 2. KERNEL COALESCENTE (OTTIMIZZATO) ---
__global__ void MatMulGPU_coal(float *A, float *B, float *C, int M, int N, int K) {
    int indexCol = threadIdx.x + blockIdx.x * blockDim.x;
    int indexRow = threadIdx.y + blockIdx.y * blockDim.y;

    if (indexRow < M && indexCol < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[indexRow * N + i] * B[i * K + indexCol];
        }
        C[indexRow * K + indexCol] = sum;
    }
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    float *h_A = (float*) malloc(size_A);
    float *h_B = (float*) malloc(size_B);
    for (int i = 0; i < M * N; i++) h_A[i] = 1.0f;
    for (int i = 0; i < N * K; i++) h_B[i] = 0.5f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 tpb(BLOCK_SIZE, BLOCK_SIZE);

    // Configurazione Naive
    dim3 gridNaive((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Configurazione Coalescente
    dim3 gridCoal((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Test Naive
    cudaEventRecord(start);
    MatMulGPU_naive<<<gridNaive, tpb>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);

    // Test Coalescente
    cudaEventRecord(start);
    MatMulGPU_coal<<<gridCoal, tpb>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_coal;
    cudaEventElapsedTime(&ms_coal, start, stop);

    printf("Risultati Performance (Matrice %dx%dx%d):\n", M, N, K);
    printf("- Tempo Kernel Naive:       %f ms\n", ms_naive);
    printf("- Tempo Kernel Coalescente: %f ms\n", ms_coal);
    printf("- Speedup:                  %.2fx\n", ms_naive / ms_coal);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);

    return 0;
}