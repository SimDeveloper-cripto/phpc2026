// !nvcc coalescence_and_alignment/main_v1.cu -o exercise && ./exercise
// %%writefile coalescence_and_alignment/main_v1.cu

#include <stdio.h>
#include <cuda_runtime.h>

// Controllo errori
void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// 1. Kernel Prodotto Element-per-Element COALESCENTE (Row-wise)
__global__ void MatMatGPU_coalesced(int *A, int *B, int *C, int row, int col) {
    int indexRow = blockIdx.y * blockDim.y + threadIdx.y;
    int indexCol = blockIdx.x * blockDim.x + threadIdx.x;
    int index    = indexRow * col + indexCol;

    if (indexRow < row && indexCol < col) C[index] = A[index] * B[index];
}

// 2. Kernel Prodotto Element-per-Element STRIDED (Column-wise)
__global__ void MatMatGPU_strided(int *A, int *B, int *C, int row, int col) {
    // Invertiamo il mapping: threadIdx.x controlla le righe !!
    int indexRow = blockIdx.x * blockDim.x + threadIdx.x; 
    int indexCol = blockIdx.y * blockDim.y + threadIdx.y;
    int index    = indexRow * col + indexCol; 

    if (indexRow < row && indexCol < col) C[index] = A[index] * B[index];
}

// 3. Kernel con PITCH
__global__ void MatMatGPU_pitch(int* A, int* B, int* C, int pitchA, int pitchB, int pitchC, int row, int col) {
    int indexCol = blockIdx.x * blockDim.x + threadIdx.x; 
    int indexRow = blockIdx.y * blockDim.y + threadIdx.y; 

    // L'indice usa il pitch per saltare tra le righe
    int idxA = indexRow * pitchA + indexCol;
    int idxB = indexRow * pitchB + indexCol;
    int idxC = indexRow * pitchC + indexCol;

    if (indexRow < row && indexCol < col) C[idxC] = A[idxA] * B[idxB];
}

int main() {
    int M = 8192; // Righe
    int N = 8192; // Colonne

    size_t size = M * N * sizeof(int);

    // Allocazione su Host
    int *h_A = (int*) malloc(size);
    int *h_B = (int*) malloc(size);
    int *h_C = (int*) malloc(size);

    for(int i = 0; i < M * N; i++) {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    // --- TEST 1: COALESCED ---
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Usiamo blocchi da 1024 thread
    dim3 threads(32, 32);
    // Numero di blocchi necessari per coprire tutta la matrice
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    MatMatGPU_coalesced<<<blocks, threads>>>(d_A, d_B, d_C, M, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_coalesced;
    cudaEventElapsedTime(&ms_coalesced, start, stop);
    printf("Tempo Coalesced (Row-major): %f ms\n", ms_coalesced);

    // --- TEST 2: STRIDED ---
    cudaEventRecord(start);
    MatMatGPU_strided<<<blocks, threads>>>(d_A, d_B, d_C, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_strided;
    cudaEventElapsedTime(&ms_strided, start, stop);
    printf("Tempo Strided (Col-major): %f ms\n", ms_strided);

    // --- TEST 3: WITH PITCH
    int *p_A, *p_B, *p_C;
    size_t pitchA, pitchB, pitchC;

    cudaMallocPitch((void**)&p_A, &pitchA, N * sizeof(int), M);
    cudaMallocPitch((void**)&p_B, &pitchB, N * sizeof(int), M);
    cudaMallocPitch((void**)&p_C, &pitchC, N * sizeof(int), M);

    cudaMemcpy2D(p_A, pitchA, h_A, N * sizeof(int), N * sizeof(int), M, cudaMemcpyHostToDevice);
    cudaMemcpy2D(p_B, pitchB, h_B, N * sizeof(int), N * sizeof(int), M, cudaMemcpyHostToDevice);

    cudaEventRecord(start);

    // pitch / sizeof(int) --> per avere il pitch in numero di elementi
    MatMatGPU_pitch<<<blocks, threads>>>(p_A, p_B, p_C, pitchA / sizeof(int), pitchB / sizeof(int), pitchC / sizeof(int), M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_pitch;
    cudaEventElapsedTime(&ms_pitch, start, stop);
    printf("Tempo con Pitch e Allineamento: %f ms\n", ms_pitch);

    // Pulizia
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(p_A); cudaFree(p_B); cudaFree(p_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}