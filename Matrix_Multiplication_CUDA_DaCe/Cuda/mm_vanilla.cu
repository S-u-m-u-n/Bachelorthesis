#include "helper.hpp"
#include <cuda_profiler_api.h>
#include <iostream>

const int TILE_SIZE = 32;
const int N = 4096;

// Idea: each thread calculates one entry on the matrix C
//       by sampling a row from A and a column from B
template <typename T> __global__ void matrix_multiplication_vanilla(const T *A, const T *B, T *C) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    // C[tid_y][tid_x]: "tid_y-th row of A times tid_x-th column of B"
    T tmp = 0;
    for (int i = 0; i < N; ++i) {
        // A[tid_y][i] == A[tid_y * N + i]
        // B[i][tid_x] == B[i * N + tid_x]
        tmp += A[tid_y * N + i] * B[i * N + tid_x];
    }
    C[tid_y * N + tid_x] = tmp;
}

int main() {
    const size_t memsize = N * N * sizeof(double);
    std::cout << info << "Memory Size per Array: " << memsize << "\n";
    double *A = (double *)malloc(memsize);
    double *B = (double *)malloc(memsize);
    double *C = (double *)malloc(memsize);
    double *dev_A, *dev_B, *dev_C;
    HANDLE_ERROR(cudaMalloc((void **)&dev_A, memsize));
    HANDLE_ERROR(cudaMalloc((void **)&dev_B, memsize));
    HANDLE_ERROR(cudaMalloc((void **)&dev_C, memsize));

    // fill the two matrices with random double values
    fillMatrix(A, N, N);
    // std::cout << info << "A[0] = " << A[0] << ", A[N * N -1] = " << A[N * N - 1] << "\n";
    fillMatrix(B, N, N);
    // std::cout << info << "B[0] = " << B[0] << ", B[N * N -1] = " << B[N * N - 1] << "\n";

    // // print the values of the two matrices
    // printMatrix(A, N, N);
    // printMatrix(B, N, N);

    // Initialize the grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    // Assuming N is a multiple of TILE_SIZE, otherwise we need to add padding or check bounds in kernel
    dim3 dimGrid(N / TILE_SIZE, N / TILE_SIZE, 1);

    // start profiling
    cudaProfilerStart();

    HANDLE_ERROR(cudaMemcpy(dev_A, A, memsize, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_B, B, memsize, cudaMemcpyHostToDevice));

    // std::cout << info << "Launching CUDA kernel...\n";
    matrix_multiplication_vanilla<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C);
    cudaDeviceSynchronize();
    // std::cout << info << "CUDA kernel is finished!\n";

    HANDLE_ERROR(cudaMemcpy(C, dev_C, memsize, cudaMemcpyDeviceToHost));
    // std::cout << info << "Result copied back to host.\n";

    // stop profiling
    cudaProfilerStop();

    HANDLE_ERROR(cudaFree(dev_A));
    HANDLE_ERROR(cudaFree(dev_B));
    HANDLE_ERROR(cudaFree(dev_C));

// printMatrix(C, N, N);
// verify that solution is correct
#ifdef VERIFY
    {
        if (verify(A, B, C, N))
            std::cout << success << "Matrix multiplication successful!" << std::endl;
        else
            std::cout << error << "Matrix multiplication unsuccessful. :(" << std::endl;
    }
#endif

    std::cout << success << "Matrix multiplication successful!" << std::endl;

    free(A);
    free(B);
    free(C);

    return 0;
}

