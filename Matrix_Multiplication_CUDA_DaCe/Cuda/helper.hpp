#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/driver_types.h>
#include <assert.h>
#include <iostream>
#include <random>

const std::string error = "\033[1;31m[ERROR]\033[0m ";
const std::string success = "\033[1;32m[SUCCESS]\033[0m ";
const std::string info = "\033[1;33m[INFO]\033[0m ";

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// prints all the values of a matrix
template <typename T> void printMatrix(const T *A, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j)
            std::cout << A[i * rows + j] << " ";
        std::cout << "\n";
    }
}

// verifies the GPU result on the CPU
template <typename T> bool verify(const T *A, const T *B, const T *C, const int N) {
    const double tol = 0.001;
    bool correct = true;
    for (int i = 0; correct && i < N; ++i) {
        for (int j = 0; correct && j < N; ++j) {
            double tmp = 0;
            std::cout << info << "--- Calculating C[" << i << "][" << j << "]\n";
            // std::cout << info << "--- Real result: " << C[i * N + j] << "\n";
            for (int k = 0; k < N; ++k) {
                // std::cout << info << "tmp + A[" << i << "][" << k << "] * B[" << k << "][" << j << "]\n";
                // std::cout << info << tmp << " + " << A[i * N + k] << " * " << B[k * N + j] << "\n";
                tmp += A[i * N + k] * B[k * N + j];
            }
            double err = abs(C[i * N + j] - tmp);
            if (err >= tol) {
                std::cout << error << "Difference between CPU and GPU is too large: " << err << "\n";
                std::cout << error << tmp << " != " << C[i * N + j] << "\n";
                correct = false;
            }
        }
    }

    return correct;
}

// fills a matrix with random floating values
template <typename T> void fillMatrix(T *A, int rows, int columns) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 10);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j)
            A[i * rows + j] = dist(e2);
    }
}

