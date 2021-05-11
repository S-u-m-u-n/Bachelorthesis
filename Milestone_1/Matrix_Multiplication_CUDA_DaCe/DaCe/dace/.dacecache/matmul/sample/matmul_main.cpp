#include <cstdlib>
#include "../include/matmul.h"

int main(int argc, char **argv) {
    matmulHandle_t handle;
    int K = 42;
    int M = 42;
    int N = 42;
    double * __restrict__ A = (double*) calloc((K * M), sizeof(double));
    double * __restrict__ B = (double*) calloc((K * N), sizeof(double));
    double * __restrict__ __return = (double*) calloc((M * N), sizeof(double));


    handle = __dace_init_matmul(K, M, N);
    __program_matmul(handle, A, B, __return, K, M, N);
    __dace_exit_matmul(handle);

    free(A);
    free(B);
    free(__return);


    return 0;
}
