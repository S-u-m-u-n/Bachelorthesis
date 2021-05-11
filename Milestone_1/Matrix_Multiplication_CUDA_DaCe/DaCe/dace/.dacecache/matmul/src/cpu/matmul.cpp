/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct matmul_t {
    dace::cuda::Context *gpu_context;
};

DACE_EXPORTED void __dace_runkernel_gemm_init_map_1_0_0(matmul_t *__state, double * __restrict__ _c, int M, int N);
DACE_EXPORTED void __dace_runkernel_gemm_map_1_1_8(matmul_t *__state, const double * __restrict__ _a, const double * __restrict__ _b, double * __restrict__ _c, int K, int M, int N);
inline void _MatMult_gemm_sdfg_0_0_5(matmul_t *__state, double * _a, double * _b, double * _c, int K, int M, int N) {

    {


        __dace_runkernel_gemm_init_map_1_0_0(__state, _c, M, N);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);



    }
    {


        __dace_runkernel_gemm_map_1_1_8(__state, _a, _b, _c, K, M, N);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);



    }
    
}

void __program_matmul_internal(matmul_t *__state, double * __restrict__ A, double * __restrict__ B, double * __restrict__ __return, int K, int M, int N)
{

    {
        double * gpu_A;
        cudaMalloc(&gpu_A, (K * M) * sizeof(double));
        double * gpu_B;
        cudaMalloc(&gpu_B, (K * N) * sizeof(double));
        double * gpu___return;
        cudaMalloc(&gpu___return, (M * N) * sizeof(double));


        cudaMemcpyAsync(gpu_A, A, (K * M) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[0]);
        cudaMemcpyAsync(gpu_B, B, (K * N) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[1]);

        cudaEventRecord(__state->gpu_context->events[1], __state->gpu_context->streams[1]);
        cudaStreamWaitEvent(__state->gpu_context->streams[0], __state->gpu_context->events[1], 0);

        _MatMult_gemm_sdfg_0_0_5(__state, &gpu_A[0], &gpu_B[0], &gpu___return[0], K, M, N);
        cudaMemcpyAsync(__return, gpu___return, (M * N) * sizeof(double), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);


        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu___return);

    }
}

DACE_EXPORTED void __program_matmul(matmul_t *__state, double * __restrict__ A, double * __restrict__ B, double * __restrict__ __return, int K, int M, int N)
{
    __program_matmul_internal(__state, A, B, __return, K, M, N);
}
DACE_EXPORTED int __dace_init_cuda(matmul_t *__state, int K, int M, int N);
DACE_EXPORTED int __dace_exit_cuda(matmul_t *__state);

DACE_EXPORTED matmul_t *__dace_init_matmul(int K, int M, int N)
{
    int __result = 0;
    matmul_t *__state = new matmul_t;


    __result |= __dace_init_cuda(__state, K, M, N);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_matmul(matmul_t *__state)
{
    __dace_exit_cuda(__state);
    delete __state;
}

