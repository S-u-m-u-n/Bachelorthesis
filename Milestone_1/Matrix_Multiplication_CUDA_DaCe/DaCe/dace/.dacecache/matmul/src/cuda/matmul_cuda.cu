
#include <cuda_runtime.h>
#include <dace/dace.h>


struct matmul_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(matmul_t *__state, int K, int M, int N);
DACE_EXPORTED void __dace_exit_cuda(matmul_t *__state);



int __dace_init_cuda(matmul_t *__state, int K, int M, int N) {
    int count;

    // Check that we are able to run cuda code
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        printf("ERROR: GPU drivers are not configured or cuda-capable device "
               "not found\n");
        return 1;
    }
    if (count == 0)
    {
        printf("ERROR: No cuda-capable devices found\n");
        return 2;
    }

    // Initialize cuda before we run the application
    float *dev_X;
    cudaMalloc((void **) &dev_X, 1);
    cudaFree(dev_X);

    __state->gpu_context = new dace::cuda::Context(2, 2);

    // Create cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 2; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(matmul_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 2; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void gemm_init_map_1_0_0(double * __restrict__ _c, int M, int N) {
    {
        {
            int _o1 = (blockIdx.x * 32 + threadIdx.x);
            int _o0 = (blockIdx.y * 1 + threadIdx.y);
            if (_o1 < N) {
                {
                    {
                        double out;

                        ///////////////////
                        // Tasklet code (gemm_init)
                        out = 0;
                        ///////////////////

                        _c[((N * _o0) + _o1)] = out;
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_gemm_init_map_1_0_0(matmul_t *__state, double * __restrict__ _c, int M, int N);
void __dace_runkernel_gemm_init_map_1_0_0(matmul_t *__state, double * __restrict__ _c, int M, int N)
{

    void  *gemm_init_map_1_0_0_args[] = { (void *)&_c, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)gemm_init_map_1_0_0, dim3(int_ceil(int_ceil(N, 1), 32), int_ceil(int_ceil(M, 1), 1), int_ceil(1, 1)), dim3(32, 1, 1), gemm_init_map_1_0_0_args, 0, __state->gpu_context->streams[0]);
}
__global__ void gemm_map_1_1_8(const double * __restrict__ _a, const double * __restrict__ _b, double * __restrict__ _c, int K, int M, int N) {
    {
        {
            int tile___i1 = (128 * blockIdx.x);
            int tile___i0 = (128 * blockIdx.y);
            {
                for (auto tile___i2 = 0; tile___i2 < K; tile___i2 += 8) {
                    __shared__ double trans__a[1024];
                    __shared__ double trans__b[1024];

                    dace::CopyND<double, 1, false, 128, 8>::template ConstDst<8, 1>::Copy(
                    _a + ((K * tile___i0) + tile___i2), trans__a, K, 1);

                    dace::CopyND<double, 1, false, 8, 128>::template ConstDst<128, 1>::Copy(
                    _b + ((N * tile___i2) + tile___i1), trans__b, N, 1);
                    {
                        {
                            {
                                {
                                    int __i2 = threadIdx.x;
                                    int __i1 = threadIdx.y;
                                    int __i0 = threadIdx.z;
                                    if (__i2 < 8) {
                                        if (__i1 < 128) {
                                            if (__i0 < 128) {
                                                {
                                                    double __a = trans__a[((8 * __i0) + __i2)];
                                                    double __b = trans__b[(__i1 + (128 * __i2))];
                                                    double __out;

                                                    ///////////////////
                                                    // Tasklet code (gemm)
                                                    __out = (__a * __b);
                                                    ///////////////////

                                                    dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce_atomic(_c + (((N * (__i0 + tile___i0)) + __i1) + tile___i1), __out);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_gemm_map_1_1_8(matmul_t *__state, const double * __restrict__ _a, const double * __restrict__ _b, double * __restrict__ _c, int K, int M, int N);
void __dace_runkernel_gemm_map_1_1_8(matmul_t *__state, const double * __restrict__ _a, const double * __restrict__ _b, double * __restrict__ _c, int K, int M, int N)
{

    void  *gemm_map_1_1_8_args[] = { (void *)&_a, (void *)&_b, (void *)&_c, (void *)&K, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)gemm_map_1_1_8, dim3(int_ceil(N, 128), int_ceil(M, 128), 1), dim3(8, 128, 128), gemm_map_1_1_8_args, 0, __state->gpu_context->streams[0]);
}

