
#include <cuda_runtime.h>
#include <dace/dace.h>

constexpr long long VECLEN = 4;
constexpr float alpha = 1;
constexpr float beta = 0;
constexpr long long size_thread_block_tile_m = 128;
constexpr long long size_thread_block_tile_n = 128;
constexpr long long size_K_tile = 8;
constexpr long long num_thread_blocks_m = 32;
constexpr long long num_thread_blocks_n = 32;
constexpr long long num_K_tiles = 512;
constexpr long long size_warp_tile_m = 32;
constexpr long long size_warp_tile_n = 64;
constexpr long long size_thread_tile_m = 8;
constexpr long long size_thread_tile_n = 8;
constexpr long long SPLIT_K = 1;
constexpr long long num_threads_per_threadblock = 256;

struct gemm_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(gemm_t *__state, int K, int M, int N);
DACE_EXPORTED void __dace_exit_cuda(gemm_t *__state);



int __dace_init_cuda(gemm_t *__state, int K, int M, int N) {
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

    __state->gpu_context = new dace::cuda::Context(2, 4);

    // Create cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 4; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(gemm_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 4; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void initialize_matmul_result_1_0_1(float * __restrict__ output, int M, int N) {
    constexpr long long VECLEN = 4;
    constexpr long long size_thread_block_tile_m = 128;
    constexpr long long size_thread_block_tile_n = 128;
    constexpr long long num_thread_blocks_m = 32;
    constexpr long long num_thread_blocks_n = 32;
    constexpr long long num_warps_n = 2;
    constexpr long long num_K_tiles = 512;
    constexpr long long size_warp_tile_m = 32;
    constexpr long long size_warp_tile_n = 64;
    constexpr long long size_thread_tile_m = 8;
    constexpr long long size_thread_tile_n = 8;
    constexpr long long warp_width = 8;
    constexpr long long warp_height = 4;
    constexpr long long size_K_tile = 8;
    constexpr long long size_K_split = 4096;
    constexpr long long SWIZZLE = 1;
    constexpr long long SPLIT_K = 1;
    constexpr long long num_threads_per_threadblock = 256;
    {
        {
            int j = (blockIdx.x * 32 + threadIdx.x);
            int i = (blockIdx.y * 1 + threadIdx.y);
            if (j < N) {
                {
                    {
                        float out;

                        ///////////////////
                        // Tasklet code (matmul_init)
                        out = 0;
                        ///////////////////

                        output[((N * i) + j)] = out;
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_initialize_matmul_result_1_0_1(gemm_t *__state, float * __restrict__ output, int M, int N);
void __dace_runkernel_initialize_matmul_result_1_0_1(gemm_t *__state, float * __restrict__ output, int M, int N)
{

    void  *initialize_matmul_result_1_0_1_args[] = { (void *)&output, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)initialize_matmul_result_1_0_1, dim3(int_ceil(int_ceil(N, 1), 32), int_ceil(int_ceil(M, 1), 1), int_ceil(1, 1)), dim3(32, 1, 1), initialize_matmul_result_1_0_1_args, 0, __state->gpu_context->streams[0]);
}
__global__ void Thread_block_grid_1_1_9(const float * __restrict__ input_A, const float * __restrict__ input_B, float * __restrict__ output, int K, int M, int N) {
    constexpr long long VECLEN = 4;
    constexpr long long size_thread_block_tile_m = 128;
    constexpr long long size_thread_block_tile_n = 128;
    constexpr long long num_thread_blocks_m = 32;
    constexpr long long num_thread_blocks_n = 32;
    constexpr long long num_warps_n = 2;
    constexpr long long num_K_tiles = 512;
    constexpr long long size_warp_tile_m = 32;
    constexpr long long size_warp_tile_n = 64;
    constexpr long long size_thread_tile_m = 8;
    constexpr long long size_thread_tile_n = 8;
    constexpr long long warp_width = 8;
    constexpr long long warp_height = 4;
    constexpr long long size_K_tile = 8;
    constexpr long long size_K_split = 4096;
    constexpr long long SWIZZLE = 1;
    constexpr long long SPLIT_K = 1;
    constexpr long long num_threads_per_threadblock = 256;
    {
        {
            int thread_block_j = blockIdx.x;
            int thread_block_i = blockIdx.y;
            {
                for (auto k_tile = 0; k_tile < num_K_tiles; k_tile += 1) {
                    __shared__ float shared_memory_A[1024];
                    __shared__ float shared_memory_B[1024];
                    dace::GlobalToShared2D<float, max(1, num_threads_per_threadblock), 1, 1, size_thread_block_tile_m, size_K_tile, 8, 1, true>(input_A + (((K * size_thread_block_tile_m) * thread_block_i) + (k_tile * size_K_tile)), K, 1, shared_memory_A);
                    dace::GlobalToShared2D<float, max(1, num_threads_per_threadblock), 1, 1, size_K_tile, size_thread_block_tile_n, 128, 1, true>(input_B + (((N * k_tile) * size_K_tile) + (size_thread_block_tile_n * thread_block_j)), N, 1, shared_memory_B);
                    {
                        {
                            float register_storage_C[64]  DACE_ALIGN(64) = {0};
                            __syncthreads();
                            int thread = threadIdx.x;
                            if (thread < num_threads_per_threadblock) {
                                {
                                    for (auto k = 0; k < size_K_tile; k += 1) {
                                        float register_storage_A[8]  DACE_ALIGN(64);
                                        float register_storage_B[8]  DACE_ALIGN(64);

                                        dace::CopyND<float, 1, false, size_thread_tile_m>::template ConstDst<1>::Copy(
                                        shared_memory_A + ((k + ((8 * size_thread_tile_m) * ((thread % 32) / warp_width))) + ((8 * size_warp_tile_m) * ((thread / 32) / num_warps_n))), register_storage_A, 8);

                                        dace::CopyND<float, 1, false, size_thread_tile_n>::template ConstDst<1>::Copy(
                                        shared_memory_B + (((128 * k) + (size_thread_tile_n * ((thread % 32) % warp_width))) + (size_warp_tile_n * ((thread / 32) % num_warps_n))), register_storage_B, 1);
                                        {
                                            #pragma unroll
                                            for (auto i = 0; i < size_thread_tile_m; i += 1) {
                                                #pragma unroll
                                                for (auto j = 0; j < size_thread_tile_n; j += 1) {
                                                    {
                                                        float __a = register_storage_A[i];
                                                        float __b = register_storage_B[j];
                                                        float __out;

                                                        ///////////////////
                                                        // Tasklet code (matrix_multiplication)
                                                        __out = (__a * __b);
                                                        ///////////////////

                                                        dace::wcr_fixed<dace::ReductionType::Sum, float>::reduce(register_storage_C + ((8 * i) + j), __out);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                dace::CopyND<float, 1, false, size_thread_tile_m, size_thread_tile_n>::template ConstSrc<8, 1>::Accumulate_atomic(
                                register_storage_C, output + ((((N * (((size_thread_block_tile_m * thread_block_i) + (size_thread_tile_m * ((thread % 32) / warp_width))) + (size_warp_tile_m * ((thread / 32) / num_warps_n)))) + (size_thread_block_tile_n * thread_block_j)) + (size_thread_tile_n * ((thread % 32) % warp_width))) + (size_warp_tile_n * ((thread / 32) % num_warps_n))), [] (const float& x, const float& y) { return (x + y); }, N, 1);
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_Thread_block_grid_1_1_9(gemm_t *__state, const float * __restrict__ input_A, const float * __restrict__ input_B, float * __restrict__ output, int K, int M, int N);
void __dace_runkernel_Thread_block_grid_1_1_9(gemm_t *__state, const float * __restrict__ input_A, const float * __restrict__ input_B, float * __restrict__ output, int K, int M, int N)
{

    void  *Thread_block_grid_1_1_9_args[] = { (void *)&input_A, (void *)&input_B, (void *)&output, (void *)&K, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)Thread_block_grid_1_1_9, dim3(int_ceil(num_thread_blocks_n, 1), int_ceil(num_thread_blocks_m, 1), 1), dim3(max(1, num_threads_per_threadblock), 1, 1), Thread_block_grid_1_1_9_args, 0, __state->gpu_context->streams[0]);
}

