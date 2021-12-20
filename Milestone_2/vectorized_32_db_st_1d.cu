
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

DACE_DFI void nested_nested_state_1_1_5(const dace::vec<float, 4> * input_A, const dace::vec<float, 4> * input_B, float * output, int K, int M, int N) {
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
    constexpr long long num_warps_n = 2;
    constexpr long long warp_width = 8;
    constexpr long long warp_height = 4;
    constexpr long long size_K_split = 4096;
    constexpr long long SWIZZLE = 1;
    float register_storage_B[8]  DACE_ALIGN(64);
    float register_storage_A[8]  DACE_ALIGN(64);
    __shared__ float shared_memory_B[2048];
    __shared__ float shared_memory_A[2048];
    float register_storage_C[64]  DACE_ALIGN(64) = {0};
    long long k_tile;

    {

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                dace::GlobalToShared2D<dace::vec<float, 4>, max(1, num_threads_per_threadblock), 1, 1, size_thread_block_tile_m, size_K_tile / 4, 8 / 4, 1, true>(input_A, K / 4, 1, (dace::vec<float, 4> *)shared_memory_A);
            } // End omp section
            #pragma omp section
            {
                dace::GlobalToShared2D<dace::vec<float, 4>, max(1, num_threads_per_threadblock), 1, 1, size_K_tile, size_thread_block_tile_n / 4, 128 / 4, 1, true>(input_B, N / 4, 1, (dace::vec<float, 4> *)shared_memory_B);
            } // End omp section
        } // End omp sections

    }

    for (k_tile = 0; (k_tile < (num_K_tiles - 1)); k_tile = k_tile + 1) {
        {

            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    {
                        {
                            __syncthreads();
                            int thread = threadIdx.x;
                            if (thread < num_threads_per_threadblock) {
                                {
                                    for (auto k = 0; k < size_K_tile; k += 1) {

                                        dace::CopyND<float, 1, false, size_thread_tile_m>::template ConstDst<1>::Copy(
                                        shared_memory_A + (((k + ((8 * size_thread_tile_m) * bitwise_and(right_shift((thread % 32), 1), (warp_height - 1)))) + ((8 * size_warp_tile_m) * ((thread / 32) / num_warps_n))) + (1024 * (k_tile % 2))), register_storage_A, 8);

                                        dace::CopyND<float, 1, false, size_thread_tile_n>::template ConstDst<1>::Copy(
                                        shared_memory_B + ((((128 * k) + (size_thread_tile_n * bitwise_or(right_shift(bitwise_and((thread % 32), 24), 2), bitwise_and((thread % 32), 1)))) + (size_warp_tile_n * ((thread / 32) % num_warps_n))) + (1024 * (k_tile % 2))), register_storage_B, 1);
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
                            }
                        }
                    }
                } // End omp section
                #pragma omp section
                {
                    dace::GlobalToShared2D<dace::vec<float, 4>, max(1, num_threads_per_threadblock), 1, 1, size_thread_block_tile_m, size_K_tile / 4, 8 / 4, 1, true>(input_A + (size_K_tile * (k_tile + 1)) / 4, K /4, 1, ((dace::vec<float, 4> *) shared_memory_A) + (1024 * ((k_tile + 1) % 2)) / 4);
                } // End omp section
                #pragma omp section
                {
                    dace::GlobalToShared2D<dace::vec<float, 4>, max(1, num_threads_per_threadblock), 1, 1, size_K_tile, size_thread_block_tile_n / 4, 128 / 4, 1, true>(input_B + ((N * size_K_tile) * (k_tile + 1)) / 4, N / 4, 1, ((dace::vec<float, 4> *) shared_memory_B) + (1024 * ((k_tile + 1) % 2)) / 4);
                } // End omp section
            } // End omp sections

        }

    }
    {

        {
            {
                __syncthreads();
                int thread = threadIdx.x;
                if (thread < num_threads_per_threadblock) {
                    {
                        for (auto k = 0; k < size_K_tile; k += 1) {

                            dace::CopyND<float, 1, false, size_thread_tile_m>::template ConstDst<1>::Copy(
                            shared_memory_A + (((k + ((8 * size_thread_tile_m) * bitwise_and(right_shift((thread % 32), 1), (warp_height - 1)))) + ((8 * size_warp_tile_m) * ((thread / 32) / num_warps_n))) + (1024 * (k_tile % 2))), register_storage_A, 8);

                            dace::CopyND<float, 1, false, size_thread_tile_n>::template ConstDst<1>::Copy(
                            shared_memory_B + ((((128 * k) + (size_thread_tile_n * bitwise_or(right_shift(bitwise_and((thread % 32), 24), 2), bitwise_and((thread % 32), 1)))) + (size_warp_tile_n * ((thread / 32) % num_warps_n))) + (1024 * (k_tile % 2))), register_storage_B, 1);
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

                    dace::CopyND<float, 1, false, size_thread_tile_m, size_thread_tile_n>::template ConstSrc<8, 1>::Accumulate(
                    register_storage_C, output + ((((N * ((((- size_thread_tile_m) * bitwise_and(right_shift(0, 1), (warp_height - 1))) + (size_thread_tile_m * bitwise_and(right_shift((thread % 32), 1), (warp_height - 1)))) + (size_warp_tile_m * ((thread / 32) / num_warps_n)))) - (size_thread_tile_n * bitwise_or(right_shift(bitwise_and(0, 24), 2), bitwise_and(0, 1)))) + (size_thread_tile_n * bitwise_or(right_shift(bitwise_and((thread % 32), 24), 2), bitwise_and((thread % 32), 1)))) + (size_warp_tile_n * ((thread / 32) % num_warps_n))), [] (const float& x, const float& y) { return (x + y); }, N, 1);
                }
            }
        }

    }
    
}



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

void __dace_exit_cuda(gemm_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 2; ++i) {
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
__global__ void Thread_block_grid_1_1_3(const dace::vec<float, 4> * __restrict__ input_A, const dace::vec<float, 4> * __restrict__ input_B, float * __restrict__ output, int K, int M, int N) {
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
            nested_nested_state_1_1_5(&input_A[((K * size_thread_block_tile_m) * thread_block_i) / 4], &input_B[(size_thread_block_tile_n * thread_block_j) / 4], &output[(((N * ((size_thread_block_tile_m * thread_block_i) + (size_thread_tile_m * bitwise_and(right_shift(0, 1), (warp_height - 1))))) + (size_thread_block_tile_n * thread_block_j)) + (size_thread_tile_n * bitwise_or(right_shift(bitwise_and(0, 24), 2), bitwise_and(0, 1))))], K, M, N);
        }
    }
}


DACE_EXPORTED void __dace_runkernel_Thread_block_grid_1_1_3(gemm_t *__state, const float * __restrict__ input_A, const float * __restrict__ input_B, float * __restrict__ output, int K, int M, int N);
void __dace_runkernel_Thread_block_grid_1_1_3(gemm_t *__state, const float * __restrict__ input_A, const float * __restrict__ input_B, float * __restrict__ output, int K, int M, int N)
{

    void  *Thread_block_grid_1_1_3_args[] = { (void *)&input_A, (void *)&input_B, (void *)&output, (void *)&K, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)Thread_block_grid_1_1_3, dim3(int_ceil(num_thread_blocks_n, 1), int_ceil(num_thread_blocks_m, 1), 1), dim3(max(1, num_threads_per_threadblock), 1, 1), Thread_block_grid_1_1_3_args, 0, __state->gpu_context->streams[0]);
}

