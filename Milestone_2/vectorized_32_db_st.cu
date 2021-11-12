
#include <cuda_runtime.h>
#include <dace/dace.h>

constexpr long long VECLEN = 4;
constexpr double alpha = 1;
constexpr double beta = 1;
constexpr long long size_thread_block_tile_m = 64;
constexpr long long size_thread_block_tile_n = 64;
constexpr long long size_K_tile = 4;
constexpr long long num_thread_blocks_m = 64;
constexpr long long num_thread_blocks_n = 64;
constexpr long long num_K_tiles = 1024;
constexpr long long size_warp_tile_m = 32;
constexpr long long size_warp_tile_n = 16;
constexpr long long size_thread_tile_m = 4;
constexpr long long size_thread_tile_n = 4;
constexpr long long SPLIT_K = 1;

struct gemm_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(gemm_t *__state, int K, int M, int N);
DACE_EXPORTED void __dace_exit_cuda(gemm_t *__state);

DACE_DFI void nested_nested_state_1_1_5(const dace::vec<float, 4> * input_A, const dace::vec<float, 4> * input_B, float * output, int K, int M, int N) {
    constexpr long long VECLEN = 4;
    constexpr double alpha = 1;
    constexpr double beta = 1;
    constexpr long long size_thread_block_tile_m = 64;
    constexpr long long size_thread_block_tile_n = 64;
    constexpr long long size_K_tile = 4;
    constexpr long long num_thread_blocks_m = 64;
    constexpr long long num_thread_blocks_n = 64;
    constexpr long long num_K_tiles = 1024;
    constexpr long long size_warp_tile_m = 32;
    constexpr long long size_warp_tile_n = 16;
    constexpr long long size_thread_tile_m = 4;
    constexpr long long size_thread_tile_n = 4;
    constexpr long long SPLIT_K = 1;
    constexpr long long warp_width = 4;
    constexpr long long warp_height = 8;
    constexpr long long size_K_split = 4096;
    constexpr long long SWIZZLE = 1;
    float register_storage_B[4]  DACE_ALIGN(64);
    float register_storage_C[16]  DACE_ALIGN(64) = {0};
    __shared__ float shared_memory_A[512];
    float register_storage_A[4]  DACE_ALIGN(64);
    __shared__ float shared_memory_B[512];
    long long k_tile;

    {

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                dace::GlobalToShared2D<dace::vec<float, 4>, max(1, int_ceil(size_thread_block_tile_n, size_thread_tile_n)), max(1, int_ceil(size_thread_block_tile_m, size_thread_tile_m)), 1, size_thread_block_tile_m, size_K_tile / 4, 4 / 4, 1, true>(input_A, K / 4, 1, (dace::vec<float, 4> *)shared_memory_A);
            } // End omp section
            #pragma omp section
            {
                dace::GlobalToShared2D<dace::vec<float, 4>, max(1, int_ceil(size_thread_block_tile_n, size_thread_tile_n)), max(1, int_ceil(size_thread_block_tile_m, size_thread_tile_m)), 1, size_K_tile, size_thread_block_tile_n / 4, 64 / 4, 1, true>(input_B, N / 4, 1, (dace::vec<float, 4> *)shared_memory_B);
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
                            {
                                __syncthreads();
                                int thread_j = (threadIdx.x * size_thread_tile_n);
                                int thread_i = (threadIdx.y * size_thread_tile_m);
                                if (thread_j < size_thread_block_tile_n) {
                                    if (thread_i < size_thread_block_tile_m) {
                                        {
                                            for (auto k = 0; k < size_K_tile; k += 1) {

                                                dace::CopyND<float, 1, false, size_thread_tile_m>::template ConstDst<1>::Copy(
                                                shared_memory_A + (((k + ((4 * size_thread_tile_m) * bitwise_and(right_shift((((thread_j % size_warp_tile_n) / size_thread_tile_n) + ((warp_width * (thread_i % size_warp_tile_m)) / size_thread_tile_m)), 1), (warp_height - 1)))) + ((4 * size_warp_tile_m) * (thread_i / size_warp_tile_m))) + (256 * (k_tile % 2))), register_storage_A, 4);

                                                dace::CopyND<float, 1, false, size_thread_tile_n>::template ConstDst<1>::Copy(
                                                shared_memory_B + ((((64 * k) + (size_thread_tile_n * bitwise_or(right_shift(bitwise_and((((thread_j % size_warp_tile_n) / size_thread_tile_n) + ((warp_width * (thread_i % size_warp_tile_m)) / size_thread_tile_m)), ((warp_height * warp_width) / 2)), (warp_width - 1)), bitwise_and((((thread_j % size_warp_tile_n) / size_thread_tile_n) + ((warp_width * (thread_i % size_warp_tile_m)) / size_thread_tile_m)), 1)))) + (size_warp_tile_n * (thread_j / size_warp_tile_n))) + (256 * (k_tile % 2))), register_storage_B, 1);
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

                                                                dace::wcr_fixed<dace::ReductionType::Sum, float>::reduce(register_storage_C + ((4 * i) + j), __out);
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
                } // End omp section
                #pragma omp section
                {
                    dace::GlobalToShared2D<dace::vec<float, 4>, max(1, int_ceil(size_thread_block_tile_n, size_thread_tile_n)), max(1, int_ceil(size_thread_block_tile_m, size_thread_tile_m)), 1, size_thread_block_tile_m, size_K_tile / 4, 4 / 4, 1, true>(input_A + (size_K_tile * (k_tile + 1)) / 4, K / 4, 1, ((dace::vec<float, 4> *) shared_memory_A) + (256 * ((k_tile + 1) % 2)) / 4);
                } // End omp section
                #pragma omp section
                {
                    dace::GlobalToShared2D<dace::vec<float, 4>, max(1, int_ceil(size_thread_block_tile_n, size_thread_tile_n)), max(1, int_ceil(size_thread_block_tile_m, size_thread_tile_m)), 1, size_K_tile, size_thread_block_tile_n / 4, 64 / 4, 1, true>(input_B + ((N * size_K_tile) * (k_tile + 1)) / 4, N / 4, 1, ((dace::vec<float, 4> *) shared_memory_B) + (256 * ((k_tile + 1) % 2)) / 4);
                } // End omp section
            } // End omp sections

        }

    }
    {

        {
            {
                {
                    __syncthreads();
                    int thread_j = (threadIdx.x * size_thread_tile_n);
                    int thread_i = (threadIdx.y * size_thread_tile_m);
                    if (thread_j < size_thread_block_tile_n) {
                        if (thread_i < size_thread_block_tile_m) {
                            {
                                for (auto k = 0; k < size_K_tile; k += 1) {

                                    dace::CopyND<float, 1, false, size_thread_tile_m>::template ConstDst<1>::Copy(
                                    shared_memory_A + (((k + ((4 * size_thread_tile_m) * bitwise_and(right_shift((((thread_j % size_warp_tile_n) / size_thread_tile_n) + ((warp_width * (thread_i % size_warp_tile_m)) / size_thread_tile_m)), 1), (warp_height - 1)))) + ((4 * size_warp_tile_m) * (thread_i / size_warp_tile_m))) + (256 * (k_tile % 2))), register_storage_A, 4);

                                    dace::CopyND<float, 1, false, size_thread_tile_n>::template ConstDst<1>::Copy(
                                    shared_memory_B + ((((64 * k) + (size_thread_tile_n * bitwise_or(right_shift(bitwise_and((((thread_j % size_warp_tile_n) / size_thread_tile_n) + ((warp_width * (thread_i % size_warp_tile_m)) / size_thread_tile_m)), ((warp_height * warp_width) / 2)), (warp_width - 1)), bitwise_and((((thread_j % size_warp_tile_n) / size_thread_tile_n) + ((warp_width * (thread_i % size_warp_tile_m)) / size_thread_tile_m)), 1)))) + (size_warp_tile_n * (thread_j / size_warp_tile_n))) + (256 * (k_tile % 2))), register_storage_B, 1);
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

                                                    dace::wcr_fixed<dace::ReductionType::Sum, float>::reduce(register_storage_C + ((4 * i) + j), __out);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            dace::CopyND<float, 1, false, size_thread_tile_m, size_thread_tile_n>::template ConstSrc<4, 1>::Accumulate(
                            register_storage_C, output + ((((N * ((((- size_thread_tile_m) * bitwise_and(right_shift(0, 1), (warp_height - 1))) + (size_thread_tile_m * bitwise_and(right_shift((((thread_j % size_warp_tile_n) / size_thread_tile_n) + ((warp_width * (thread_i % size_warp_tile_m)) / size_thread_tile_m)), 1), (warp_height - 1)))) + (size_warp_tile_m * (thread_i / size_warp_tile_m)))) - (size_thread_tile_n * bitwise_or(right_shift(bitwise_and(0, ((warp_height * warp_width) / 2)), (warp_width - 1)), bitwise_and(0, 1)))) + (size_thread_tile_n * bitwise_or(right_shift(bitwise_and((((thread_j % size_warp_tile_n) / size_thread_tile_n) + ((warp_width * (thread_i % size_warp_tile_m)) / size_thread_tile_m)), ((warp_height * warp_width) / 2)), (warp_width - 1)), bitwise_and((((thread_j % size_warp_tile_n) / size_thread_tile_n) + ((warp_width * (thread_i % size_warp_tile_m)) / size_thread_tile_m)), 1)))) + (size_warp_tile_n * (thread_j / size_warp_tile_n))), [] (const float& x, const float& y) { return (x + y); }, N, 1);
                        }
                    }
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

    __state->gpu_context = new dace::cuda::Context(3, 3);

    // Create cuda streams and events
    for(int i = 0; i < 3; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 3; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(gemm_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 3; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 3; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void initialize_matmul_result_1_0_1(float * __restrict__ output, int M, int N) {
    constexpr long long VECLEN = 4;
    constexpr long long size_thread_block_tile_m = 64;
    constexpr long long size_thread_block_tile_n = 64;
    constexpr long long num_thread_blocks_m = 64;
    constexpr long long num_thread_blocks_n = 64;
    constexpr long long num_K_tiles = 1024;
    constexpr long long size_warp_tile_m = 32;
    constexpr long long size_warp_tile_n = 16;
    constexpr long long size_thread_tile_m = 4;
    constexpr long long size_thread_tile_n = 4;
    constexpr long long warp_width = 4;
    constexpr long long warp_height = 8;
    constexpr long long size_K_tile = 4;
    constexpr long long size_K_split = 4096;
    constexpr long long SWIZZLE = 1;
    constexpr long long SPLIT_K = 1;
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
    constexpr long long size_thread_block_tile_m = 64;
    constexpr long long size_thread_block_tile_n = 64;
    constexpr long long num_thread_blocks_m = 64;
    constexpr long long num_thread_blocks_n = 64;
    constexpr long long num_K_tiles = 1024;
    constexpr long long size_warp_tile_m = 32;
    constexpr long long size_warp_tile_n = 16;
    constexpr long long size_thread_tile_m = 4;
    constexpr long long size_thread_tile_n = 4;
    constexpr long long warp_width = 4;
    constexpr long long warp_height = 8;
    constexpr long long size_K_tile = 4;
    constexpr long long size_K_split = 4096;
    constexpr long long SWIZZLE = 1;
    constexpr long long SPLIT_K = 1;
    {
        {
            int thread_block_j = blockIdx.x;
            int thread_block_i = blockIdx.y;
            nested_nested_state_1_1_5(&input_A[((K * size_thread_block_tile_m) * thread_block_i) / 4], &input_B[(size_thread_block_tile_n * thread_block_j) / 4], &output[(((N * ((size_thread_block_tile_m * thread_block_i) + (size_thread_tile_m * bitwise_and(right_shift(0, 1), (warp_height - 1))))) + (size_thread_block_tile_n * thread_block_j)) + (size_thread_tile_n * bitwise_or(right_shift(bitwise_and(0, ((warp_height * warp_width) / 2)), (warp_width - 1)), bitwise_and(0, 1))))], K, M, N);
        }
    }
}


DACE_EXPORTED void __dace_runkernel_Thread_block_grid_1_1_3(gemm_t *__state, const float * __restrict__ input_A, const float * __restrict__ input_B, float * __restrict__ output, int K, int M, int N);
void __dace_runkernel_Thread_block_grid_1_1_3(gemm_t *__state, const float * __restrict__ input_A, const float * __restrict__ input_B, float * __restrict__ output, int K, int M, int N)
{

    void  *Thread_block_grid_1_1_3_args[] = { (void *)&input_A, (void *)&input_B, (void *)&output, (void *)&K, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)Thread_block_grid_1_1_3, dim3(int_ceil(num_thread_blocks_n, 1), int_ceil(num_thread_blocks_m, 1), 1), dim3(max(1, int_ceil(size_thread_block_tile_n, size_thread_tile_n)), max(1, int_ceil(size_thread_block_tile_m, size_thread_tile_m)), 1), Thread_block_grid_1_1_3_args, 0, __state->gpu_context->streams[0]);
}
__global__ void multiply_matrix_with_constant_0_0_14(const float * __restrict__ A_matmul_B, float * __restrict__ A_matmul_B_times_alpha, int M, int N) {
    constexpr long long VECLEN = 4;
    constexpr double alpha = 1;
    constexpr double beta = 1;
    constexpr long long size_thread_block_tile_m = 64;
    constexpr long long size_thread_block_tile_n = 64;
    constexpr long long size_K_tile = 4;
    constexpr long long num_thread_blocks_m = 64;
    constexpr long long num_thread_blocks_n = 64;
    constexpr long long num_K_tiles = 1024;
    constexpr long long size_warp_tile_m = 32;
    constexpr long long size_warp_tile_n = 16;
    constexpr long long size_thread_tile_m = 4;
    constexpr long long size_thread_tile_n = 4;
    constexpr long long SPLIT_K = 1;
    {
        {
            int j = (blockIdx.x * 32 + threadIdx.x);
            int i = (blockIdx.y * 1 + threadIdx.y);
            if (j < N) {
                {
                    {
                        float __in = A_matmul_B[((N * i) + j)];
                        float __out;

                        ///////////////////
                        // Tasklet code (multiply_matrix_with_constant)
                        __out = (alpha * __in);
                        ///////////////////

                        A_matmul_B_times_alpha[((N * i) + j)] = __out;
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_multiply_matrix_with_constant_0_0_14(gemm_t *__state, const float * __restrict__ A_matmul_B, float * __restrict__ A_matmul_B_times_alpha, int M, int N);
void __dace_runkernel_multiply_matrix_with_constant_0_0_14(gemm_t *__state, const float * __restrict__ A_matmul_B, float * __restrict__ A_matmul_B_times_alpha, int M, int N)
{

    void  *multiply_matrix_with_constant_0_0_14_args[] = { (void *)&A_matmul_B, (void *)&A_matmul_B_times_alpha, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)multiply_matrix_with_constant_0_0_14, dim3(int_ceil(int_ceil(N, 1), 32), int_ceil(int_ceil(M, 1), 1), int_ceil(1, 1)), dim3(32, 1, 1), multiply_matrix_with_constant_0_0_14_args, 0, __state->gpu_context->streams[0]);
}
__global__ void multiply_matrix_with_constant_0_0_11(float * __restrict__ C_times_beta, const float * __restrict__ gpu_C, int M, int N) {
    constexpr long long VECLEN = 4;
    constexpr double alpha = 1;
    constexpr double beta = 1;
    constexpr long long size_thread_block_tile_m = 64;
    constexpr long long size_thread_block_tile_n = 64;
    constexpr long long size_K_tile = 4;
    constexpr long long num_thread_blocks_m = 64;
    constexpr long long num_thread_blocks_n = 64;
    constexpr long long num_K_tiles = 1024;
    constexpr long long size_warp_tile_m = 32;
    constexpr long long size_warp_tile_n = 16;
    constexpr long long size_thread_tile_m = 4;
    constexpr long long size_thread_tile_n = 4;
    constexpr long long SPLIT_K = 1;
    {
        {
            int j = (blockIdx.x * 32 + threadIdx.x);
            int i = (blockIdx.y * 1 + threadIdx.y);
            if (j < N) {
                {
                    {
                        float __in = gpu_C[((N * i) + j)];
                        float __out;

                        ///////////////////
                        // Tasklet code (multiply_matrix_with_constant)
                        __out = (beta * __in);
                        ///////////////////

                        C_times_beta[((N * i) + j)] = __out;
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_multiply_matrix_with_constant_0_0_11(gemm_t *__state, float * __restrict__ C_times_beta, const float * __restrict__ gpu_C, int M, int N);
void __dace_runkernel_multiply_matrix_with_constant_0_0_11(gemm_t *__state, float * __restrict__ C_times_beta, const float * __restrict__ gpu_C, int M, int N)
{

    void  *multiply_matrix_with_constant_0_0_11_args[] = { (void *)&C_times_beta, (void *)&gpu_C, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)multiply_matrix_with_constant_0_0_11, dim3(int_ceil(int_ceil(N, 1), 32), int_ceil(int_ceil(M, 1), 1), int_ceil(1, 1)), dim3(32, 1, 1), multiply_matrix_with_constant_0_0_11_args, 0, __state->gpu_context->streams[2]);
}
__global__ void add_matrices_0_0_17(const float * __restrict__ A_matmul_B_times_alpha, const float * __restrict__ C_times_beta, float * __restrict__ gpu_result, int M, int N) {
    constexpr long long VECLEN = 4;
    constexpr double alpha = 1;
    constexpr double beta = 1;
    constexpr long long size_thread_block_tile_m = 64;
    constexpr long long size_thread_block_tile_n = 64;
    constexpr long long size_K_tile = 4;
    constexpr long long num_thread_blocks_m = 64;
    constexpr long long num_thread_blocks_n = 64;
    constexpr long long num_K_tiles = 1024;
    constexpr long long size_warp_tile_m = 32;
    constexpr long long size_warp_tile_n = 16;
    constexpr long long size_thread_tile_m = 4;
    constexpr long long size_thread_tile_n = 4;
    constexpr long long SPLIT_K = 1;
    {
        {
            int j = (blockIdx.x * 32 + threadIdx.x);
            int i = (blockIdx.y * 1 + threadIdx.y);
            if (j < N) {
                {
                    {
                        float __in1 = A_matmul_B_times_alpha[((N * i) + j)];
                        float __in2 = C_times_beta[((N * i) + j)];
                        float __out;

                        ///////////////////
                        // Tasklet code (add_matrices)
                        __out = (__in1 + __in2);
                        ///////////////////

                        gpu_result[((N * i) + j)] = __out;
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_add_matrices_0_0_17(gemm_t *__state, const float * __restrict__ A_matmul_B_times_alpha, const float * __restrict__ C_times_beta, float * __restrict__ gpu_result, int M, int N);
void __dace_runkernel_add_matrices_0_0_17(gemm_t *__state, const float * __restrict__ A_matmul_B_times_alpha, const float * __restrict__ C_times_beta, float * __restrict__ gpu_result, int M, int N)
{

    void  *add_matrices_0_0_17_args[] = { (void *)&A_matmul_B_times_alpha, (void *)&C_times_beta, (void *)&gpu_result, (void *)&M, (void *)&N };
    cudaLaunchKernel((void*)add_matrices_0_0_17, dim3(int_ceil(int_ceil(N, 1), 32), int_ceil(int_ceil(M, 1), 1), int_ceil(1, 1)), dim3(32, 1, 1), add_matrices_0_0_17_args, 0, __state->gpu_context->streams[0]);
}

