
#include <cuda_runtime.h>
#include <dace/dace.h>

constexpr double alpha = 1;
constexpr double beta = 1;
constexpr long long size_thread_block_tile_m = 128;
constexpr long long size_thread_block_tile_n = 64;
constexpr long long size_K_tile = 8;
constexpr long long num_thread_blocks_m = 8;
constexpr long long num_thread_blocks_n = 16;
constexpr long long num_K_tiles = 128;
constexpr long long size_warp_tile_m = 64;
constexpr long long size_warp_tile_n = 32;
constexpr long long size_thread_tile_m = 8;
constexpr long long size_thread_tile_n = 8;
constexpr long long SPLIT_K = 1;

struct gemm_t {
  dace::cuda::Context *gpu_context;
};

DACE_EXPORTED int __dace_init_cuda(gemm_t *__state, int K, int M, int N);
DACE_EXPORTED void __dace_exit_cuda(gemm_t *__state);

int __dace_init_cuda(gemm_t *__state, int K, int M, int N) {
  int count;

  // Check that we are able to run cuda code
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    printf("ERROR: GPU drivers are not configured or cuda-capable device "
           "not found\n");
    return 1;
  }
  if (count == 0) {
    printf("ERROR: No cuda-capable devices found\n");
    return 2;
  }

  // Initialize cuda before we run the application
  float *dev_X;
  cudaMalloc((void **)&dev_X, 1);
  cudaFree(dev_X);

  __state->gpu_context = new dace::cuda::Context(3, 5);

  // Create cuda streams and events
  for (int i = 0; i < 3; ++i) {
    cudaStreamCreateWithFlags(&__state->gpu_context->streams[i],
                              cudaStreamNonBlocking);
  }
  for (int i = 0; i < 5; ++i) {
    cudaEventCreateWithFlags(&__state->gpu_context->events[i],
                             cudaEventDisableTiming);
  }

  return 0;
}

void __dace_exit_cuda(gemm_t *__state) {

  // Destroy cuda streams and events
  for (int i = 0; i < 3; ++i) {
    cudaStreamDestroy(__state->gpu_context->streams[i]);
  }
  for (int i = 0; i < 5; ++i) {
    cudaEventDestroy(__state->gpu_context->events[i]);
  }

  delete __state->gpu_context;
}

__global__ void initialize_matmul_result_1_0_1(double *__restrict__ output,
                                               int M, int N) {
  constexpr long long size_thread_block_tile_m = 128;
  constexpr long long size_thread_block_tile_n = 64;
  constexpr long long num_thread_blocks_m = 8;
  constexpr long long num_thread_blocks_n = 16;
  constexpr long long num_K_tiles = 128;
  constexpr long long size_warp_tile_m = 64;
  constexpr long long size_warp_tile_n = 32;
  constexpr long long size_thread_tile_m = 8;
  constexpr long long size_thread_tile_n = 8;
  constexpr long long warp_width = 4;
  constexpr long long warp_height = 8;
  constexpr long long size_K_tile = 8;
  constexpr long long size_K_split = 1024;
  constexpr long long SWIZZLE = 1;
  constexpr long long SPLIT_K = 1;
  {
    {
      int j = (blockIdx.x * 32 + threadIdx.x);
      int i = (blockIdx.y * 1 + threadIdx.y);
      if (j < N) {
        {
          {
            double out;

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

DACE_EXPORTED void __dace_runkernel_initialize_matmul_result_1_0_1(
    gemm_t *__state, double *__restrict__ output, int M, int N);
void __dace_runkernel_initialize_matmul_result_1_0_1(
    gemm_t *__state, double *__restrict__ output, int M, int N) {

  void *initialize_matmul_result_1_0_1_args[] = {(void *)&output, (void *)&M,
                                                 (void *)&N};
  cudaLaunchKernel((void *)initialize_matmul_result_1_0_1,
                   dim3(int_ceil(int_ceil(N, 1), 32),
                        int_ceil(int_ceil(M, 1), 1), int_ceil(1, 1)),
                   dim3(32, 1, 1), initialize_matmul_result_1_0_1_args, 0,
                   __state->gpu_context->streams[0]);
}
__global__ void
Thread_block_grid_1_1_9(const dace::vec<double, 2> *__restrict__ input_A,
                        const dace::vec<double, 2> *__restrict__ input_B,
                        double *__restrict__ output, int K, int M, int N) {
  constexpr long long size_thread_block_tile_m = 128;
  constexpr long long size_thread_block_tile_n = 64;
  constexpr long long num_thread_blocks_m = 8;
  constexpr long long num_thread_blocks_n = 16;
  constexpr long long num_K_tiles = 128;
  constexpr long long size_warp_tile_m = 64;
  constexpr long long size_warp_tile_n = 32;
  constexpr long long size_thread_tile_m = 8;
  constexpr long long size_thread_tile_n = 8;
  constexpr long long warp_width = 4;
  constexpr long long warp_height = 8;
  constexpr long long size_K_tile = 8;
  constexpr long long size_K_split = 1024;
  constexpr long long SWIZZLE = 1;
  constexpr long long SPLIT_K = 1;
  {
    {
      int thread_block_j = blockIdx.x;
      int thread_block_i = blockIdx.y;
      {
        for (auto k_tile = 0; k_tile < num_K_tiles; k_tile += 1) {
          __shared__ double shared_memory_A[1024];
          __shared__ double shared_memory_B[512];
          dace::GlobalToShared2D<
              dace::vec<double, 2>,
              max(1, int_ceil(size_thread_block_tile_n, size_thread_tile_n)),
              max(1, int_ceil(size_thread_block_tile_m, size_thread_tile_m)), 1,
              size_thread_block_tile_m, size_K_tile / 2, 8 / 2, 1, true>(
              input_A + (((K * size_thread_block_tile_m) * thread_block_i) +
                         (k_tile * size_K_tile)) /
                            2,
              K / 2, 1, (dace::vec<double, 2> *)shared_memory_A);
          dace::GlobalToShared2D<
              dace::vec<double, 2>,
              max(1, int_ceil(size_thread_block_tile_n, size_thread_tile_n)),
              max(1, int_ceil(size_thread_block_tile_m, size_thread_tile_m)), 1,
              size_K_tile, size_thread_block_tile_n / 2, 64 / 2, 1, true>(
              input_B + (((N * k_tile) * size_K_tile) +
                         (size_thread_block_tile_n * thread_block_j)) /
                            2,
              N / 2, 1, (dace::vec<double, 2> *)shared_memory_B);
          {
            {
              {
                double register_storage_C[64] DACE_ALIGN(64) = {0};
                __syncthreads();
                int thread_j = (threadIdx.x * size_thread_tile_n);
                int thread_i = (threadIdx.y * size_thread_tile_m);
                if (thread_j < size_thread_block_tile_n) {
                  if (thread_i < size_thread_block_tile_m) {
                    {
                      for (auto k = 0; k < size_K_tile; k += 1) {
                        double register_storage_A[8] DACE_ALIGN(64);
                        double register_storage_B[8] DACE_ALIGN(64);

                        dace::CopyND<double, 1, false, size_thread_tile_m>::
                            template ConstDst<1>::Copy(shared_memory_A +
                                                           (k + (8 * thread_i)),
                                                       register_storage_A, 8);

                        dace::CopyND<double, 1, false, size_thread_tile_n>::
                            template ConstDst<1>::Copy(
                                shared_memory_B + ((64 * k) + thread_j),
                                register_storage_B, 1);
                        {
#pragma unroll
                          for (auto i = 0; i < size_thread_tile_m; i += 1) {
#pragma unroll
                            for (auto j = 0; j < size_thread_tile_n; j += 1) {
                              {
                                double __a = register_storage_A[i];
                                double __b = register_storage_B[j];
                                double __out;

                                ///////////////////
                                // Tasklet code (matrix_multiplication)
                                __out = (__a * __b);
                                ///////////////////

                                dace::wcr_fixed<
                                    dace::ReductionType::Sum,
                                    double>::reduce(register_storage_C +
                                                        ((8 * i) + j),
                                                    __out);
                              }
                            }
                          }
                        }
                      }
                    }

                    dace::CopyND<double, 1, false, size_thread_tile_m,
                                 size_thread_tile_n>::template ConstSrc<8, 1>::
                        Accumulate_atomic(
                            register_storage_C,
                            output +
                                (((N * ((size_thread_block_tile_m *
                                         thread_block_i) +
                                        thread_i)) +
                                  (size_thread_block_tile_n * thread_block_j)) +
                                 thread_j),
                            [](const double &x, const double &y) {
                              return (x + y);
                            },
                            N, 1);
                  }
                }
              }
            }
          }
          __syncthreads();
        }
      }
    }
  }
}

DACE_EXPORTED void __dace_runkernel_Thread_block_grid_1_1_9(
    gemm_t *__state, const double *__restrict__ input_A,
    const double *__restrict__ input_B, double *__restrict__ output, int K,
    int M, int N);
void __dace_runkernel_Thread_block_grid_1_1_9(
    gemm_t *__state, const double *__restrict__ input_A,
    const double *__restrict__ input_B, double *__restrict__ output, int K,
    int M, int N) {

  void *Thread_block_grid_1_1_9_args[] = {(void *)&input_A, (void *)&input_B,
                                          (void *)&output,  (void *)&K,
                                          (void *)&M,       (void *)&N};
  cudaLaunchKernel(
      (void *)Thread_block_grid_1_1_9,
      dim3(int_ceil(num_thread_blocks_n, 1), int_ceil(num_thread_blocks_m, 1),
           1),
      dim3(max(1, int_ceil(size_thread_block_tile_n, size_thread_tile_n)),
           max(1, int_ceil(size_thread_block_tile_m, size_thread_tile_m)), 1),
      Thread_block_grid_1_1_9_args, 0, __state->gpu_context->streams[0]);
}
__global__ void multiply_matrix_with_constant_0_0_14(
    const double *__restrict__ A_matmul_B,
    double *__restrict__ A_matmul_B_times_alpha, int M, int N) {
  constexpr double alpha = 1;
  constexpr double beta = 1;
  constexpr long long size_thread_block_tile_m = 128;
  constexpr long long size_thread_block_tile_n = 64;
  constexpr long long size_K_tile = 8;
  constexpr long long num_thread_blocks_m = 8;
  constexpr long long num_thread_blocks_n = 16;
  constexpr long long num_K_tiles = 128;
  constexpr long long size_warp_tile_m = 64;
  constexpr long long size_warp_tile_n = 32;
  constexpr long long size_thread_tile_m = 8;
  constexpr long long size_thread_tile_n = 8;
  constexpr long long SPLIT_K = 1;
  {
    {
      int j = (blockIdx.x * 32 + threadIdx.x);
      int i = (blockIdx.y * 1 + threadIdx.y);
      if (j < N) {
        {
          {
            double __in = A_matmul_B[((N * i) + j)];
            double __out;

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

DACE_EXPORTED void __dace_runkernel_multiply_matrix_with_constant_0_0_14(
    gemm_t *__state, const double *__restrict__ A_matmul_B,
    double *__restrict__ A_matmul_B_times_alpha, int M, int N);
void __dace_runkernel_multiply_matrix_with_constant_0_0_14(
    gemm_t *__state, const double *__restrict__ A_matmul_B,
    double *__restrict__ A_matmul_B_times_alpha, int M, int N) {

  void *multiply_matrix_with_constant_0_0_14_args[] = {
      (void *)&A_matmul_B, (void *)&A_matmul_B_times_alpha, (void *)&M,
      (void *)&N};
  cudaLaunchKernel((void *)multiply_matrix_with_constant_0_0_14,
                   dim3(int_ceil(int_ceil(N, 1), 32),
                        int_ceil(int_ceil(M, 1), 1), int_ceil(1, 1)),
                   dim3(32, 1, 1), multiply_matrix_with_constant_0_0_14_args, 0,
                   __state->gpu_context->streams[0]);
}
__global__ void
multiply_matrix_with_constant_0_0_11(double *__restrict__ C_times_beta,
                                     const double *__restrict__ gpu_C, int M,
                                     int N) {
  constexpr double alpha = 1;
  constexpr double beta = 1;
  constexpr long long size_thread_block_tile_m = 128;
  constexpr long long size_thread_block_tile_n = 64;
  constexpr long long size_K_tile = 8;
  constexpr long long num_thread_blocks_m = 8;
  constexpr long long num_thread_blocks_n = 16;
  constexpr long long num_K_tiles = 128;
  constexpr long long size_warp_tile_m = 64;
  constexpr long long size_warp_tile_n = 32;
  constexpr long long size_thread_tile_m = 8;
  constexpr long long size_thread_tile_n = 8;
  constexpr long long SPLIT_K = 1;
  {
    {
      int j = (blockIdx.x * 32 + threadIdx.x);
      int i = (blockIdx.y * 1 + threadIdx.y);
      if (j < N) {
        {
          {
            double __in = gpu_C[((N * i) + j)];
            double __out;

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

DACE_EXPORTED void __dace_runkernel_multiply_matrix_with_constant_0_0_11(
    gemm_t *__state, double *__restrict__ C_times_beta,
    const double *__restrict__ gpu_C, int M, int N);
void __dace_runkernel_multiply_matrix_with_constant_0_0_11(
    gemm_t *__state, double *__restrict__ C_times_beta,
    const double *__restrict__ gpu_C, int M, int N) {

  void *multiply_matrix_with_constant_0_0_11_args[] = {
      (void *)&C_times_beta, (void *)&gpu_C, (void *)&M, (void *)&N};
  cudaLaunchKernel((void *)multiply_matrix_with_constant_0_0_11,
                   dim3(int_ceil(int_ceil(N, 1), 32),
                        int_ceil(int_ceil(M, 1), 1), int_ceil(1, 1)),
                   dim3(32, 1, 1), multiply_matrix_with_constant_0_0_11_args, 0,
                   __state->gpu_context->streams[2]);
}
__global__ void
add_matrices_0_0_17(const double *__restrict__ A_matmul_B_times_alpha,
                    const double *__restrict__ C_times_beta,
                    double *__restrict__ gpu_result, int M, int N) {
  constexpr double alpha = 1;
  constexpr double beta = 1;
  constexpr long long size_thread_block_tile_m = 128;
  constexpr long long size_thread_block_tile_n = 64;
  constexpr long long size_K_tile = 8;
  constexpr long long num_thread_blocks_m = 8;
  constexpr long long num_thread_blocks_n = 16;
  constexpr long long num_K_tiles = 128;
  constexpr long long size_warp_tile_m = 64;
  constexpr long long size_warp_tile_n = 32;
  constexpr long long size_thread_tile_m = 8;
  constexpr long long size_thread_tile_n = 8;
  constexpr long long SPLIT_K = 1;
  {
    {
      int j = (blockIdx.x * 32 + threadIdx.x);
      int i = (blockIdx.y * 1 + threadIdx.y);
      if (j < N) {
        {
          {
            double __in1 = A_matmul_B_times_alpha[((N * i) + j)];
            double __in2 = C_times_beta[((N * i) + j)];
            double __out;

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

DACE_EXPORTED void __dace_runkernel_add_matrices_0_0_17(
    gemm_t *__state, const double *__restrict__ A_matmul_B_times_alpha,
    const double *__restrict__ C_times_beta, double *__restrict__ gpu_result,
    int M, int N);
void __dace_runkernel_add_matrices_0_0_17(
    gemm_t *__state, const double *__restrict__ A_matmul_B_times_alpha,
    const double *__restrict__ C_times_beta, double *__restrict__ gpu_result,
    int M, int N) {

  void *add_matrices_0_0_17_args[] = {
      (void *)&A_matmul_B_times_alpha, (void *)&C_times_beta,
      (void *)&gpu_result, (void *)&M, (void *)&N};
  cudaLaunchKernel((void *)add_matrices_0_0_17,
                   dim3(int_ceil(int_ceil(N, 1), 32),
                        int_ceil(int_ceil(M, 1), 1), int_ceil(1, 1)),
                   dim3(32, 1, 1), add_matrices_0_0_17_args, 0,
                   __state->gpu_context->streams[0]);
}
