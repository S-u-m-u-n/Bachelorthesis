
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


#define TYPE float
#define VECTORTYPE2 float2
#define VECTORTYPE4 float4
// #define M 4096
// #define N 4096
// #define K 4096
#define THREADBLOCK_TILE_M 128
#define THREADBLOCK_TILE_N 128
#define THREADBLOCK_TILE_K 4096
#define LOAD_K 8
#define WARP_TILE_M 32
#define WARP_TILE_N 64
#define THREAD_TILE_M 8
#define THREAD_TILE_N 8
#define A_OFFSET 0
#define B_OFFSET 0
// #define SWIZZLE 1
// #define SPLIT_K 1
#define ATOMIC_REDUCTION false
#define ADDITIONAL_OCCUPANCY_SM 2
#define ALPHA 1
#define BETA 0


/**
 * This function loads the values of A from shared memory into registers.
 *
 * @param A_Shared			The shared memory to store the tile, column major
 * @param A_register		Registers to store A
 * @param k					Current k index to load
 * @param WarpIdy			The WarpId in the y dimension of the current thread
 * @param LaneIdy			The LaneId in the y dimension of the current thread
 * @param A_Shared_Offset	Offset used to access A_Shared due to double buffering
 */
 __device__ __inline__ void load_A_Shared(
    const TYPE (* __restrict__ A_Shared)[2 * (THREADBLOCK_TILE_M + A_OFFSET)
            * LOAD_K],
    TYPE (* __restrict__ A_register)[ THREAD_TILE_M], const int k,
    const int WarpIdy, const int LaneIdy, const int A_Shared_Offset) {

constexpr int TIMES = THREAD_TILE_M / 4;

constexpr int M_THREADS = WARP_TILE_M / THREAD_TILE_M;

const int Shared_j = k;

// We use as many float4 loads as we can
#pragma unroll
for (int i = 0; i < TIMES; i++) {

    const int Shared_i = WarpIdy * WARP_TILE_M + i * M_THREADS * 4
            + LaneIdy * 4;

    const TYPE* shared_mem_pointer = &(*A_Shared)[A_Shared_Offset + Shared_i
            + (THREADBLOCK_TILE_M + A_OFFSET) * Shared_j];

    const VECTORTYPE4 a =
            reinterpret_cast<const VECTORTYPE4*>(shared_mem_pointer)[0];

    TYPE* register_ptr = &(*A_register)[i * 4];

    reinterpret_cast<VECTORTYPE4*>(register_ptr)[0] = a;

}

// If there is a rest greater equal 2, we can use one more float 2 load
if (THREAD_TILE_M % 4 >= 2) {

    const int Shared_i = WarpIdy * WARP_TILE_M + TIMES * M_THREADS * 4
            + LaneIdy * 2;

    const TYPE* shared_mem_pointer = &(*A_Shared)[A_Shared_Offset + Shared_i
            + (THREADBLOCK_TILE_M + A_OFFSET) * Shared_j];

    const VECTORTYPE2 a =
            reinterpret_cast<const VECTORTYPE2*>(shared_mem_pointer)[0];

    TYPE* register_ptr = &(*A_register)[TIMES * 4];

    reinterpret_cast<VECTORTYPE2*>(register_ptr)[0] = a;

}

// And use one single load in the end, if there is still some rest
if (THREAD_TILE_M % 2 > 0) {

    constexpr int ADDITIONAL_OFFSET_SHARED =
            (THREAD_TILE_M % 4 >= 2) ? M_THREADS * 2 : 0;

    constexpr int ADDITIONAL_OFFSET_REGISTER =
            (THREAD_TILE_M % 4 >= 2) ? 2 : 0;

    const int Shared_i = WarpIdy * WARP_TILE_M + TIMES * M_THREADS * 4
            + LaneIdy + ADDITIONAL_OFFSET_SHARED;

    (*A_register)[TIMES * 4 + ADDITIONAL_OFFSET_REGISTER] =
            (*A_Shared)[A_Shared_Offset + Shared_i
                    + (THREADBLOCK_TILE_M + A_OFFSET) * Shared_j];

}

}

/**
* This function loads the values of B from shared memory into registers.
*
* @param B_Shared			The shared memory to store the tile, row major
* @param B_register		Registers to store B
* @param k					Current k index to load
* @param WarpIdx			The WarpId in the x dimension of the current thread
* @param LaneIdx			The LaneId in the x dimension of the current thread
* @param B_Shared_Offset 	Offset used to access B_Shared due to double buffering
*/
__device__ __inline__ void load_B_Shared(
TYPE (* __restrict__ B_Shared)[2 * LOAD_K * (THREADBLOCK_TILE_N + B_OFFSET)],
TYPE (* __restrict__ B_register)[ THREAD_TILE_N], const int k,
    const int WarpIdx, const int LaneIdx, const int B_Shared_Offset) {

constexpr int N_THREADS = WARP_TILE_N / THREAD_TILE_N;

constexpr int TIMES = THREAD_TILE_N / 4;

const int Shared_i = k;

// We use as many float4 loads as we can
#pragma unroll
for (int i = 0; i < TIMES; i++) {

    const int Shared_j = WarpIdx * WARP_TILE_N + LaneIdx * 4
            + i * N_THREADS * 4;

    const TYPE* shared_mem_pointer = &(*B_Shared)[B_Shared_Offset
            + Shared_i * (THREADBLOCK_TILE_N + B_OFFSET) + Shared_j];

    const VECTORTYPE4 a =
            reinterpret_cast<const VECTORTYPE4*>(shared_mem_pointer)[0];

    TYPE* register_ptr = &(*B_register)[i * 4];

    reinterpret_cast<VECTORTYPE4*>(register_ptr)[0] = a;

}

// If there is a rest greater equal 2, we can use one more float 2 load
if (THREAD_TILE_N % 4 >= 2) {

    const int Shared_j = WarpIdx * WARP_TILE_N + LaneIdx * 2
            + TIMES * N_THREADS * 4;

    const TYPE* shared_mem_pointer = &(*B_Shared)[B_Shared_Offset
            + Shared_i * (THREADBLOCK_TILE_N + B_OFFSET) + Shared_j];

    const VECTORTYPE2 a =
            reinterpret_cast<const VECTORTYPE2*>(shared_mem_pointer)[0];

    TYPE* register_ptr = &(*B_register)[TIMES * 4];

    reinterpret_cast<VECTORTYPE2*>(register_ptr)[0] = a;

}

// And use one single load in the end, if there is still some rest
if (THREAD_TILE_N % 2 > 0) {

    constexpr int ADDITIONAL_OFFSET_SHARED =
            (THREAD_TILE_N % 4 >= 2) ? N_THREADS * 2 : 0;

    constexpr int ADDITIONAL_OFFSET_REGISTER =
            (THREAD_TILE_N % 4 >= 2) ? 2 : 0;

    const int Shared_j = WarpIdx * WARP_TILE_N + LaneIdx
            + TIMES * N_THREADS * 4 + ADDITIONAL_OFFSET_SHARED;

    (*B_register)[TIMES * 4 + ADDITIONAL_OFFSET_REGISTER] =
            (*B_Shared)[B_Shared_Offset
                    + Shared_i * (THREADBLOCK_TILE_N + B_OFFSET) + Shared_j];

}

}

/**
*
* This function loads the values of A and B from shared memory into registers.
*
* @param A_Shared			The shared memory to store the tile, column major
* @param A_register		Registers to store A
* @param B_Shared			The shared memory to store the tile, row major
* @param B_register		Registers to store B
* @param k					Current k index to load
* @param WarpIdx			The WarpId in the x dimension of the current thread
* @param WarpIdy			The WarpId in the y dimension of the current thread
* @param LaneIdx			The LaneId in the x dimension of the current thread
* @param LaneIdy			The LaneId in the y dimension of the current thread
* @param A_Shared_Offset 	Offset used to access A_Shared due to double buffering
* @param B_Shared_Offset 	Offset used to access B_Shared due to double buffering
*/
__device__ __inline__ void load_Shared(
TYPE (* __restrict__ A_Shared)[2 * (THREADBLOCK_TILE_M + A_OFFSET) * LOAD_K],
TYPE (* __restrict__ A_register)[THREAD_TILE_M],
TYPE (* __restrict__ B_Shared)[2 * LOAD_K * (THREADBLOCK_TILE_N + B_OFFSET)],
TYPE (* __restrict__ B_register)[THREAD_TILE_N], const int k, const int WarpIdx,
    const int WarpIdy, const int LaneIdx, const int LaneIdy,
    const int A_Shared_Offset, const int B_Shared_Offset) {

load_A_Shared(A_Shared, A_register, k, WarpIdy, LaneIdy, A_Shared_Offset);

load_B_Shared(B_Shared, B_register, k, WarpIdx, LaneIdx, B_Shared_Offset);
}


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


    constexpr int M_WARPS = THREADBLOCK_TILE_M / WARP_TILE_M;
	constexpr int N_WARPS = THREADBLOCK_TILE_N / WARP_TILE_N;

	constexpr int N_THREADS = WARP_TILE_N / THREAD_TILE_N;
	constexpr int M_THREADS = WARP_TILE_M / THREAD_TILE_M;

    const int WarpId = threadIdx.x / 32;
	const int threadId = threadIdx.x % 32;

	const int WarpIdx = WarpId % N_WARPS;
	const int WarpIdy = WarpId / N_WARPS;

	int LaneIdx;
	int LaneIdy;

	if (N_THREADS == 1) {

		LaneIdx = 0;
		LaneIdy = threadId;

	} else if (N_THREADS == 2) {

		LaneIdx = (((threadId & 0x60) >> 4) | (threadId & 1));
		LaneIdy = ((threadId >> 1) & (M_THREADS - 1));

	} else if (N_THREADS == 4) {

		LaneIdx = (((threadId & 0x30) >> 3) | (threadId & 1));
		LaneIdy = ((threadId >> 1) & (M_THREADS - 1));

	} else if (N_THREADS == 8) {

		LaneIdx = (((threadId & 0x18) >> 2) | (threadId & 1));
		LaneIdy = ((threadId >> 1) & (M_THREADS - 1));

	} else if (N_THREADS == 16) {

		LaneIdx = (((threadId & 0x1c) >> 1) | (threadId & 1));
		LaneIdy = ((threadId >> 1) & (M_THREADS - 1));

	} else if (N_THREADS == 32) {

		LaneIdx = threadId;
		LaneIdy = 0;
	}
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

