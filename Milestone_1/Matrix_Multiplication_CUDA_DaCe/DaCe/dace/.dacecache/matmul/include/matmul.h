typedef void * matmulHandle_t;
extern "C" matmulHandle_t __dace_init_matmul(int K, int M, int N);
extern "C" void __dace_exit_matmul(matmulHandle_t handle);
extern "C" void __program_matmul(matmulHandle_t handle, double * __restrict__ A, double * __restrict__ B, double * __restrict__ __return, int K, int M, int N);
