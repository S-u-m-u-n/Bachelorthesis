# Four different approaches to matrix multiplication
1. Cuda - vanilla
2. Cuda - shared memory
3. DaCe - vanilla
4. DaCe - shared memory


## Cuda
### To compile the .cu files:
`nvcc -arch=sm_70 mm_vanilla.cu -o mm_vanilla`

`nvcc -arch=sm_70 mm_shared_memory.cu -o mm_shared_memory`

### For profiling:
`nvprof -f -o mm_vanilla.nvprof ./mm_vanilla`

`nvprof -f -o mm_shared_memory.nvprof ./mm_shared_memory`

## DaCe
> To be added
