from argparse import ArgumentParser
import numpy as np
import sympy as sy
import dace
import math
from Schedule import Schedule
import helpers
from dace.transformation.dataflow import Vectorization, DoubleBuffering, MapToForLoop

parser = ArgumentParser()
parser.add_argument("-v", "--verbose",
                    dest='verbose',
                    help="Explain what is being done. Default: False",
                    action="store_true",
                    default=False)
parser.add_argument("-q", "--quiet",
                    dest='quiet',
                    help="Only print errors.",
                    action="store_true",
                    default=False)
parser.add_argument("-c", "--colorless",
                    dest='colorless',
                    help="Does not print colors, useful when writing to a file.",
                    action="store_true",
                    default=False)
parser.add_argument("-g", "--gpu-type",
                    dest='gpu_type',
                    help="use this to specify the gpu type (\"NVIDIA\" or \"AMD\" or \"default\" (skips query)). Default: default",
                    action="store",
                    default="default")
parser.add_argument("-M", type=int, dest='M', nargs="?", default=640)
parser.add_argument("-K", type=int, dest='K', nargs="?", default=640)
parser.add_argument("-N", type=int, dest='N', nargs="?", default=640)
parser.add_argument("--alpha", type=np.float64, dest='alpha', nargs="?", default=1.0)
parser.add_argument("--beta", type=np.float64, dest='beta', nargs="?", default=0.0)
parser.add_argument("-r", "--repetitions", type=int, dest='repetitions', nargs="?", default=1)
parser.add_argument('-p', '--precision',
                    type=int,
                    dest='precision',
                    choices=[32, 64],
                    default=64,
                    help="Specify floating precision (32 or 64)")
parser.add_argument('--verify',
                    dest='verify',
                    help="Verify results. Default: False",
                    action="store_true",
                    default=False)
parser.add_argument('--all-optimizations',
                    dest='all_optimizations',
                    help="Use all possible optimizations",
                    action="store_true",
                    default=False)
parser.add_argument('--split-k', type=int, dest='split_k', nargs="?", default=1)
parser.add_argument('--accumulator-split-k', type=int, dest='accumulator_split_k', default=2)
parser.add_argument('--swizzle-thread-blocks', type=int, dest='swizzle_thread_blocks', nargs="?", default=1)
parser.add_argument('--swizzle-threads',
                    dest='swizzle_threads',
                    help="Swizzle the threads",
                    action="store_true",
                    default=False)
parser.add_argument('--vectorization',
                    dest='vectorization',
                    help="Use vectorization",
                    action="store_true",
                    default=False)
parser.add_argument('--double-buffering-shared',
                    dest='double_buffering_shared',
                    help="Use double buffering on the shared memory",
                    action="store_true",
                    default=False)
parser.add_argument('--double-buffering-register',
                    dest='double_buffering_register',
                    help="Use double buffering on the register",
                    action="store_true",
                    default=False)
parser.add_argument('--reverse-k',
                    dest='reverse_k',
                    help="Reverse the outer K loop",
                    action="store_true",
                    default=False)
parser.add_argument('--nested-pointer-offset',
                    dest='nested_pointer_offset',
                    help="Calculate the offset for pointers passed to nested (device) functions inside of the nested function instead of the function calling the nested function",
                    action="store_true",
                    default=False)
parser.add_argument('--shared-A-column',
                    dest='shared_A_column_major',
                    help="Store the shared memory of A in a column major order instead of row major",
                    action="store_true",
                    default=False)
parser.add_argument('--name',
                    dest='name',
                    type=str,
                    help="Add a suffix name to the SDFG and generated library/program. Useful for running multiple instances in parallel without interfering",
                    default="")
                    

args = parser.parse_args()
if args.verbose:
    helpers.print_info("Program launched with the following arguments: " + str(args), args.colorless)

if args.precision == 32:
    dtype = dace.float32
    ndtype = np.float32
    veclen = 4
    args.alpha = ndtype(args.alpha)
    args.beta = ndtype(args.beta)

    if args.M == 1024 and args.N == 1024 and args.K == 1024:
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, warp_tile_m=32, warp_tile_n=64, thread_block_tile_m=128, thread_block_tile_n=128) # cuCOSMA's schedule
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # (267us)
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=16, warp_tile_n=16, thread_block_tile_m=32, thread_block_tile_n=128) # (261us, wrong)
        schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 265us, correct
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=64, thread_block_tile_n=128) # 273us, correct
        # schedule = Schedule(load_k=8, thread_tile_m=4, thread_tile_n=4, warp_tile_m=32, warp_tile_n=16, thread_block_tile_m=128, thread_block_tile_n=64) # 
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 
    elif args.M == 4096 and args.N == 4096 and args.K == 4096:
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, warp_tile_m=32, warp_tile_n=64, thread_block_tile_m=128, thread_block_tile_n=128) # cuCOSMA's schedule
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=16, warp_tile_n=16, thread_block_tile_m=32, thread_block_tile_n=128) # (13.25ms, wrong)
        schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) #  13.35ms, correct
    elif args.M == 1024 and args.N == 1024 and args.K == 8192:
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, warp_tile_m=32, warp_tile_n=64, thread_block_tile_m=128, thread_block_tile_n=128) # cuCOSMA's schedule
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=64, thread_block_tile_n=64) # Only uses 16 threads/warp, still fast... wtf?
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=16, warp_tile_n=16, thread_block_tile_m=32, thread_block_tile_n=128) # Fastest, wrong
        schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 1.80ms, correct
    elif args.M == 256 and args.N == 256 and args.K == 10240:
        schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=2, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128)
    else:
        schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, warp_tile_m=32, warp_tile_n=64, thread_block_tile_m=128, thread_block_tile_n=128)

    # More similar to cucosma:
    # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, warp_tile_m=32, warp_tile_n=64, thread_block_tile_m=128, thread_block_tile_n=128)

    # schedule = Schedule(load_k=4, thread_tile_m=4, thread_tile_n=4, warp_tile_m=32, warp_tile_n=16, thread_block_tile_m=64, thread_block_tile_n=64)
    # schedule = Schedule(load_k=4, thread_tile_m=8, thread_tile_n=8, warp_tile_m=32, warp_tile_n=64, thread_block_tile_m=128, thread_block_tile_n=128)

    # schedule = Schedule(load_k=4, thread_tile_m=4, thread_tile_n=4, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128)
    # schedule = Schedule(load_k=4, thread_tile_m=4, thread_tile_n=4, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=64)
    # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, warp_tile_m=64, warp_tile_n=32, thread_block_tile_m=128, thread_block_tile_n=128)

else:
    dtype = dace.float64
    ndtype = np.float64
    veclen = 2
    # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, warp_tile_m=64, warp_tile_n=32, thread_block_tile_m=128, thread_block_tile_n=64)
    # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=2, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128)
    if args.M == 1024 and args.N == 1024 and args.K == 1024:
        schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=2, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 547us, correct
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 548us, correct
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=16, warp_tile_n=16, thread_block_tile_m=32, thread_block_tile_n=128) # 526us, wrong
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # correct but not fast?
    elif args.M == 4096 and args.N == 4096 and args.K == 4096:
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=2, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 25.28ms
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=16, warp_tile_n=16, thread_block_tile_m=32, thread_block_tile_n=128) # 24.455ms, wrong
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 27ms, correct
        schedule = Schedule(load_k=8, thread_tile_m=4, thread_tile_n=4, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 24.77, correct
        # schedule = Schedule(load_k=8, thread_tile_m=4, thread_tile_n=4, warp_tile_m=32, warp_tile_n=16, thread_block_tile_m=64, thread_block_tile_n=64) # 26.9ms
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 25.1ms
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=64, thread_block_tile_n=128) # 24.85ms
        # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=4, warp_tile_m=32, warp_tile_n=32, thread_block_tile_m=128, thread_block_tile_n=64) # 26.7ms
    elif args.M == 1024 and args.N == 1024 and args.K == 8192:
        schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=2, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 3.65ms, correct
    elif args.M == 256 and args.N == 256 and args.K == 10240:
        schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=2, warp_tile_m=16, warp_tile_n=32, thread_block_tile_m=32, thread_block_tile_n=128) # 290us
    else:
        schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, warp_tile_m=32, warp_tile_n=64, thread_block_tile_m=128, thread_block_tile_n=128)

    # schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, warp_tile_m=32, warp_tile_n=64, thread_block_tile_m=128, thread_block_tile_n=64)
    # schedule = Schedule(load_k=16, thread_tile_m=4, thread_tile_n=4, warp_tile_m=32, warp_tile_n=16, thread_block_tile_m=64, thread_block_tile_n=32)
    # schedule = Schedule(load_k=4, thread_tile_m=4, thread_tile_n=4, warp_tile_m=32, warp_tile_n=16, thread_block_tile_m=128, thread_block_tile_n=64)
    # schedule = Schedule(load_k=8, thread_tile_m=2, thread_tile_n=2, warp_tile_m=16, warp_tile_n=8, thread_block_tile_m=128, thread_block_tile_n=64)
    # schedule = Schedule(load_k=8, thread_tile_m=1, thread_tile_n=1, warp_tile_m=8, warp_tile_n=4, thread_block_tile_m=128, thread_block_tile_n=64)
    # schedule = Schedule(load_k=4, thread_tile_m=8, thread_tile_n=8, warp_tile_m=64, warp_tile_n=32, thread_block_tile_m=128, thread_block_tile_n=64)
    # schedule = Schedule(load_k=2, thread_tile_m=8, thread_tile_n=8, warp_tile_m=64, warp_tile_n=32, thread_block_tile_m=128, thread_block_tile_n=64)
    # schedule = Schedule(load_k=1, thread_tile_m=8, thread_tile_n=8, warp_tile_m=64, warp_tile_n=32, thread_block_tile_m=128, thread_block_tile_n=64)
    # schedule = Schedule(load_k=1, thread_tile_m=1, thread_tile_n=1, warp_tile_m=8, warp_tile_n=4, thread_block_tile_m=16, thread_block_tile_n=8)

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')

sdfg = dace.SDFG('gemm' + str(args.name))
state = sdfg.add_state(label='gemm_state' + str(args.name))
nested_sdfg = dace.SDFG('nested_gemm' + str(args.name))
sdfg.add_constant('VECLEN', veclen)
nested_sdfg.add_constant('VECLEN', veclen)

# if args.vectorization:
    # sdfg.add_array('A', shape=[M, K // 2], dtype=dace.vector(dace.float64, 2))
    # sdfg.add_array('B', shape=[K, N // 2], dtype=dace.vector(dace.float64, 2))
# else:
sdfg.add_array('A', shape=[M, K], dtype=dtype)
sdfg.add_array('B', shape=[K, N], dtype=dtype)
sdfg.add_array('C', shape=[M, N], dtype=dtype)
A_in = state.add_read('A')
B_in = state.add_read('B')
C_out = state.add_write('C')

# if args.vectorization:
    # sdfg.add_transient('gpu_A', shape=[M, K // 2], dtype=dace.vector(dace.float64, 2), storage=dace.StorageType.GPU_Global)
    # sdfg.add_transient('gpu_B', shape=[K, N // 2], dtype=dace.vector(dace.float64, 2), storage=dace.StorageType.GPU_Global)
# else:
sdfg.add_transient('gpu_A', shape=[M, K], dtype=dtype, storage=dace.StorageType.GPU_Global)
sdfg.add_transient('gpu_B', shape=[K, N], dtype=dtype, storage=dace.StorageType.GPU_Global)

sdfg.add_transient('gpu_C', shape=[M, N], dtype=dtype, storage=dace.StorageType.GPU_Global)
sdfg.add_transient('gpu_result', shape=[M, N], dtype=dtype, storage=dace.StorageType.GPU_Global)

gpu_A = state.add_access('gpu_A')
gpu_B = state.add_access('gpu_B')
gpu_result = state.add_access('gpu_result')

sdfg.add_constant('alpha', ndtype(args.alpha))
sdfg.add_constant('beta', ndtype(args.beta))
# sdfg.add_constant('alpha', args.alpha)
# sdfg.add_constant('beta', args.beta)

sdfg.add_transient('A_matmul_B', shape=[M, N], dtype=dtype, storage=dace.StorageType.GPU_Global, lifetime=dace.AllocationLifetime.SDFG)

A_matmul_B = state.add_write('A_matmul_B')

#########################################################
# Connect arrays to GPU transient and the GPU result transient to host array
# if args.vectorization:
    # state.add_edge(A_in, None, gpu_A, None, memlet=dace.Memlet.simple(A_in.data, '0:M, 0:K//2'))
    # state.add_edge(B_in, None, gpu_B, None, memlet=dace.Memlet.simple(B_in.data, '0:K, 0:N//2'))
# else:
state.add_edge(A_in, None, gpu_A, None, memlet=dace.Memlet.simple(A_in.data, '0:M, 0:K'))
state.add_edge(B_in, None, gpu_B, None, memlet=dace.Memlet.simple(B_in.data, '0:K, 0:N'))

state.add_edge(gpu_result, None, C_out, None, memlet=dace.Memlet.simple(gpu_result.data, '0:M, 0:N'))

final_C = None
final_A_matmul_B = A_matmul_B

#########################################################
# Multiply the result of (A @ B) with alpha, only if alpha != 1
if(math.fabs(args.alpha - 1.0) > 1e-6):
    sdfg.add_transient('A_matmul_B_times_alpha', shape=[M, N], dtype=dtype, storage=dace.StorageType.GPU_Global, lifetime=dace.AllocationLifetime.SDFG)
    A_matmul_B_times_alpha = state.add_write('A_matmul_B_times_alpha')

    map_entry, map_exit = state.add_map(
            'multiply_matrix_with_constant',
            dict(i='0:M', j='0:N'),
            schedule=dace.dtypes.ScheduleType.GPU_Device)

    tasklet = state.add_tasklet('multiply_matrix_with_constant', ['__in'], ['__out'], '__out = (alpha * __in)')

    state.add_memlet_path(A_matmul_B,
                            map_entry,
                            tasklet,
                            dst_conn='__in',
                            memlet=dace.Memlet(f"{A_matmul_B.data}[i, j]"))

    state.add_memlet_path(tasklet,
                            map_exit,
                            A_matmul_B_times_alpha,
                            src_conn='__out',
                            memlet=dace.Memlet(f"{A_matmul_B_times_alpha.data}[i, j]"))

    final_A_matmul_B = A_matmul_B_times_alpha


#########################################################
# Multiply C with beta, only if beta != 0
if(math.fabs(args.beta) > 1e-6):
    C_in = state.add_read('C')
    gpu_C = state.add_access('gpu_C')

    state.add_edge(C_in, None, gpu_C, None, memlet=dace.Memlet.simple(C_in.data, '0:M, 0:N'))

    sdfg.add_transient('C_times_beta', shape=[M, N], dtype=dtype, storage=dace.StorageType.GPU_Global, lifetime=dace.AllocationLifetime.SDFG)
    C_times_beta = state.add_write('C_times_beta')

    map_entry, map_exit = state.add_map(
            'multiply_matrix_with_constant',
            dict(i='0:M', j='0:N'),
            schedule=dace.dtypes.ScheduleType.GPU_Device)

    tasklet = state.add_tasklet('multiply_matrix_with_constant', ['__in'], ['__out'], '__out = (beta * __in)')

    state.add_memlet_path(gpu_C,
                            map_entry,
                            tasklet,
                            dst_conn='__in',
                            memlet=dace.Memlet(f"{gpu_C.data}[i, j]"))

    state.add_memlet_path(tasklet,
                            map_exit,
                            C_times_beta,
                            src_conn='__out',
                            memlet=dace.Memlet(f"{C_times_beta.data}[i, j]"))

    #########################################################
    # Add the result of (A @ B) * alpha and C * beta, also only if beta != 0
    map_entry, map_exit = state.add_map(
            'add_matrices',
            dict(i='0:M', j='0:N'),
            schedule=dace.dtypes.ScheduleType.GPU_Device)

    tasklet = state.add_tasklet('add_matrices', ['__in1', '__in2'], ['__out'], '__out = (__in1 + __in2)')

    state.add_memlet_path(final_A_matmul_B,
                            map_entry,
                            tasklet,
                            dst_conn='__in1',
                            memlet=dace.Memlet(f"{final_A_matmul_B.data}[i, j]"))

    state.add_memlet_path(C_times_beta,
                            map_entry,
                            tasklet,
                            dst_conn='__in2',
                            memlet=dace.Memlet(f"{C_times_beta.data}[i, j]"))

    state.add_memlet_path(tasklet,
                            map_exit,
                            gpu_result,
                            src_conn='__out',
                            memlet=dace.Memlet(f"{gpu_result.data}[i, j]"))

# Directly store the result to the output
else:
    state.add_memlet_path(final_A_matmul_B,
                        gpu_result,
                        memlet=dace.Memlet(gpu_result.data))

#########################################################
# Create nested sdfg
nested_sdfg_node = state.add_nested_sdfg(nested_sdfg, state, {'input_A', 'input_B'}, {'output'}, schedule=dace.ScheduleType.GPU_Default, symbol_mapping={"K": "K", "M": "M", "N": "N"})

# if args.vectorization:
    # state.add_edge(gpu_A, None, nested_sdfg_node, 'input_A', memlet=dace.Memlet.simple(gpu_A.data, '0:M, 0:K//2'))
    # state.add_edge(gpu_B, None, nested_sdfg_node, 'input_B', memlet=dace.Memlet.simple(gpu_B.data, '0:K, 0:N//2'))
# else:
state.add_edge(gpu_A, None, nested_sdfg_node, 'input_A', memlet=dace.Memlet.simple(gpu_A.data, '0:M, 0:K'))
state.add_edge(gpu_B, None, nested_sdfg_node, 'input_B', memlet=dace.Memlet.simple(gpu_B.data, '0:K, 0:N'))

state.add_edge(nested_sdfg_node, 'output', A_matmul_B, None, memlet=dace.Memlet.simple(A_matmul_B.data, '0:M, 0:N'))

nested_initstate = nested_sdfg.add_state(label='nested_initstate')
nested_initstate.executions = 1
nested_initstate.dynamic_executions = False
nested_state = nested_sdfg.add_state(label='nested_state')
nested_state.executions = 1
nested_state.dynamic_executions = False

nested_sdfg.add_edge(nested_initstate, nested_state, dace.InterstateEdge()) # connect the two states

for e in state.in_edges(nested_sdfg_node):
    if e.dst_conn == "input_A":
        desc_a = sdfg.arrays[e.data.data]
    elif e.dst_conn == "input_B":
        desc_b = sdfg.arrays[e.data.data]

for e in state.out_edges(nested_sdfg_node):
    if e.src_conn == "output":
        desc_res = sdfg.arrays[e.data.data]

desc_a = desc_a.clone()
desc_a.transient = False
desc_a.storage = dace.StorageType.Default
desc_b = desc_b.clone()
desc_b.transient = False
desc_b.storage = dace.StorageType.Default
desc_res = desc_res.clone()
desc_res.transient = False
desc_res.storage = dace.StorageType.Default
desc_res.lifetime = dace.AllocationLifetime.Scope

input_A_nested = nested_sdfg.add_datadesc('input_A', desc_a)
input_B_nested = nested_sdfg.add_datadesc('input_B', desc_b)
output_nested = nested_sdfg.add_datadesc('output', desc_res)

_A = nested_state.add_read(input_A_nested)
_B = nested_state.add_read(input_B_nested)
# A_matmul_B_nested_initstate = nested_initstate.add_write(output)
A_matmul_B_nested_initstate = nested_initstate.add_access(output_nested)
A_matmul_B_nested_state = nested_state.add_write(output_nested)
# A_matmul_B_nested_state = nested_state.add_access(output)

#########################################################
### matmul init state
map_entry, map_exit = nested_initstate.add_map(
        'initialize_matmul_result',
        dict(i='0:M', j='0:N'),
        schedule=dace.dtypes.ScheduleType.Default)

tasklet = nested_initstate.add_tasklet('matmul_init', [], ['out'], 'out = 0')

nested_initstate.add_memlet_path(map_entry,
            tasklet,
            memlet=dace.Memlet())

nested_initstate.add_memlet_path(tasklet,
            map_exit,
            A_matmul_B_nested_initstate,
            src_conn='out',
            memlet=dace.Memlet(f"{A_matmul_B_nested_initstate.data}[i, j]"))

#########################################################
### matmul state

# if args.vectorization:
#     nested_sdfg.add_transient('shared_memory_A', shape=[schedule.thread_block_tile_m, schedule.load_k // veclen], dtype=dace.vector(dtype, veclen), storage=dace.StorageType.GPU_Shared)
#     nested_sdfg.add_transient('shared_memory_B', shape=[schedule.load_k, schedule.thread_block_tile_n // veclen], dtype=dace.vector(dtype, veclen), storage=dace.StorageType.GPU_Shared)
# else:
# nested_sdfg.add_transient('shared_memory_A', shape=[schedule.thread_block_tile_m, schedule.load_k], dtype=dtype, storage=dace.StorageType.GPU_Shared)
if args.shared_A_column_major:
    # nested_sdfg.add_transient('shared_memory_A', shape=[schedule.load_k, schedule.thread_block_tile_m], dtype=dtype, storage=dace.StorageType.GPU_Shared, strides=[1, schedule.thread_block_tile_m]) # Note: we store the shared memory A in a column-major fashion
    # nested_sdfg.add_transient('shared_memory_A', shape=[schedule.load_k, schedule.thread_block_tile_m], dtype=dtype, storage=dace.StorageType.GPU_Shared) # Note: we store the shared memory A in a column-major fashion
    nested_sdfg.add_transient('shared_memory_A', shape=[schedule.thread_block_tile_m, schedule.load_k], dtype=dtype, storage=dace.StorageType.GPU_Shared, strides=[1, schedule.thread_block_tile_m]) # Note: we store the shared memory A in a column-major fashion
else:
    nested_sdfg.add_transient('shared_memory_A', shape=[schedule.thread_block_tile_m, schedule.load_k], dtype=dtype, storage=dace.StorageType.GPU_Shared)

nested_sdfg.add_transient('shared_memory_B', shape=[schedule.load_k, schedule.thread_block_tile_n], dtype=dtype, storage=dace.StorageType.GPU_Shared)

nested_sdfg.add_transient('register_storage_A', shape=[schedule.thread_tile_m], dtype=dtype, storage=dace.StorageType.Register)
nested_sdfg.add_transient('register_storage_B', shape=[schedule.thread_tile_n], dtype=dtype, storage=dace.StorageType.Register)
nested_sdfg.add_transient('register_storage_C', shape=[schedule.thread_tile_m, schedule.thread_tile_n], dtype=dtype, storage=dace.StorageType.Register)

if args.split_k > 1:
    # helpers.print_info("Applying Split K...")
    nested_sdfg.add_transient('partial_split_k_output', shape=[args.split_k, M, N], dtype=dtype, storage=dace.StorageType.GPU_Global)
    partial_split_k_output = nested_state.add_access('partial_split_k_output')
    nested_sdfg.add_transient('accumulator', shape=[args.accumulator_split_k], dtype=dtype, storage=dace.StorageType.Register)
    accumulator = nested_state.add_access('accumulator')
    accumulator.setzero = True

shared_memory_A = nested_state.add_access('shared_memory_A')
shared_memory_B = nested_state.add_access('shared_memory_B')
register_storage_A = nested_state.add_access('register_storage_A')
register_storage_B = nested_state.add_access('register_storage_B')
register_storage_C = nested_state.add_access('register_storage_C')
register_storage_C.setzero = True


num_thread_blocks_n = args.swizzle_thread_blocks * int(args.N // schedule.thread_block_tile_n)
num_threads_per_thread_block = int((schedule.thread_block_tile_m // schedule.thread_tile_m) * (schedule.thread_block_tile_n // schedule.thread_tile_n))

sdfg.add_constant('size_thread_block_tile_m', schedule.thread_block_tile_m)
sdfg.add_constant('size_thread_block_tile_n', schedule.thread_block_tile_n)
sdfg.add_constant('size_K_tile', schedule.load_k)
# sdfg.add_constant('num_thread_blocks_m', int(args.M / schedule.thread_block_tile_m) if not args.swizzle_thread_blocks else int((args.M / schedule.thread_block_tile_m + 2 - 1) / 2))
sdfg.add_constant('num_thread_blocks_m', int((args.M // schedule.thread_block_tile_m + args.swizzle_thread_blocks - 1) // args.swizzle_thread_blocks))
# sdfg.add_constant('num_thread_blocks_n', int(args.N / schedule.thread_block_tile_n) if not args.swizzle_thread_blocks else 2*int(args.N / schedule.thread_block_tile_n))
sdfg.add_constant('num_thread_blocks_n', args.swizzle_thread_blocks * int(args.N // schedule.thread_block_tile_n))
sdfg.add_constant('num_K_tiles', int(args.K / (schedule.load_k * args.split_k)))
sdfg.add_constant('size_warp_tile_m', schedule.warp_tile_m)
sdfg.add_constant('size_warp_tile_n', schedule.warp_tile_n)
sdfg.add_constant('size_thread_tile_m', schedule.thread_tile_m)
sdfg.add_constant('size_thread_tile_n', schedule.thread_tile_n)
sdfg.add_constant('SPLIT_K', args.split_k)
# sdfg.add_constant('num_warps_n', num_thread_blocks_n / schedule.warp_tile_n)
sdfg.add_constant('ACCUMULATOR_SIZE', args.accumulator_split_k)


nested_sdfg.add_constant('size_thread_block_tile_m', schedule.thread_block_tile_m)
nested_sdfg.add_constant('size_thread_block_tile_n', schedule.thread_block_tile_n)
nested_sdfg.add_constant('num_thread_blocks_m', int((args.M // schedule.thread_block_tile_m + args.swizzle_thread_blocks - 1) // args.swizzle_thread_blocks))
nested_sdfg.add_constant('num_thread_blocks_n', num_thread_blocks_n)
nested_sdfg.add_constant('num_warps_n', int(schedule.thread_block_tile_n / schedule.warp_tile_n))
nested_sdfg.add_constant('num_K_tiles', int(args.K / (schedule.load_k * args.split_k)))
nested_sdfg.add_constant('size_warp_tile_m', schedule.warp_tile_m)
nested_sdfg.add_constant('size_warp_tile_n', schedule.warp_tile_n)
nested_sdfg.add_constant('size_thread_tile_m', schedule.thread_tile_m)
nested_sdfg.add_constant('size_thread_tile_n', schedule.thread_tile_n)
# nested_sdfg.add_constant('size_thread_tile_k', schedule.thread_tile_k) # = size_K_tile
nested_sdfg.add_constant('warp_width', math.ceil(schedule.warp_tile_n / schedule.thread_tile_n))
nested_sdfg.add_constant('warp_height', math.ceil(schedule.warp_tile_m / schedule.thread_tile_m))
nested_sdfg.add_constant('size_K_tile', schedule.load_k)
nested_sdfg.add_constant('size_K_split', args.K // args.split_k)
nested_sdfg.add_constant('SWIZZLE', args.swizzle_thread_blocks)
nested_sdfg.add_constant('SPLIT_K', args.split_k)
nested_sdfg.add_constant('ACCUMULATOR_SIZE', args.accumulator_split_k)

sdfg.add_constant('num_threads_per_thread_block', num_threads_per_thread_block)
nested_sdfg.add_constant('num_threads_per_thread_block', num_threads_per_thread_block)

tasklet = nested_state.add_tasklet('matrix_multiplication', {'__a', '__b'}, {'__out'}, '__out = (__a * __b)')


# This map creates threadblocks
thread_block_grid_map_entry, thread_block_grid_map_exit = nested_state.add_map(
        'Thread_block_grid',
        dict(thread_block_i='0:num_thread_blocks_m', thread_block_j='0:num_thread_blocks_n', thread_block_k='0:SPLIT_K') if args.split_k > 1 else
        dict(thread_block_i='0:num_thread_blocks_m', thread_block_j='0:num_thread_blocks_n'),
        schedule=dace.dtypes.ScheduleType.GPU_Device)

K_tile_map_entry, K_tile_map_exit = nested_state.add_map(
        'K_tile',
        dict(k_tile='0:num_K_tiles'),
        schedule=dace.dtypes.ScheduleType.Sequential)

# This map creates threads (and not warps!!)
thread_tile_map_entry, thread_tile_map_exit = nested_state.add_map(
        'Thread_tile',
        dict(thread='0:num_threads_per_thread_block'),
        schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock) # needs to be dace.dtypes.ScheduleType.GPU_ThreadBlock

thread_K_map_entry, thread_K_map_exit = nested_state.add_map(
        'Thread_K',
        dict(k='0:size_K_tile'),
        schedule=dace.dtypes.ScheduleType.Sequential)

thread_map_entry, thread_map_exit = nested_state.add_map(
        'Thread',
        dict(i='0:size_thread_tile_m', j='0:size_thread_tile_n'),
        unroll=True,
        schedule=dace.dtypes.ScheduleType.Sequential)


### Swizzle thread blocks subset
if args.swizzle_thread_blocks == 1:
    thread_block_i_offset = 'thread_block_i*size_thread_block_tile_m'
    thread_block_j_offset = 'thread_block_j*size_thread_block_tile_n'
else:
    helpers.print_info("Swizzling Thread Blocks...", False)
    thread_block_i_offset = '(thread_block_i*SWIZZLE + (thread_block_j % SWIZZLE))*size_thread_block_tile_m'
    thread_block_j_offset = '(thread_block_j // SWIZZLE)*size_thread_block_tile_n'

### Split K subset
if args.split_k == 1:
    k_tile_range = 'k_tile*size_K_tile:k_tile*size_K_tile+size_K_tile'
else:
    helpers.print_info("Splitting K...", False)
    k_tile_range = '(thread_block_k*size_K_split) + k_tile*size_K_tile:(thread_block_k*size_K_split) + k_tile*size_K_tile + size_K_tile'

### Vectorization subset
# if not args.vectorization:
# vec_adjust = ''
# else:
    # helpers.print_info("Applying Vectorization...", False)
    # vec_adjust = '// VECLEN'

warpId = '(thread // 32)'
threadId = '(thread % 32)'

warpIdx = '(' + warpId + ' % num_warps_n)' # right direction
warpIdy = '(' + warpId + ' // num_warps_n)' # down direction
warp_x_offset = '(' + warpIdx + ' * size_warp_tile_n)' # right direction
warp_y_offset = '(' + warpIdy + ' * size_warp_tile_m)' # down direction

### Swizzle threads subset
if not args.swizzle_threads:
    LaneIdx = '(' + threadId + ' % warp_width)' # right direction
    LaneIdy = '(' + threadId + ' // warp_width)' # down direction
else:
    helpers.print_info("Swizzling Threads...", False)
    bitwise_and = sy.Function('bitwise_and')
    bitwise_or = sy.Function('bitwise_or')
    right_shift = sy.Function('right_shift')

    # warp_x_offset = '(thread_j // size_warp_tile_n)*size_warp_tile_n'
    # warp_y_offset = '(thread_i // size_warp_tile_m)*size_warp_tile_m'

    # thread_i_idx = '(thread_i // size_warp_tile_m)*size_warp_tile_m + size_thread_tile_m * bitwise_and(right_shift(' + thread_id + ', 1), warp_height - 1)'
    # thread_i_idx = warp_y_offset + ' + size_thread_tile_m * bitwise_and(right_shift(' + thread_id + ', 1), warp_height - 1)'
    # thread_i_idx = 'size_thread_tile_m * bitwise_and(right_shift(' + thread_id + ', 1), warp_height - 1)'

    warp_width = math.ceil(schedule.warp_tile_n / schedule.thread_tile_n)
    if warp_width == 1:
        LaneIdx = threadId
        LaneIdy = '0'
    elif warp_width == 2:
        LaneIdx = 'bitwise_or(right_shift(bitwise_and(' + threadId + ', 96), 4), bitwise_and(' + threadId + ', 1))'
        LaneIdy = 'bitwise_and(right_shift(' + threadId + ', 1), warp_height - 1)'
    elif warp_width == 4:
        LaneIdx = 'bitwise_or(right_shift(bitwise_and(' + threadId + ', 48), 3), bitwise_and(' + threadId + ', 1))'
        LaneIdy = 'bitwise_and(right_shift(' + threadId + ', 1), warp_height - 1)'
    elif warp_width == 8:
        LaneIdx = 'bitwise_or(right_shift(bitwise_and(' + threadId + ', 24), 2), bitwise_and(' + threadId + ', 1))'
        LaneIdy = 'bitwise_and(right_shift(' + threadId + ', 1), warp_height - 1)'
    elif warp_width == 16:
        LaneIdx = 'bitwise_or(right_shift(bitwise_and(' + threadId + ', 28), 1), bitwise_and(' + threadId + ', 1))'
        LaneIdy = 'bitwise_and(right_shift(' + threadId + ', 1), warp_height - 1)'
    elif warp_width == 32:
        LaneIdx = '0'
        LaneIdy = threadId

thread_x_offset = '(' + LaneIdx + ' * size_thread_tile_n)'
thread_y_offset = '(' + LaneIdy + ' * size_thread_tile_m)'
# thread_x_range = warp_x_offset + ' + ' + thread_x_offset + ':' + warp_x_offset + ' + ' + thread_x_offset + '+size_thread_tile_n'
# thread_y_range = warp_y_offset + ' + ' + thread_y_offset + ':' + warp_y_offset + ' + ' + thread_y_offset + '+size_thread_tile_m'


####################################################################################################################
### Data Movement: _A
# _A -> shared_memory_A

nested_state.add_memlet_path(_A, thread_block_grid_map_entry, K_tile_map_entry, shared_memory_A, memlet=dace.Memlet.simple(_A.data, thread_block_i_offset + ':' + thread_block_i_offset + '+size_thread_block_tile_m, ' + k_tile_range))
# nested_state.add_memlet_path(_A, thread_block_grid_map_entry, K_tile_map_entry, shared_memory_A, memlet=dace.Memlet.simple(_A.data, k_tile_range + ', ' + thread_block_i_offset + ':' + thread_block_i_offset + '+size_thread_block_tile_m'))

# shared_memory_A -> register_storage_A (load size_thread_tile_m elements into register storage)
# nested_state.add_memlet_path(shared_memory_A, thread_tile_map_entry, thread_K_map_entry, register_storage_A, memlet=dace.Memlet.simple(shared_memory_A, 'k, ' + warp_y_offset + ' + ' + thread_y_offset + ':' + warp_y_offset + ' + ' + thread_y_offset + '+size_thread_tile_m'))
nested_state.add_memlet_path(shared_memory_A, thread_tile_map_entry, thread_K_map_entry, register_storage_A, memlet=dace.Memlet.simple(shared_memory_A, warp_y_offset + ' + ' + thread_y_offset + ':' + warp_y_offset + ' + ' + thread_y_offset + '+size_thread_tile_m, k'))

# TODO: for double precision, we probably want four memlets here to load 8 elements
# nested_state.add_memlet_path(shared_memory_A, thread_tile_map_entry, thread_K_map_entry, register_storage_A, memlet=dace.Memlet(f"{shared_memory_A}[k, {warp_y_offset} + {thread_y_offset} : {warp_y_offset} + {thread_y_offset} + size_thread_tile_m/2] -> 0:3"))
# additional_offset = 'warp_height * VECLEN'
# nested_state.add_memlet_path(shared_memory_A, thread_tile_map_entry, thread_K_map_entry, register_storage_A, memlet=dace.Memlet(f"{shared_memory_A}[k, {additional_offset} + {warp_y_offset} + {thread_y_offset} : {additional_offset} + {warp_y_offset} + {thread_y_offset} + size_thread_tile_m/2] -> 4:7"))

# register_storage_A -> tasklet
nested_state.add_memlet_path(register_storage_A,
                        thread_map_entry,
                        tasklet,
                        dst_conn='__a',
                        memlet=dace.Memlet(f"{register_storage_A.data}[i]"))

####################################################################################################################
### Data Movement: _B
# _B -> shared_memory_B


nested_state.add_memlet_path(_B, thread_block_grid_map_entry, K_tile_map_entry, shared_memory_B, memlet=dace.Memlet.simple(_B.data, k_tile_range + ', ' + thread_block_j_offset  + ':' + thread_block_j_offset + '+size_thread_block_tile_n'))

# shared_memory_B -> register_storage_B (load size_thread_tile_n elements into register storage)
nested_state.add_memlet_path(shared_memory_B, thread_tile_map_entry, thread_K_map_entry, register_storage_B, memlet=dace.Memlet.simple(shared_memory_B, 'k, ' + warp_x_offset + ' + ' + thread_x_offset + ':' + warp_x_offset + ' + ' + thread_x_offset + '+size_thread_tile_n'))

# TODO: for double precision, we probably want four memlets here to load 8 elements
# nested_state.add_memlet_path(shared_memory_B, thread_tile_map_entry, thread_K_map_entry, register_storage_B, memlet=dace.Memlet(f"{shared_memory_B}[k, {warp_x_offset} + {thread_x_offset} : {warp_x_offset} + {thread_x_offset} + size_thread_tile_n/2] -> 0:3"))
# additional_offset = 'warp_width * VECLEN'
# nested_state.add_memlet_path(shared_memory_B, thread_tile_map_entry, thread_K_map_entry, register_storage_B, memlet=dace.Memlet(f"{shared_memory_B}[k, {additional_offset} + {warp_x_offset} + {thread_x_offset} : {additional_offset} + {warp_x_offset} + {thread_x_offset} + size_thread_tile_n/2] -> 4:7"))

# register_storage_B -> tasklet
nested_state.add_memlet_path(register_storage_B,
                        thread_map_entry,
                        tasklet,
                        dst_conn='__b',
                        memlet=dace.Memlet(f"{register_storage_B.data}[j]"))

####################################################################################################################
### Data Movement: output
# tasklet -> register_storage_C
subset = thread_block_i_offset + ' + ' + warp_y_offset + ' + ' + thread_y_offset + ':' + thread_block_i_offset + ' + ' + warp_y_offset + ' + ' + thread_y_offset + '+size_thread_tile_m' + ', ' + thread_block_j_offset + ' + ' + warp_x_offset + ' + ' + thread_x_offset + ':' + thread_block_j_offset + ' + ' + warp_x_offset + ' + ' + thread_x_offset + '+size_thread_tile_n'

wcr_no_conflicts = False 
if num_threads_per_thread_block == 32 or args.double_buffering_shared:
    wcr_no_conflicts = True

# print('No write conflicts: ' + str(wcr_no_conflicts))

nested_state.add_memlet_path(tasklet,
                    thread_map_exit,
                    thread_K_map_exit,
                    register_storage_C,
                    src_conn='__out',
                    memlet=dace.Memlet(
                        f"{register_storage_C.data}[i, j]",
                        wcr='(lambda x, y: (x + y))',
                        wcr_nonatomic=True)
                    )

    
# int global_i = (size_thread_block_tile_m * block_idx_y) + ((size_thread_tile_m / 2) * bitwise_and(right_shift((thread % 32), 1), (warp_height - 1))) + ((size_warp_tile_m) * ((thread / 32) / num_warps_n));

# int global_j = (size_thread_block_tile_n * block_idx_x) + (size_thread_tile_n / 2 * bitwise_or(right_shift(bitwise_and((thread % 32), 24), 2), bitwise_and((thread % 32), 1))) + (size_warp_tile_n * ((thread / 32) % num_warps_n));

# for (int i = 0; i < 2; ++i) {
#     for (int j = 0; j < 4; ++j) {
#         for (int k = 0; k < 2; ++k) {
#             dace::CopyND<float, 1, false, 4>::template ConstDst<1>::Copy(
#                 Thread_Tile + 32 * i + 8 * j + 4 * k, output + (global_i + 16 * i + j) * M + global_j + 32 * k, 1);
#         }
#     }
# }

# register_storage_C -> A_matmul_B_nested_state (= result that will be transferred to outer sdfg)
if args.split_k == 1:
    nested_state.add_memlet_path(register_storage_C,
                                thread_tile_map_exit,
                                K_tile_map_exit,
                                thread_block_grid_map_exit,
                                A_matmul_B_nested_state,
                                memlet=dace.Memlet(
                                    data=A_matmul_B_nested_state.data,
                                    subset=subset,
                                    wcr='(lambda x, y: (x + y))',
                                    wcr_nonatomic=wcr_no_conflicts) # needed so we have a non-atomic accumulate accross thread blocks
                                )

    # if (math.fabs(args.beta) < 1e-6):
        # for i in range(2):
            # for j in range(4):
                # for k in range(2):
                    # nested_state.add_memlet_path(register_storage_C, thread_tile_map_exit, K_tile_map_exit, thread_block_grid_map_exit, A_matmul_B_nested_state,
                        # memlet=dace.Memlet(f"{register_storage_C}[4*i + j, 4*k:4*k+4] -> {thread_block_i_offset} + {warp_y_offset} + 16*i + j, {thread_block_j_offset} + {warp_x_offset} + 32*k:{thread_block_j_offset} + {warp_x_offset} + 32*k+4"))
    # else:
        # for i in range(2):
            # for j in range(4):
                # for k in range(2):
                    # nested_state.add_memlet_path(register_storage_C, thread_tile_map_exit, K_tile_map_exit, thread_block_grid_map_exit, A_matmul_B_nested_state,
                        # memlet=dace.Memlet(f"{register_storage_C}[4*i + j, 4*k:4*k+4] -> [{thread_block_i_offset} + {warp_y_offset} + 16*i + j, {thread_block_j_offset} + {warp_x_offset} + 32*k:{thread_block_j_offset} + {warp_x_offset} + 32*k+4]", wcr='(lambda x, y: (x + y))', wcr_nonatomic=wcr_no_conflicts))

else:
    subset = 'thread_block_k, ' + subset

    nested_state.add_memlet_path(register_storage_C,
                    thread_tile_map_exit,
                    K_tile_map_exit,
                    thread_block_grid_map_exit,
                    partial_split_k_output,
                    memlet=dace.Memlet(
                        data=partial_split_k_output.data,
                        subset=subset,
                        wcr='(lambda x, y: (x + y))',
                        # wcr_nonatomic=wcr_no_conflicts) # needed so we have a non-atomic accumulate accross thread blocks
                        wcr_nonatomic=True) # needed so we have a non-atomic accumulate accross thread blocks
                    )
                        # memlet=dace.Memlet(data=partial_split_k_output.data, subset=subset))

    # Reduce the split k
    tasklet_code = '__out[0] = __in[0]'
    for i in range(args.accumulator_split_k - 1):
        tasklet_code += '\n__out[' + str(i+1) + '] = __in[' + str(i+1) + ']'
    if args.verbose:
        print(tasklet_code)

    tasklet = nested_state.add_tasklet('reduce_split_k', ['__in'], ['__out'], tasklet_code)

    reduction_entry, reduction_exit = nested_state.add_map(
            'reduction_map',
            dict(i='0:M', j='0:N:ACCUMULATOR_SIZE'),
            schedule=dace.dtypes.ScheduleType.GPU_Device)

    reduce_split_k_entry, reduce_split_k_exit = nested_state.add_map(
            'reduce_split_k',
            dict(k ='0:SPLIT_K'),
            schedule=dace.dtypes.ScheduleType.Sequential)

    nested_state.add_memlet_path(partial_split_k_output,
                            reduction_entry,
                            reduce_split_k_entry,
                            tasklet,
                            dst_conn='__in',
                            memlet=dace.Memlet("partial_split_k_output[k, i, j:j+ACCUMULATOR_SIZE]"))

    nested_state.add_memlet_path(tasklet,
                            reduce_split_k_exit,
                            accumulator,
                            src_conn='__out',
                            memlet=dace.Memlet(f"{accumulator.data}[0:ACCUMULATOR_SIZE]", wcr='(lambda x, y: (x + y))'))

    nested_state.add_memlet_path(accumulator,
                            reduction_exit,
                            A_matmul_B_nested_state,
                            memlet=dace.Memlet(f"{A_matmul_B_nested_state.data}[i, j:j+ACCUMULATOR_SIZE]"))

    # TODO: Maybe we could use Vectorization.apply_to() here to make the reduction kernel even faster?
    # Vectorization.apply_to(nested_state.parent,
                # dict(vector_len=veclen, preamble=False, postamble=False),
                # _map_entry=reduction_entry,
                # _tasklet=tasklet,
                # _map_exit=reduction_exit)

                # _map_entry=entry,
                # _tasklet=state.out_edges(entry)[0].dst,
                # _map_exit=state.exit_node(entry))
        
# print("we are here")
# sdfg.save('sdfg_api_v4.sdfg')
# nested_sdfg.validate()
# sdfg.validate()

if args.double_buffering_register:
    helpers.print_info("Applying Double Buffering on the registers...", False)
    DoubleBuffering.apply_to(nested_sdfg, dict(full_data=True), map_entry=thread_K_map_entry, transient=register_storage_A) # Double buffering on the registers
    # DoubleBuffering.apply_to(nested_sdfg, map_entry=thread_K_map_entry, transient=register_storage_A) # Double buffering on the registers

# Reverse K map:
if args.reverse_k and not args.double_buffering_shared:
    print("Reversing K map")
    MapToForLoop.apply_to(nested_sdfg, dict(reverse=True, full_data=True), map_entry=K_tile_map_entry)

# sdfg.save('sdfg_api_v4_tmp.sdfg')

if args.double_buffering_shared:
    helpers.print_info("Applying Double Buffering on the shared memory...", False)
    DoubleBuffering.apply_to(nested_sdfg, dict(reverse=args.reverse_k, full_data=args.nested_pointer_offset), map_entry=K_tile_map_entry, transient=shared_memory_A) # Double buffering on the shared memory (with reversed K map)


# nested_sdfg.fill_scope_connectors()
sdfg.fill_scope_connectors()
# sdfg.save('sdfg_api_v4.sdfg')
# nested_sdfg.validate()
sdfg.validate()

sdfg.arg_names = ['A', 'B', 'C', 'alpha', 'beta']
sdfg.save('sdfg_api_v4' + str(args.name) + '.sdfg')
csdfg = sdfg.compile()

for i in range(args.repetitions):
    A = np.random.rand(args.M, args.K).astype(ndtype)
    B = np.random.rand(args.K, args.N).astype(ndtype)
    C = np.zeros((args.M, args.N)).astype(ndtype)

    # helpers.print_info("Running sdfg...", False)
    csdfg(A=A, B=B, C=C, alpha=dtype(args.alpha), beta=dtype(args.beta), M=args.M, N=args.N, K=args.K)

    if args.verify:
        helpers.print_info("Verifying results...", False)
        
        def matmul(A: dtype[M, K], B: dtype[K, N], C: dtype[M, N], alpha: dtype, beta: dtype):
            return alpha * (A @ B) + beta * C

        C_init = np.zeros((args.M, args.N)).astype(ndtype)
        C_correct = matmul(A=A, B=B, C=C_init, alpha=dtype(args.alpha), beta=dtype(args.beta))
        # print(C)
        # print('--')
        diff = np.linalg.norm(C - C_correct) / (args.M * args.N)
        helpers.print_info("Difference: ", False)
        print(diff)
        if diff <= 10e-6:
            helpers.print_success("The SDFG is correct!", False)
        else:
            helpers.print_error("The SDFG is incorrect!", False)

        # # Can replace this with np.allclose(A, B)
        def areSame(A,B):
            for i in range(args.M):
                for j in range(args.N):
                    diff = math.fabs(A[i][j] - B[i][j])
                    if (diff > 0.01):
                        helpers.print_error("Error at position (" + str(i) + ", " + str(j) + "): matrices are not equal! Difference is: " + str(diff), False)
                        helpers.print_error(str(B[i][j]) + " should be " + str(A[i][j]), False)
                        print()
                        return False
            return True

        helpers.print_info("Correct result: ", False)
        for i in range(8):
            for j in range(8):
                print("%.2f" % C_correct[i][j], end=" ")
            print()

        print()
        print()
        helpers.print_info("SDFG result: ", False)
        for i in range(8):
            for j in range(8):
                print("%.2f" % C[i][j], end=" ")
            print()

        if areSame(C_correct, C):
            helpers.print_success("The SDFG is correct!", False)
        else:
            helpers.print_error("The SDFG is incorrect!", False)
