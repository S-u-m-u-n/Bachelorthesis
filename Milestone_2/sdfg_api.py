from argparse import ArgumentParser
import numpy as np
import sympy as sy
import dace
import math
from Schedule import Schedule
import helpers
from dace.transformation.dataflow import Vectorization, DoubleBuffering, AccumulateTransient

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
parser.add_argument("--beta", type=np.float64, dest='beta', nargs="?", default=1.0)
parser.add_argument("-r", "--repetitions", type=int, dest='repetitions', nargs="?", default=1)
parser.add_argument("--version",
                    choices=['dace' 'cublas'],
                    default='dace',
                    help='''Different available versions:
dace: Transform `matmul` to a reasonably-optimized version for GPU;
cublas: Run `matmul` with the CUBLAS library node implementation.''')
parser.add_argument('-p', '--precision',
                    dest='precision',
                    choices=['16', '32', '64', '128'],
                    default='64',
                    help="Specify bit precision (16, 32, 64 or 128) - currently unsupported.")
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
parser.add_argument('--swizzle-thread-blocks', type=int, dest='swizzle_thread_blocks', nargs="?", default=1)
parser.add_argument('--swizzle-threads',
                    dest='swizzle_threads',
                    help="Use swizzle on the threads",
                    action="store_true",
                    default=False)
parser.add_argument('--vectorization',
                    dest='vectorization',
                    help="Use vectorization",
                    action="store_true",
                    default=False)
parser.add_argument('--double-buffering',
                    dest='double_buffering',
                    help="Use double buffering",
                    action="store_true",
                    default=False)

args = parser.parse_args()
if args.verbose:
    helpers.print_info("Program launched with the following arguments: " + str(args), args.colorless)


schedule = Schedule(load_k=8, thread_tile_m=4, thread_tile_n=4, thread_tile_k=8, warp_tile_m=64, warp_tile_n=32,
                        thread_block_tile_m=64, thread_block_tile_n=32, thread_block_tile_k=640,
                        SWIZZLE_thread_block=2, SWIZZLE_thread_tile=True, split_k=2, double_buffering=True)

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
M_example = args.M
N_example = args.N
K_example = args.K
alpha = args.alpha
beta = args.beta

sdfg = dace.SDFG('gemm')
state = sdfg.add_state(label='gemm_state')
nested_sdfg = dace.SDFG('nested_gemm')

sdfg.add_array('A', shape=[M, K], dtype=dace.float64)
sdfg.add_array('B', shape=[K, N], dtype=dace.float64)
sdfg.add_array('C', shape=[M, N], dtype=dace.float64)
A_in = state.add_read('A')
B_in = state.add_read('B')
C_in = state.add_read('C')
C_out = state.add_write('C')

if args.vectorization:
    helpers.print_info('Applying Vectorization...', False)
    sdfg.add_transient('gpu_A', shape=[M, K / 2], dtype=dace.vector(dace.float64, 2), storage=dace.StorageType.GPU_Global)
    sdfg.add_transient('gpu_B', shape=[K, N / 2], dtype=dace.vector(dace.float64, 2), storage=dace.StorageType.GPU_Global)
else:
    sdfg.add_transient('gpu_A', shape=[M, K], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_transient('gpu_B', shape=[K, N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)

sdfg.add_transient('gpu_C', shape=[M, N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg.add_transient('gpu_result', shape=[M, N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)


gpu_A = state.add_access('gpu_A')
gpu_B = state.add_access('gpu_B')
gpu_C = state.add_access('gpu_C')
gpu_result = state.add_access('gpu_result')

sdfg.add_constant('alpha', 1.0)
sdfg.add_constant('beta', 1.0)

sdfg.add_transient('C_times_beta', shape=[M, N], dtype=dace.float64, storage=dace.StorageType.GPU_Global, lifetime=dace.AllocationLifetime.SDFG)
sdfg.add_transient('A_matmul_B', shape=[M, N], dtype=dace.float64, storage=dace.StorageType.GPU_Global, lifetime=dace.AllocationLifetime.SDFG)
sdfg.add_transient('A_matmul_B_times_alpha', shape=[M, N], dtype=dace.float64, storage=dace.StorageType.GPU_Global, lifetime=dace.AllocationLifetime.SDFG)

C_times_beta = state.add_write('C_times_beta')
A_matmul_B = state.add_write('A_matmul_B')
A_matmul_B_times_alpha = state.add_write('A_matmul_B_times_alpha')

#########################################################
# Connect arrays to GPU transient and the GPU result transient to host array
state.add_edge(A_in, None, gpu_A, None, memlet=dace.Memlet.simple(A_in.data, '0:M, 0:K'))
state.add_edge(B_in, None, gpu_B, None, memlet=dace.Memlet.simple(B_in.data, '0:K, 0:N'))
state.add_edge(C_in, None, gpu_C, None, memlet=dace.Memlet.simple(C_in.data, '0:M, 0:N'))
state.add_edge(gpu_result, None, C_out, None, memlet=dace.Memlet.simple(gpu_result.data, '0:M, 0:N'))

#########################################################
# Multiply C with beta
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
# Multiply the result of (A @ B) with alpha
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

#########################################################
# Add the result of (A @ B) * alpha and C * beta
map_entry, map_exit = state.add_map(
        'add_matrices',
        dict(i='0:M', j='0:N'),
        schedule=dace.dtypes.ScheduleType.GPU_Device)

tasklet = state.add_tasklet('add_matrices', ['__in1', '__in2'], ['__out'], '__out = (__in1 + __in2)')

state.add_memlet_path(A_matmul_B_times_alpha,
                        map_entry,
                        tasklet,
                        dst_conn='__in1',
                        memlet=dace.Memlet(f"{A_matmul_B_times_alpha.data}[i, j]"))

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

#########################################################
# Create nested sdfg
nested_sdfg_node = state.add_nested_sdfg(nested_sdfg, state, {'input_A', 'input_B'}, {'output'}, schedule=dace.ScheduleType.GPU_Default, symbol_mapping={"K": "K", "M": "M", "N": "N"})

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

input_A = nested_sdfg.add_datadesc('input_A', desc_a)
input_B = nested_sdfg.add_datadesc('input_B', desc_b)
output = nested_sdfg.add_datadesc('output', desc_res)

_A = nested_state.add_read(input_A)
_B = nested_state.add_read(input_B)
# A_matmul_B_nested_initstate = nested_initstate.add_write(output)
A_matmul_B_nested_initstate = nested_initstate.add_access(output)
# A_matmul_B_nested_state = nested_state.add_write(output)
A_matmul_B_nested_state = nested_state.add_access(output)

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
nested_sdfg.add_transient('shared_memory_A', shape=[schedule.thread_block_tile_m, schedule.load_k], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
nested_sdfg.add_transient('shared_memory_B', shape=[schedule.load_k, schedule.thread_block_tile_n], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
nested_sdfg.add_transient('register_storage_A', shape=[schedule.thread_tile_m, 1], dtype=dace.float64, storage=dace.StorageType.Register)
nested_sdfg.add_transient('register_storage_B', shape=[1, schedule.thread_tile_n], dtype=dace.float64, storage=dace.StorageType.Register)
nested_sdfg.add_transient('register_storage_C', shape=[schedule.thread_tile_m, schedule.thread_tile_n], dtype=dace.float64, storage=dace.StorageType.Register)

if args.split_k > 1:
    nested_sdfg.add_transient('partial_split_k_output', shape=[M, N, args.split_k], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
    partial_split_k_output = nested_state.add_access('partial_split_k_output')


shared_memory_A = nested_state.add_access('shared_memory_A')
shared_memory_B = nested_state.add_access('shared_memory_B')
# shared_memory_A.setzero = True
# shared_memory_B.setzero = True
register_storage_A = nested_state.add_access('register_storage_A')
register_storage_B = nested_state.add_access('register_storage_B')
register_storage_C = nested_state.add_access('register_storage_C')
# register_storage_A.setzero = True
# register_storage_B.setzero = True
register_storage_C.setzero = True

sdfg.add_constant('size_thread_block_tile_m', schedule.thread_block_tile_m)
sdfg.add_constant('size_thread_block_tile_n', schedule.thread_block_tile_n)
sdfg.add_constant('size_K_tile', schedule.load_k)
# sdfg.add_constant('num_thread_blocks_m', int(M_example / schedule.thread_block_tile_m) if not args.swizzle_thread_blocks else int((M_example / schedule.thread_block_tile_m + 2 - 1) / 2))
sdfg.add_constant('num_thread_blocks_m', int((M_example / schedule.thread_block_tile_m + args.swizzle_thread_blocks - 1) // args.swizzle_thread_blocks))
# sdfg.add_constant('num_thread_blocks_n', int(N_example / schedule.thread_block_tile_n) if not args.swizzle_thread_blocks else 2*int(N_example / schedule.thread_block_tile_n))
sdfg.add_constant('num_thread_blocks_n', args.swizzle_thread_blocks * int(N_example / schedule.thread_block_tile_n))
sdfg.add_constant('num_K_tiles', int(K_example / schedule.load_k))
sdfg.add_constant('size_warp_tile_m', schedule.warp_tile_m)
sdfg.add_constant('size_warp_tile_n', schedule.warp_tile_n)
sdfg.add_constant('size_thread_tile_m', schedule.thread_tile_m)
sdfg.add_constant('size_thread_tile_n', schedule.thread_tile_n)
nested_sdfg.add_constant('size_thread_block_tile_m', schedule.thread_block_tile_m)
nested_sdfg.add_constant('size_thread_block_tile_n', schedule.thread_block_tile_n)
nested_sdfg.add_constant('size_K_tile', schedule.load_k)
nested_sdfg.add_constant('size_K_split', int(K_example / args.split_k))
nested_sdfg.add_constant('num_thread_blocks_m', int((M_example / schedule.thread_block_tile_m + args.swizzle_thread_blocks - 1) // args.swizzle_thread_blocks))
nested_sdfg.add_constant('num_thread_blocks_n', args.swizzle_thread_blocks * int(N_example / schedule.thread_block_tile_n))
nested_sdfg.add_constant('num_K_tiles', int(K_example / (schedule.load_k * args.split_k)))
nested_sdfg.add_constant('size_warp_tile_m', schedule.warp_tile_m)
nested_sdfg.add_constant('size_warp_tile_n', schedule.warp_tile_n)
nested_sdfg.add_constant('size_thread_tile_m', schedule.thread_tile_m)
nested_sdfg.add_constant('size_thread_tile_n', schedule.thread_tile_n)
# nested_sdfg.add_constant('size_thread_tile_k', schedule.thread_tile_k) # = size_K_tile
nested_sdfg.add_constant('warp_width', math.ceil(schedule.warp_tile_n / schedule.thread_tile_n))
nested_sdfg.add_constant('warp_height', math.ceil(schedule.warp_tile_m / schedule.thread_tile_m))
nested_sdfg.add_constant('SWIZZLE', args.swizzle_thread_blocks)
nested_sdfg.add_constant('SPLIT_K', args.split_k)

if args.vectorization:
    tasklet = nested_state.add_tasklet('matrix_multiplication', {'__a': dace.vector(dace.float64, 2), '__b': dace.vector(dace.float64, 2)}, {'__out': dace.float64}, '__out = (__a * __b)')
else:
    tasklet = nested_state.add_tasklet('matrix_multiplication', {'__a', '__b'}, {'__out'}, '__out = (__a * __b)')

if args.split_k > 1:
    split_k_map_entry, split_k_map_exit = nested_state.add_map(
        'Split_K',
        dict(split_k_idx='0:SPLIT_K'),
        schedule=dace.dtypes.ScheduleType.Sequential)

# This map creates threadblocks
thread_block_grid_map_entry, thread_block_grid_map_exit = nested_state.add_map(
        'Thread_block_grid',
        dict(thread_block_i='0:num_thread_blocks_m', thread_block_j='0:num_thread_blocks_n'),
        # schedule=dace.dtypes.ScheduleType.Default)
        schedule=dace.dtypes.ScheduleType.GPU_Device)

K_tile_map_entry, K_tile_map_exit = nested_state.add_map(
        'K_tile',
        dict(k_tile='0:num_K_tiles'),
        schedule=dace.dtypes.ScheduleType.Sequential)

# This map creates threads (and not warps!!)
thread_tile_map_entry, thread_tile_map_exit = nested_state.add_map(
        'Thread_tile',
        dict(thread_i='0:size_thread_block_tile_m:size_thread_tile_m', thread_j='0:size_thread_block_tile_n:size_thread_tile_n'),
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

if args.swizzle_threads:
    helpers.print_info("Swizzling Threads...", False)
    bitwise_and = sy.Function('bitwise_and')
    bitwise_or = sy.Function('bitwise_or')
    right_shift = sy.Function('right_shift')

####################################################################################################################
### Data Movement: _A
# _A -> shared_memory_A
if args.split_k == 1:
    nested_state.add_memlet_path(_A, thread_block_grid_map_entry, K_tile_map_entry, shared_memory_A, memlet=dace.Memlet.simple(_A.data,
    'thread_block_i*size_thread_block_tile_m:thread_block_i*size_thread_block_tile_m+size_thread_block_tile_m, k_tile*size_K_tile:k_tile*size_K_tile+size_K_tile' if args.swizzle_thread_blocks == 1 else
    '(thread_block_i*SWIZZLE + (thread_block_j % SWIZZLE))*size_thread_block_tile_m:(thread_block_i*SWIZZLE + (thread_block_j % SWIZZLE))*size_thread_block_tile_m+size_thread_block_tile_m, k_tile*size_K_tile:k_tile*size_K_tile + size_K_tile'
    ))
else:
    nested_state.add_memlet_path(_A, split_k_map_entry, thread_block_grid_map_entry, K_tile_map_entry, shared_memory_A, memlet=dace.Memlet.simple(_A.data,
    'thread_block_i*size_thread_block_tile_m:thread_block_i*size_thread_block_tile_m+size_thread_block_tile_m, (size_K_split*split_k_idx) + k_tile*size_K_tile:(size_K_split*split_k_idx) + k_tile*size_K_tile+size_K_tile' if args.swizzle_thread_blocks == 1 else
    '(thread_block_i*SWIZZLE + (thread_block_j % SWIZZLE))*size_thread_block_tile_m:(thread_block_i*SWIZZLE + (thread_block_j % SWIZZLE))*size_thread_block_tile_m+size_thread_block_tile_m, (size_K_split*split_k_idx) + k_tile*size_K_tile:(size_K_split*split_k_idx) + k_tile*size_K_tile + size_K_tile'
    ))
# shared_memory_A -> register_storage_A
nested_state.add_memlet_path(shared_memory_A, thread_tile_map_entry, thread_K_map_entry, register_storage_A, memlet=dace.Memlet.simple(shared_memory_A, # load size_thread_tile_m elements into register storage
'thread_i:thread_i+size_thread_tile_m, k' if not args.swizzle_threads else
'''(thread_i // size_warp_tile_m)*size_warp_tile_m + size_thread_tile_m * bitwise_and(right_shift(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1), warp_height - 1)
:(thread_i // size_warp_tile_m)*size_warp_tile_m + size_thread_tile_m * bitwise_and(right_shift(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1), warp_height - 1)
+size_thread_tile_m, k''')) # load size_thread_tile_m elements into register storage
# register_storage_A -> tasklet
nested_state.add_memlet_path(register_storage_A,
                        thread_map_entry,
                        tasklet,
                        dst_conn='__a',
                        memlet=dace.Memlet(f"{register_storage_A.data}[i, 0]"))

####################################################################################################################
### Data Movement: _B
# _B -> shared_memory_B
if args.split_k == 1:
    nested_state.add_memlet_path(_B, thread_block_grid_map_entry, K_tile_map_entry, shared_memory_B, memlet=dace.Memlet.simple(_B.data,
    'k_tile*size_K_tile:k_tile*size_K_tile + size_K_tile, thread_block_j*size_thread_block_tile_n:thread_block_j*size_thread_block_tile_n+size_thread_block_tile_n'  if args.swizzle_thread_blocks == 1 else
    'k_tile*size_K_tile:k_tile*size_K_tile + size_K_tile, (thread_block_j // SWIZZLE)*size_thread_block_tile_n:(thread_block_j // SWIZZLE)*size_thread_block_tile_n + size_thread_block_tile_n'
    ))
else:
    nested_state.add_memlet_path(_B, split_k_map_entry, thread_block_grid_map_entry, K_tile_map_entry, shared_memory_B, memlet=dace.Memlet.simple(_B.data,
    '(size_K_split*split_k_idx) + k_tile*size_K_tile:(size_K_split*split_k_idx) + k_tile*size_K_tile + size_K_tile, thread_block_j*size_thread_block_tile_n:thread_block_j*size_thread_block_tile_n+size_thread_block_tile_n'  if args.swizzle_thread_blocks == 1 else
    '(size_K_split*split_k_idx) + k_tile*size_K_tile:(size_K_split*split_k_idx) + k_tile*size_K_tile + size_K_tile, (thread_block_j // SWIZZLE)*size_thread_block_tile_n:(thread_block_j // SWIZZLE)*size_thread_block_tile_n + size_thread_block_tile_n'
    ))
# shared_memory_B -> register_storage_B
nested_state.add_memlet_path(shared_memory_B, thread_tile_map_entry, thread_K_map_entry, register_storage_B, memlet=dace.Memlet.simple(shared_memory_B, # load size_thread_tile_n elements into register storage
'k, thread_j:thread_j + size_thread_tile_n' if not args.swizzle_threads else
'''k, (thread_j // size_warp_tile_n)*size_warp_tile_n + size_thread_tile_n * bitwise_or(right_shift(bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), warp_height * warp_width // 2), warp_width - 1), bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1))
:(thread_j // size_warp_tile_n)*size_warp_tile_n + size_thread_tile_n * bitwise_or(right_shift(bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), warp_height * warp_width // 2), warp_width - 1), bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1))
+size_thread_tile_n''')) # load size_thread_tile_n elements into register storage
# register_storage_B -> tasklet
nested_state.add_memlet_path(register_storage_B,
                        thread_map_entry,
                        tasklet,
                        dst_conn='__b',
                        memlet=dace.Memlet(f"{register_storage_B.data}[0, j]"))
                        # if not args.vectorization else dace.Memlet(f"{register_storage_B.data}[0, j:j+2]"))
                        # memlet=dace.Memlet(data=register_storage_B.data,
                        # subset="0:0, j:j" if not args.vectorization else "0:0, j:j+2"))

####################################################################################################################
### Data Movement: output
# tasklet -> register_storage_C
if (not args.swizzle_threads and args.swizzle_thread_blocks == 1): # Do not swizzle anything
    subset = '''thread_block_i*size_thread_block_tile_m + thread_i
:thread_block_i*size_thread_block_tile_m + thread_i + size_thread_tile_m
, thread_block_j*size_thread_block_tile_n + thread_j:
thread_block_j*size_thread_block_tile_n + thread_j + size_thread_tile_n'''
elif (args.swizzle_threads and args.swizzle_thread_blocks == 1): # Only swizzle threads
    subset = '''thread_block_i*size_thread_block_tile_m + (thread_i // size_warp_tile_m) * size_warp_tile_m + size_thread_tile_m * bitwise_and(right_shift(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1), warp_height - 1)
:thread_block_i*size_thread_block_tile_m + (thread_i // size_warp_tile_m) * size_warp_tile_m + size_thread_tile_m * bitwise_and(right_shift(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1), warp_height - 1)
+ size_thread_tile_m
,thread_block_j*size_thread_block_tile_n + (thread_j // size_warp_tile_n) * size_warp_tile_n + size_thread_tile_n * bitwise_or(right_shift(bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), warp_height * warp_width // 2), warp_width - 1), bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1))
:thread_block_j*size_thread_block_tile_n + (thread_j // size_warp_tile_n) * size_warp_tile_n + size_thread_tile_n * bitwise_or(right_shift(bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), warp_height * warp_width // 2), warp_width - 1), bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1))
+ size_thread_tile_n'''
elif (not args.swizzle_threads and args.swizzle_thread_blocks > 1): # Only swizzle thread blocks
    helpers.print_info("Swizzling Thread Blocks...", False)
    subset = '''(thread_block_i*SWIZZLE + (thread_block_j % SWIZZLE))*size_thread_block_tile_m + thread_i
:(thread_block_i*SWIZZLE + (thread_block_j % SWIZZLE))*size_thread_block_tile_m + thread_i + size_thread_tile_m
, (thread_block_j // SWIZZLE)*size_thread_block_tile_n + thread_j:
(thread_block_j // SWIZZLE)*size_thread_block_tile_n + thread_j + size_thread_tile_n'''
elif (args.swizzle_threads and args.swizzle_thread_blocks > 1): # Swizzle threads and thread blocks
    helpers.print_info("Swizzling Thread Blocks...", False)
    subset = '''(thread_block_i*SWIZZLE + (thread_block_j % SWIZZLE))*size_thread_block_tile_m + (thread_i // size_warp_tile_m) * size_warp_tile_m + size_thread_tile_m * bitwise_and(right_shift(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1), warp_height - 1)
:(thread_block_i*SWIZZLE + (thread_block_j % SWIZZLE))*size_thread_block_tile_m + (thread_i // size_warp_tile_m) * size_warp_tile_m + size_thread_tile_m * bitwise_and(right_shift(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1), warp_height - 1)
+ size_thread_tile_m
,(thread_block_j // SWIZZLE)*size_thread_block_tile_n + (thread_j // size_warp_tile_n) * size_warp_tile_n + size_thread_tile_n * bitwise_or(right_shift(bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), warp_height * warp_width // 2), warp_width - 1), bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1))
:(thread_block_j // SWIZZLE)*size_thread_block_tile_n + (thread_j // size_warp_tile_n) * size_warp_tile_n + size_thread_tile_n * bitwise_or(right_shift(bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), warp_height * warp_width // 2), warp_width - 1), bitwise_and(warp_width * ((thread_i % size_warp_tile_m) / size_thread_tile_m) + ((thread_j % size_warp_tile_n) / size_thread_tile_n), 1))
+ size_thread_tile_n'''
else:
    helpers.print_error("Invalid Swizzle case!", False)
    exit(1)

nested_state.add_memlet_path(tasklet,
                        thread_map_exit,
                        thread_K_map_exit,
                        register_storage_C,
                        src_conn='__out',
                        memlet=dace.Memlet(f"{register_storage_C.data}[i, j]", wcr='(lambda x, y: (x + y))'))
                        # if not args.vectorization else
                            # dace.Memlet(f"{register_storage_C.data}[i, j:j+2]", wcr='(lambda x, y: (x + y))'))
# register_storage_C -> A_matmul_B_nested_state (= result that will be transferred to outer sdfg)
if args.split_k == 1:
    nested_state.add_memlet_path(register_storage_C,
                            thread_tile_map_exit,
                            K_tile_map_exit,
                            thread_block_grid_map_exit,
                            # split_k_map_exit,
                            A_matmul_B_nested_state,
                            memlet=dace.Memlet(data=A_matmul_B_nested_state.data,
                            subset= subset,
                            wcr='(lambda x, y: (x + y))',
                            ))
                            # wcr_nonatomic=True)) # needed so we have a non-atomic accumulate accross thread blocks
else:
    nested_state.add_memlet_path(register_storage_C,
                            thread_tile_map_exit,
                            K_tile_map_exit,
                            thread_block_grid_map_exit,
                            partial_split_k_output,
                            memlet=dace.Memlet(data=partial_split_k_output.data,
                            subset= subset,
                            wcr='(lambda x, y: (x + y))',
                            ))
                            # wcr_nonatomic=True)) # needed so we have a non-atomic accumulate accross thread blocks
    
    nested_state.add_memlet_path(partial_split_k_output,
                            split_k_map_exit,
                            A_matmul_B_nested_state,
                            memlet=dace.Memlet(f"{A_matmul_B_nested_state.data}[i, j]", wcr='(lambda x, y: (x + y))', wcr_nonatomic=True))
    
    # AccumulateTransient.apply_to(nested_state.parent, map_exit=thread_block_grid_map_exit, outer_map_exit=split_k_map_exit)


if args.double_buffering:
    helpers.print_info("Applying Double Buffering...", False)
    DoubleBuffering.apply_to(nested_state.parent, _map_entry=K_tile_map_entry, _transient=shared_memory_A)

nested_sdfg.fill_scope_connectors()
sdfg.fill_scope_connectors()
sdfg.save('sdfg_api.sdfg')
nested_sdfg.validate()
sdfg.validate()

sdfg.arg_names = ['A', 'B', 'C', 'alpha', 'beta']
sdfg.save('sdfg_api.sdfg')
csdfg = sdfg.compile()

for i in range(args.repetitions):
    A = np.random.rand(M_example, K_example).astype(np.float64)
    B = np.random.rand(K_example, N_example).astype(np.float64)
    C = np.zeros((M_example, N_example)).astype(np.float64)

    # helpers.print_info("Running sdfg...", False)
    csdfg(A=A, B=B, C=C, alpha=alpha, beta=beta, M=M_example, N=N_example, K=K_example)

    if args.verify:
        helpers.print_info("Verifying results...", False)
        
        def matmul(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N], alpha: dace.float64, beta: dace.float64):
            return alpha * (A @ B) + beta * C

        C_init = np.zeros((M_example, N_example)).astype(np.float64)
        C_correct = matmul(A=A, B=B, C=C_init, alpha=alpha, beta=beta)
        # print(C)
        # print('--')
        diff = np.linalg.norm(C - C_correct) / (M_example * N_example)
        helpers.print_info("Difference: ", False)
        print(diff)
        if diff <= 1e-6:
            helpers.print_success("The SDFG is correct!", False)
        else:
            helpers.print_error("The SDFG is incorrect!", False)

        # Can replace this with np.allclose(A, B)
        def areSame(A,B):
            for i in range(M_example):
                for j in range(N_example):
                    diff = math.fabs(A[i][j] - B[i][j])
                    if (diff > 0.000001):
                        helpers.print_error("Error at position (" + str(i) + ", " + str(j) + "): matrices are not equal! Difference is: " + str(diff), False)
                        helpers.print_error(str(B[i][j]) + " should be " + str(A[i][j]), False)
                        print()
                        return False
            return True

        helpers.print_info("Correct result: ", False)
        for i in range(16):
            for j in range(16):
                print("%.2f" % C_correct[i][j], end=" ")
            print()

        print()
        print()
        helpers.print_info("SDFG result: ", False)
        for i in range(16):
            for j in range(16):
                print("%.2f" % C[i][j], end=" ")
            print()

        if areSame(C_correct, C):
            helpers.print_success("The SDFG is correct!", False)
        else:
            helpers.print_error("The SDFG is incorrect!", False)
