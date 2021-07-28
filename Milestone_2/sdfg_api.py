from argparse import ArgumentParser
import numpy as np
import dace
import math
from Schedule import Schedule
import helpers

schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, thread_tile_k=8, warp_tile_m=64, warp_tile_n=32,
                        thread_block_tile_m=128, thread_block_tile_n=128, thread_block_tile_k=640,
                        SWIZZLE_thread_block=2, SWIZZLE_thread_tile=True, split_k=2, double_buffering=False)

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
M_example = 640
N_example = 640
K_example = 640

sdfg = dace.SDFG('gemm')
state = sdfg.add_state(label='gemm_state')
nested_sdfg = dace.SDFG('nested_gemm')

sdfg.add_array('A', shape=[M, K], dtype=dace.float64)
sdfg.add_array('B', shape=[K, N], dtype=dace.float64)
sdfg.add_array('C', shape=[M, N], dtype=dace.float64)
sdfg.add_array('result', shape=[M, N], dtype=dace.float64)
A_in = state.add_read('A')
B_in = state.add_read('B')
C_in = state.add_read('C')
result = state.add_write('result')

sdfg.add_transient('gpu_A', shape=[M, K], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg.add_transient('gpu_B', shape=[K, N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg.add_transient('gpu_C', shape=[M, N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
sdfg.add_transient('gpu_result', shape=[M, N], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
gpu_A = state.add_access('gpu_A')
gpu_B = state.add_access('gpu_B')
gpu_C = state.add_access('gpu_C')
gpu_result = state.add_access('gpu_result') # add_access or add_write?

# sdfg.add_constant('alpha', value=1.0, dtype=dace.float64)
# sdfg.add_constant('beta', value=1.0, dtype=dace.float64)
# alpha_in = state.add_read('alpha')
# beta_in = state.add_read('beta')
# sdfg.add_array('alpha', shape=[1], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
# sdfg.add_array('beta', shape=[1], dtype=dace.float64, storage=dace.StorageType.GPU_Global)
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
state.add_edge(
    A_in, None,
    gpu_A, None,
    memlet=dace.Memlet.simple(A_in.data, '0:M, 0:K'))

state.add_edge(
    B_in, None,
    gpu_B, None,
    memlet=dace.Memlet.simple(B_in.data, '0:K, 0:N'))

state.add_edge(
    C_in, None,
    gpu_C, None,
    memlet=dace.Memlet.simple(C_in.data, '0:M, 0:N'))

state.add_edge(
    gpu_result, None,
    result, None,
    memlet=dace.Memlet.simple(gpu_result.data, '0:M, 0:N'))

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

# state.add_memlet_path(beta_in,
#                         map_entry,
#                         tasklet,
#                         dst_conn='__in2',
#                         memlet=dace.Memlet(f"{beta_in.data}[0]"))

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

# state.add_memlet_path(alpha_in,
#                         map_entry,
#                         tasklet,
#                         dst_conn='__in2',
#                         memlet=dace.Memlet(f"{alpha_in.data}[0]"))

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


state.add_edge(
    gpu_A, None,
    nested_sdfg_node, 'input_A',
    memlet=dace.Memlet.simple(gpu_A.data, '0:M, 0:K'))

state.add_edge(
    gpu_B, None,
    nested_sdfg_node, 'input_B',
    memlet=dace.Memlet.simple(gpu_B.data, '0:K, 0:N'))

# print(nested_sdfg_node.last_connector)
state.add_edge(
    nested_sdfg_node, 'output', 
    A_matmul_B, None,
    memlet=dace.Memlet.simple(A_matmul_B.data, '0:M, 0:N')) # A_matmul_B.data or A_matmul_B_nested.data or A_matmul_B_nested_state.data?

nested_initstate = nested_sdfg.add_state(label='nested_initstate')
nested_initstate.executions = 1
nested_initstate.dynamic_executions = False
nested_state = nested_sdfg.add_state(label='nested_state')
nested_state.executions = 1
nested_state.dynamic_executions = False

nested_sdfg.add_edge(nested_initstate, nested_state, dace.InterstateEdge()) # connect the two states
# print(sdfg.start_state)
# print(nested_sdfg.start_state)
# nested_sdfg.

for e in state.in_edges(nested_sdfg_node):
    if e.dst_conn == "input_A":
        desc_a = sdfg.arrays[e.data.data]
    elif e.dst_conn == "input_B":
        desc_b = sdfg.arrays[e.data.data]

for e in state.out_edges(nested_sdfg_node): # Can we find the connector without the edge?
    if e.src_conn == "output":
        desc_res = sdfg.arrays[e.data.data]

# print(nested_sdfg_node.out_connectors)

# print(desc_res)
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
A_matmul_B_nested_initstate = nested_initstate.add_write(output)
A_matmul_B_nested_state = nested_state.add_write(output)

# A_matmul_B_nested_read = state.add_read('A_matmul_B_nested')

# for e in state.out_edges(nested_sdfg_node):
    # if e.src_conn == "output":
        # desc_res = e

# desc_res.memlet = dace.Memlet.simple(A_matmul_B_nested_state.data, '0:M, 0:N') # A_matmul_B.data or A_matmul_B_nested.data or A_matmul_B_nested_state.data?

#########################################################
### matmul init state
map_entry, map_exit = nested_initstate.add_map(
        'initialize_matmul_result',
        dict(i='0:M', j='0:N'),
        schedule=dace.dtypes.ScheduleType.Default)

tasklet = nested_initstate.add_tasklet('matmul_init', [], ['out'], 'out = 0') # TODO: replace with out = 0 once this works

nested_initstate.add_memlet_path(map_entry,
            tasklet,
            # A_matmul_B_nested_initstate,
            # src_conn='out',
            memlet=dace.Memlet()) # TODO: can this be an empty Memlet???

nested_initstate.add_memlet_path(tasklet,
            map_exit,
            A_matmul_B_nested_initstate,
            src_conn='out',
            # memlet=dace.Memlet.simple(A_matmul_B_nested_initstate.data, '0:M, 0:N'))
            memlet=dace.Memlet(f"{A_matmul_B_nested_initstate.data}[i, j]"))

#########################################################
### matmul state
nested_sdfg.add_transient('shared_memory_A', shape=[schedule.thread_block_tile_m, schedule.load_k], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
nested_sdfg.add_transient('shared_memory_B', shape=[schedule.load_k, schedule.thread_block_tile_n], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
shared_memory_A = nested_state.add_access('shared_memory_A')
shared_memory_B = nested_state.add_access('shared_memory_B')

# nested_sdfg.add_transient('register_storage_A', shape=[schedule.thread_tile_m, schedule.thread_tile_k], dtype=dace.float64, storage=dace.StorageType.Default)
# nested_sdfg.add_transient('register_storage_B', shape=[schedule.thread_tile_k, schedule.thread_tile_n], dtype=dace.float64, storage=dace.StorageType.Default)
# nested_sdfg.add_transient('register_storage_C', shape=[schedule.thread_tile_m, schedule.thread_tile_n], dtype=dace.float64, storage=dace.StorageType.Default)
# register_storage_A = nested_state.add_access('register_storage_A')
# register_storage_B = nested_state.add_access('register_storage_B')
# register_storage_C = nested_state.add_access('register_storage_C')
# register_storage_C.setzero = True

# nested_sdfg.add_constant('size_thread_block_tile_m', schedule.thread_block_tile_m)
# nested_sdfg.add_constant('size_thread_block_tile_n', schedule.thread_block_tile_n)
# nested_sdfg.add_constant('size_K_tile', schedule.load_k)
# nested_sdfg.add_constant('num_thread_blocks_m', int(M_example / schedule.thread_block_tile_m))
# nested_sdfg.add_constant('num_thread_blocks_n', int(N_example / schedule.thread_block_tile_n))
# nested_sdfg.add_constant('num_K_tiles', int(K_example / schedule.load_k))
# nested_sdfg.add_constant('size_warp_tile_m', schedule.warp_tile_m)
# nested_sdfg.add_constant('size_warp_tile_n', schedule.warp_tile_n)
# nested_sdfg.add_constant('size_thread_tile_m', schedule.thread_tile_m)
# nested_sdfg.add_constant('size_thread_tile_n', schedule.thread_tile_n)
# nested_sdfg.add_constant('size_thread_tile_k', schedule.thread_tile_k) # = size_K_tile

sdfg.add_constant('size_thread_block_tile_m', schedule.thread_block_tile_m)
sdfg.add_constant('size_thread_block_tile_n', schedule.thread_block_tile_n)
sdfg.add_constant('size_K_tile', schedule.load_k)
sdfg.add_constant('num_thread_blocks_m', int(M_example / schedule.thread_block_tile_m))
sdfg.add_constant('num_thread_blocks_n', int(N_example / schedule.thread_block_tile_n))
sdfg.add_constant('num_K_tiles', int(K_example / schedule.load_k))
sdfg.add_constant('size_warp_tile_m', schedule.warp_tile_m)
sdfg.add_constant('size_warp_tile_n', schedule.warp_tile_n)
sdfg.add_constant('size_thread_tile_m', schedule.thread_tile_m)
sdfg.add_constant('size_thread_tile_n', schedule.thread_tile_n)
sdfg.add_constant('size_thread_tile_k', schedule.thread_tile_k) # = size_K_tile


tasklet = nested_state.add_tasklet('matrix_multiplication', ['__a', '__b'], ['__out'], '__out = (__a * __b)')

# map_entry, map_exit = nested_state.add_map(
#         'matmul_map',
#         dict(i='0:M', j='0:N', k='0:K'),
#         schedule=dace.dtypes.ScheduleType.Default)

# nested_state.add_memlet_path(_A, map_entry, tasklet, dst_conn='__a', memlet=dace.Memlet(f"{_A.data}[i, k]"))
# nested_state.add_memlet_path(_B, map_entry, tasklet, dst_conn='__b', memlet=dace.Memlet(f"{_B.data}[k, j]"))
# nested_state.add_memlet_path(tasklet, map_exit, A_matmul_B_nested_state, src_conn='__out', memlet=dace.Memlet(f"{A_matmul_B_nested_state.data}[i, j]", wcr='(lambda x, y: (x + y))'))

thread_block_grid_map_entry, thread_block_grid_map_exit = nested_state.add_map(
        'Thread_block_grid',
        dict(thread_block_i='0:num_thread_blocks_m', thread_block_j='0:num_thread_blocks_n'),
        schedule=dace.dtypes.ScheduleType.Default)

K_tile_map_entry, K_tile_map_exit = nested_state.add_map(
        'K_tile',
        dict(k_tile='0:num_K_tiles'),
        schedule=dace.dtypes.ScheduleType.Sequential)

warp_map_entry, warp_map_exit = nested_state.add_map(
        'Warp',
        dict(warp_i='0:size_thread_block_tile_m:size_warp_tile_m', warp_j='0:size_thread_block_tile_n:size_warp_tile_n'),
        schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock) # needs to be dace.dtypes.ScheduleType.GPU_ThreadBlock

thread_tile_map_entry, thread_tile_map_exit = nested_state.add_map(
        'Thread_tile',
        dict(thread_tile_i='0:size_warp_tile_m:size_thread_tile_m', thread_tile_j='0:size_warp_tile_n:size_thread_tile_n'),
        schedule=dace.dtypes.ScheduleType.Sequential)

# thread_K_map_entry, thread_K_map_exit = nested_state.add_map(
#         'Thread_K',
#         dict(k='0:size_K_tile'),
#         schedule=dace.dtypes.ScheduleType.Default)

# thread_map_entry, thread_map_exit = nested_state.add_map(
#         'Thread',
#         dict(i='0:size_thread_tile_m', j='0:size_thread_tile_n'),
#         unroll=True,
#         schedule=dace.dtypes.ScheduleType.Default)

map_entry, map_exit = nested_state.add_map(
        'matmul_map',
        dict(i='0:size_thread_tile_m', j='0:size_thread_tile_n', k='0:size_K_tile'),
        schedule=dace.dtypes.ScheduleType.Sequential)

nested_state.add_memlet_path(_A, thread_block_grid_map_entry, K_tile_map_entry, shared_memory_A, memlet=dace.Memlet.simple(_A.data, 'thread_block_i * size_thread_block_tile_m:(thread_block_i+1) * size_thread_block_tile_m,   k_tile * size_K_tile:(k_tile+1) * size_K_tile'))
nested_state.add_memlet_path(shared_memory_A, warp_map_entry, thread_tile_map_entry, map_entry, tasklet, dst_conn='__a', memlet=dace.Memlet(f"{shared_memory_A.data}[i, k]"))

nested_state.add_memlet_path(_B, thread_block_grid_map_entry, K_tile_map_entry, shared_memory_B, memlet=dace.Memlet.simple(_B.data, 'k_tile * size_K_tile:(k_tile+1) * size_K_tile,  thread_block_j * size_thread_block_tile_n:(thread_block_j+1) * size_thread_block_tile_n'))
nested_state.add_memlet_path(shared_memory_B, warp_map_entry, thread_tile_map_entry, map_entry, tasklet, dst_conn='__b', memlet=dace.Memlet(f"{shared_memory_B.data}[k, j]"))

# nested_state.add_memlet_path(_A, thread_block_grid_map_entry, K_tile_map_entry, warp_map_entry, thread_tile_map_entry, map_entry, tasklet, dst_conn='__a', memlet=dace.Memlet(f"{_A.data}[i, k]"))
# nested_state.add_memlet_path(_B, thread_block_grid_map_entry, K_tile_map_entry, warp_map_entry, thread_tile_map_entry, map_entry, tasklet, dst_conn='__b', memlet=dace.Memlet(f"{_B.data}[k, j]"))
nested_state.add_memlet_path(tasklet, map_exit, thread_tile_map_exit, warp_map_exit, K_tile_map_exit, thread_block_grid_map_exit, A_matmul_B_nested_state, src_conn='__out', memlet=dace.Memlet(f"{A_matmul_B_nested_state.data}[i, j]", wcr='(lambda x, y: (x + y))'))

# ### From matrix to shared memory
# nested_state.add_memlet_path(_A,
#                         thread_block_grid_map_entry,
#                         K_tile_map_entry,
#                         shared_memory_A,
#                         # memlet=dace.Memlet(f"{_A.data}[i, j]"))
#                         memlet=dace.Memlet.simple(_A.data, 'thread_block_i*size_thread_block_tile_m:thread_block_i*size_thread_block_tile_m+size_thread_block_tile_m, k_tile*size_K_tile:k_tile*size_K_tile+size_K_tile'))

# nested_state.add_memlet_path(_B,
#                         thread_block_grid_map_entry,
#                         K_tile_map_entry,
#                         shared_memory_B,
#                         # memlet=dace.Memlet(f"{_B.data}[i, j]"))
#                         # memlet=dace.Memlet.simple(_B.data, '0:K, 0:N'))
#                         memlet=dace.Memlet.simple(_B.data, 'k_tile*size_K_tile:k_tile*size_K_tile+size_K_tile, thread_block_j*size_thread_block_tile_n:thread_block_j*size_thread_block_tile_n+size_thread_block_tile_n'))

# ### From shared memory to register storage
# nested_state.add_memlet_path(shared_memory_A,
#                         warp_map_entry,
#                         thread_tile_map_entry,
#                         thread_K_map_entry,
#                         register_storage_A,
#                         # memlet=dace.Memlet(f"{shared_memory_A.data}[i, j]", volume=schedule.thread_block_tile_m * schedule.load_k))
#                         memlet=dace.Memlet.simple(shared_memory_A.data, 'warp_i + thread_tile_i:warp_i + thread_tile_i + size_thread_tile_m, k_tile'))

# nested_state.add_memlet_path(shared_memory_B,
#                         warp_map_entry,
#                         thread_tile_map_entry,
#                         thread_K_map_entry,
#                         register_storage_B,
#                         # memlet=dace.Memlet(f"{shared_memory_B.data}[i, j]", volume=schedule.load_k * schedule.thread_block_tile_n))
#                         memlet=dace.Memlet.simple(shared_memory_B.data, 'k_tile, warp_j + thread_tile_j:warp_j + thread_tile_j + size_thread_tile_n'))

### Delete
# nested_state.add_memlet_path(shared_memory_A,
#                         warp_map_entry,
#                         thread_tile_map_entry,
#                         map_entry,
#                         tasklet,
#                         dst_conn='__a',
#                         # memlet=dace.Memlet.simple(shared_memory_A.data, 'warp_i+thread_tile_i:warp_i+thread_tile_i+size_thread_tile_m, k_tile'))
#                         memlet=dace.Memlet(f"{shared_memory_A.data}[i, k]"))

# nested_state.add_memlet_path(shared_memory_B,
#                         warp_map_entry,
#                         thread_tile_map_entry,
#                         map_entry,
#                         tasklet,
#                         dst_conn='__b',
#                         # memlet=dace.Memlet.simple(shared_memory_B.data, 'k_tile, warp_j+thread_tile_j:warp_j+thread_tile_j+size_thread_tile_n'))
#                         memlet=dace.Memlet(f"{shared_memory_B.data}[k, j]"))

# nested_state.add_memlet_path(tasklet, map_exit, thread_tile_map_exit, warp_map_exit, K_tile_map_exit, thread_block_grid_map_exit, A_matmul_B_nested_state, src_conn='__out', memlet=dace.Memlet(f"{A_matmul_B_nested_state.data}[i, j]", wcr='(lambda x, y: (x + y))'))
### Until here

# ### From register storage to tasklet
# nested_state.add_memlet_path(register_storage_A,
#                         thread_map_entry,
#                         tasklet,
#                         dst_conn='__a',
#                         memlet=dace.Memlet(f"{register_storage_A.data}[i, 0]"))

# nested_state.add_memlet_path(register_storage_B,
#                         thread_map_entry,
#                         tasklet,
#                         dst_conn='__b',
#                         memlet=dace.Memlet(f"{register_storage_B.data}[0, j]"))

# ### Outgoing
# nested_state.add_memlet_path(tasklet,
#                         thread_map_exit,
#                         thread_K_map_exit,
#                         register_storage_C,
#                         src_conn='__out',
#                         # memlet=dace.Memlet.simple(register_storage_C.data, '0:size_thread_tile_m, 0:size_thread_tile_n', wcr_str='(lambda x, y: (x + y))'))
#                         # memlet=dace.Memlet(register_storage_C.data, wcr='(lambda x, y: (x + y))'))
#                         memlet=dace.Memlet(f"{register_storage_C.data}[i, j]", wcr='(lambda x, y: (x + y))'))
#                         # memlet=dace.Memlet(f"{register_storage_C.data}[i, j]", wcr='(lambda x, y: (x + y))', volume=size_thread_tile_m * size_thread_tile_n))
#                         # memlet=dace.Memlet(wcr='(lambda x, y: (x + y))', subset=f"{register_storage_C.data}[i, j]"))
#                         # memlet=dace.Memlet(f"{tasklet.data}[i, j]"))

# nested_state.add_memlet_path(register_storage_C,
#                         thread_tile_map_exit,
#                         warp_map_exit,
#                         K_tile_map_exit,
#                         thread_block_grid_map_exit,
#                         A_matmul_B_nested_state,
#                         # memlet=dace.Memlet(f"{A_matmul_B_nested_state.data}[], wcr='(lambda x, y: (x + y))'))                        
#                         memlet=dace.Memlet.simple(A_matmul_B_nested_state.data, '0:M, 0:N', wcr_str='(lambda x, y: (x + y))'))                 
#                         # memlet=dace.Memlet.simple(A_matmul_B_nested_state.data, '0:M, 0:N', wcr_str='(lambda x, y: (x + y))', num_accesses=M * N))

nested_sdfg.fill_scope_connectors()
sdfg.fill_scope_connectors()
sdfg.save('sdfg_api.sdfg')
nested_sdfg.validate()
sdfg.validate()

sdfg.arg_names = ['A', 'B', 'C', 'alpha', 'beta']

sdfg.save('sdfg_api.sdfg')
csdfg = sdfg.compile()

helpers.print_info("Verifying results...", False)

A = np.random.rand(M_example, K_example).astype(np.float64)
B = np.random.rand(K_example, N_example).astype(np.float64)
C = np.zeros((M_example, N_example)).astype(np.float64)
result = np.empty((M_example, N_example)).astype(np.float64)
alpha = 1.0
beta = 1.0

def matmul(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N], alpha: dace.float64, beta: dace.float64):
    return alpha * (A @ B) + beta * C

C_correct = matmul(A=A, B=B, C=C, alpha=alpha, beta=beta)
print("Launching sdfg...")
C_test = csdfg(A=A, B=B, C=C, alpha=alpha, beta=beta, M=M_example, N=N_example, K=K_example, result=result)
print(result)
print(result[0][0])
print(result[0][1])
print(result[0][2])
print(result[0][3])
print('--')
# print(C_test)
# print('--')
# print(result)

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
        print("%.2f" % result[i][j], end=" ")
    print()


if areSame(C_correct, result):
    helpers.print_success("The SDFG is correct!", False)
else:
    helpers.print_error("The SDFG is incorrect!", False)