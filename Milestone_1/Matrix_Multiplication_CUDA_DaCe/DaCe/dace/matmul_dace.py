import dace
import numpy as np
from dace.transformation.interstate import GPUTransformSDFG, StateFusion
from dace.transformation.dataflow import MapTiling, InLocalStorage, MapExpansion, MapCollapse
from dace.transformation.optimizer import Optimizer
from dace.transformation import helpers as xfutil

def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next((n, state) for n, state in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)

def find_map_by_name(sdfg: dace.SDFG, name: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next((n, state) for n, state in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and name == n.label)

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')

@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N]):
    return A @ B

sdfg = matmul.to_sdfg()
sdfg.save('matmul_initial.sdfg')
sdfg.expand_library_nodes()

sdfg.apply_transformations(GPUTransformSDFG)
gemm, state = find_map_by_name(sdfg, "gemm_map")
xfutil.tile(state.parent, gemm, True, True, __i0 =128, __i1 = 128, __i2 = 8)
sdfg.apply_transformations(MapCollapse)
btile, state = find_map_by_param(sdfg, '__i0')
btile._map.schedule = dace.ScheduleType.GPU_ThreadBlock

node_a, state_a = find_map_by_param(sdfg, "tile___i2")
node_b, state_b = find_map_by_param(sdfg, "__i0")
smem_a = InLocalStorage.apply_to(state_a.parent, dict(array='trans__a', create_array=True), node_a=node_a, node_b=node_b)
smem_b = InLocalStorage.apply_to(state_b.parent, dict(array='trans__b', create_array=True), node_a=node_a, node_b=node_b)
sdfg.save('matmul_inlocal.sdfg')
sdfg.compile()


# print()
# print(list(sdfg.arrays_recursive()))
# print()
# print(list(sdfg.all_nodes_recursive()))
# print()
# print(smem_a)
# print(smem_a.data)
# print(sdfg.arrays[smem_a.data])
# print()

# sdfg.arrays[smem_a.data].storage = dace.StorageType.GPU_Shared
# sdfg.arrays[smem_b.data].storage = dace.StorageType.GPU_Shared
# sdfg.save('matmul_shared.sdfg')








# sdfg.apply_transformations(MapCollapse)
# sdfg.save('matmul.sdfg')
# sdfg.compile()


# gemm, state = find_map_by_name(sdfg, "gemm_map")
# print(type(gemm))
# print(gemm)
# print(type(gemm.params))
# print(gemm.params)
# sdfg.save('matmul_preInLocalStorage.sdfg')


# sdfg.apply_transformations(MapCollapse)

# node_a, state_a = find_map_by_param(sdfg, "tile___i0")
# node_b, state_b = find_map_by_param(sdfg, "tile___i2")



# node_a, state_a = find_map_by_param(sdfg, "tile___i2")
# node_b, state_b = find_map_by_param(sdfg, "__i0")
# smem_a = InLocalStorage.apply_to(state_a.parent, dict(array='trans_gpu__a', create_array=True), node_a=node_a, node_b=node_b)
# smem_b = InLocalStorage.apply_to(state_b.parent, dict(array='trans_gpu__b', create_array=True), node_a=node_a, node_b=node_b)
# smem_a = InLocalStorage.apply_to(state_a.parent, dict(array='gpu_A', create_array=True), node_a=node_a, node_b=node_b)
# smem_b = InLocalStorage.apply_to(state_a.parent, dict(array='gpu_B', create_array=True), node_a=node_a, node_b=node_b)



# sdfg.apply_transformations(GPUTransformSDFG)
# gemm, state = find_map_by_name(sdfg, "gemm_map")
# xfutil.tile(state.parent, gemm, True, True, __i0 =128, __i1 = 128, __i2 = 8)
# GPUTransformSDFG.apply_to(state.parent, node_a=gemm)
