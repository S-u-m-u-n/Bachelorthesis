import dace
import numpy as np
from dace.transformation.interstate import GPUTransformSDFG
from dace.transformation.dataflow import MapTiling

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')

sdfg = dace.SDFG('matmul')
sdfg.add_array('A', shape=[M, K], dtype=dace.float64)
sdfg.add_array('B', shape=[K, N], dtype=dace.float64)
sdfg.add_transient('tmp', shape=[M, N, K], dtype=dace.float64)
sdfg.add_array('C', shape=[M, N], dtype=dace.float64)

state = sdfg.add_state()

def mainstate(state, src_node, dst_node):
    # Creates Map (entry and exit nodes), Tasklet node, and connects the three
    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
        '%s_to_%s' % (src_node.data, dst_node.data),                                         # name
        dict(i='0:M', j='0:N', k='0:K'),                                                                         # map range
        dict(inp_A=dace.Memlet('A[i, k]'), inp_B=dace.Memlet('B[k, j]')),          # input memlets
        'out >> tmp(1, lambda a,b: a+b)[i, j]',                                                             # code
        dict(out=dace.Memlet('tmp[i,j,k]'))                                                            # output memlet
    )
    
    #######################
    # Add external connections from map to arrays

    # Add input path (src->entry) with the overall memory accessed
    # NOTE: This can be inferred automatically by the system
    #       using external_edges=True in `add_mapped_tasklet`
    #       or using the `propagate_edge` function.
    state.add_edge(
        src_node, None,
        map_entry, None,
        memlet=dace.Memlet('A[0:M, 0:K]'))

    state.add_edge(
        src_node, None,
        map_entry, None,
        memlet=dace.Memlet('B[0:K, 0:N]'))
    
    # Add output path (exit->dst)
    state.add_edge(
        map_exit, None,
        dst_node, None,
        memlet=dace.Memlet('C[0:M, 0:N]'))

A = state.add_read('A')
B = state.add_read('B')
tmp = state.add_access('tmp')
C = state.add_write('C')

mainstate(state, A, tmp)
mainstate(state, B, tmp)
mainstate(state, tmp, C)

# sdfg.apply_transformations(GPUTransformSDFG)
# sdfg.apply_transformations(MapTiling)
sdfg.save('explicit.sdfg')
