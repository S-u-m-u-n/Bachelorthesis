import sys
import subprocess
import math
from argparse import ArgumentParser
from csv import DictReader
import numpy as np
import sympy as sy
from tqdm import tqdm
import dace
from dace.transformation.interstate import GPUTransformSDFG
from dace.transformation.dataflow import MapTiling, InLocalStorage, MapExpansion, MapCollapse, StripMining, DoubleBuffering, Vectorization, MapExpansion, AccumulateTransient
from dace.transformation import helpers as xfutil
from dace.subsets import Range
import helpers
import warnings

def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next((n, state) for n, state in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)  

def find_map_by_name(sdfg: dace.SDFG, name: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next((n, state) for n, state in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and name == n.label)

A = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    [16, 17, 18, 19],
    [20, 21, 22, 23],
    [24, 25, 26, 27],
    [28, 29, 30, 31]
])

B = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
])

@dace.program
def identity(A: dace.int64[8, 4], B: dace.int64[8, 4]):
    return A * B

sdfg = identity.to_sdfg()
sdfg.expand_library_nodes()
sdfg.apply_transformations(GPUTransformSDFG)

#####################################################################
### Threadblock Tile
sdfg.save('identity.sdfg')
gemm, state = find_map_by_param(sdfg, "__i0")
xfutil.tile(state.parent, gemm, True, True, __i0=8, __i1=4)

entry_outer, state = find_map_by_param(state.parent, "tile___i0")
entry_inner, state = find_map_by_param(state.parent, "tile___i1")
MapCollapse.apply_to(state.parent, _outer_map_entry=entry_outer, _inner_map_entry=entry_inner)
entry_inner, state = find_map_by_param(state, "__i0")
entry_inner._map.schedule = dace.ScheduleType.GPU_ThreadBlock

#####################################################################
### local storage (shared memory) for loading threadblock_tiles of A and B
entry_outer, state = find_map_by_param(state.parent, "tile___i0")
shared_memory_A = InLocalStorage.apply_to(state.parent, dict(array='_a'), node_a=entry_outer, node_b=entry_inner)
shared_memory_B = InLocalStorage.apply_to(state.parent, dict(array='_b'), node_a=entry_outer, node_b=entry_inner)
state.parent.arrays[shared_memory_A.data].storage = dace.StorageType.GPU_Shared
state.parent.arrays[shared_memory_B.data].storage = dace.StorageType.GPU_Shared

#####################################################################
### Warp Tile
gemm, state = find_map_by_param(state.parent, "__i0")
xfutil.tile(state.parent, gemm, True, True, __i0=8, __i1=4)
entry_outer, state = find_map_by_param(state.parent, "tile1___i0")
entry_inner, state = find_map_by_param(state.parent, "tile1___i1")
MapCollapse.apply_to(state.parent, _outer_map_entry=entry_outer, _inner_map_entry=entry_inner)

#####################################################################
### Thread Tile
gemm, state = find_map_by_param(state.parent, "__i0")
xfutil.tile(state.parent, gemm, True, True, __i0=1, __i1=1)
entry_outer, state = find_map_by_param(state.parent, "tile2___i0")
entry_inner, state = find_map_by_param(state.parent, "tile2___i1")
MapCollapse.apply_to(state.parent, _outer_map_entry=entry_outer, _inner_map_entry=entry_inner)
sdfg.save('identity.sdfg')

# Unroll microkernel maps
entry, state = find_map_by_param(state.parent, "__i0")
entry.map.unroll = True

helpers.print_info('Applying SWIZZLE_thread_tile', False)
entry, state = find_map_by_param(state.parent, "__i0")
warp_tile_height = 8
warp_tile_width = 4

bitwise_and = sy.Function('bitwise_and')
bitwise_or = sy.Function('bitwise_or')
right_shift = sy.Function('right_shift')
def SWIZZLE_x(idx): # LaneIdx
    # return ((idx & (warp_tile_height * warp_tile_width // 2)) >> (warp_tile_width - 1)) | (idx & 1)
    return bitwise_or(
            right_shift(
                bitwise_and(idx, (warp_tile_height * warp_tile_width // 2)),
                (warp_tile_width - 1)),
            bitwise_and(idx, 1)
            )
def SWIZZLE_y(idx): # LaneIdy
    # return (idx >> 1) & (warp_tile_height - 1)
    return bitwise_and(
            idx // 2,
            warp_tile_height - 1
            )

def SWIZZLE_x_int(idx): # LaneIdx
    return ((idx & (warp_tile_height * warp_tile_width // 2)) >> (warp_tile_width - 1)) | (idx & 1)

def SWIZZLE_y_int(idx): # LaneIdy
    return (idx >> 1) & (warp_tile_height - 1)


# ... apply SWIZZLE_thread_block transformations
current_mapping_x = state.out_edges(entry)[0].data.subset
current_mapping_y = state.out_edges(entry)[1].data.subset
print(current_mapping_x)
print(current_mapping_y)
print()
# Quote from Neville's thesis, p. 11: "threads are only launched in the x dimension (threadIdx.y and threadIdx.z are always 1)
print("Thread tiles in a warp before swizzling:")
for x in range (0, warp_tile_height):
    print("-" * 3 * warp_tile_height + "-")
    for y in range (0, warp_tile_width):
        print("| " + str(warp_tile_width * x + y) + " ", end="")
    print("|")
print("-" * 3 * warp_tile_height + "-")

swizzled_idx = np.empty(warp_tile_height * warp_tile_width)
for x in range (0, warp_tile_height):
    for y in range (0, warp_tile_width):
        idx = warp_tile_width * x + y
        # print(str(idx) + " -> " + str(SWIZZLE_x(idx)) + ", " +  str(SWIZZLE_y(idx)) + " = " + str(warp_tile_width * SWIZZLE_y(idx) + SWIZZLE_x(idx)))
        # print(idx)
        # print(type(idx))
        # print(SWIZZLE_x_int(idx))
        # print(SWIZZLE_y_int(idx))
        # print(warp_tile_width * SWIZZLE_y(idx) + SWIZZLE_x(idx))
        swizzled_idx[idx] = warp_tile_width * SWIZZLE_y_int(idx) + SWIZZLE_x_int(idx)

print("Thread tiles in a warp after swizzling:")
for x in range (0, warp_tile_height):
    print("-" * 3 * warp_tile_height + "-")
    for y in range (0, warp_tile_width):
        idx = warp_tile_width * x + y
        print("| " + str(np.where(swizzled_idx == idx)[0][0]) + " ", end="")
    print("|")
print("-" * 3 * warp_tile_height + "-")

entry_warp, state = find_map_by_param(state.parent, "tile1___i0")
warp_x = state.out_edges(entry_warp)[0].data.subset[0][0] # = tile1___i0
warp_y = state.out_edges(entry_warp)[1].data.subset[1][0] # = tile1___i1
print(warp_x)
print(warp_y)

# we want to remove the warp offset (tile1___i0 and tile1___i1 in this case), because the thread_tile swizzling should be independent of the warp
old_id_x = (current_mapping_x.ndrange()[0][0] - warp_x) / 1
old_id_y = (current_mapping_y.ndrange()[1][0] - warp_y) / 1
old_id = warp_tile_height * old_id_x + old_id_y
print(old_id)
new_id_x = SWIZZLE_x(old_id)
print("SWIZZLE: " + str(old_id_x) + " is remapped to " + str(new_id_x))
new_id_y = SWIZZLE_y(old_id)
print("SWIZZLE: " + str(old_id_y) + " is remapped to " + str(new_id_y))

state.out_edges(entry)[0].data.subset = Range([
    (warp_x + new_id_x,
    warp_x + new_id_x + 1 - 1,
    current_mapping_x.ndrange()[0][2]),
    (current_mapping_x.ndrange()[1][0],
    current_mapping_x.ndrange()[1][1],
    current_mapping_x.ndrange()[1][2])
])
# print(state.out_edges(entry)[0].data.subset)

state.out_edges(entry)[1].data.subset = Range([
    (current_mapping_y.ndrange()[0][0],
    current_mapping_y.ndrange()[0][1],
    current_mapping_y.ndrange()[0][2]),
    (warp_y + new_id_y,
    warp_y + new_id_y + 1 - 1,
    current_mapping_y.ndrange()[0][2])
])
# print(state.out_edges(entry)[1].data.subset)
helpers.print_success("Successfully applied thread SWIZZLE.", False)

sdfg.save('identity_final.sdfg')
helpers.print_info('Compiling sdfg...', False)
csdfg = sdfg.compile()
helpers.print_success("Successfully compiled SDFG.", False)

C_test = csdfg(A=A, B=B)

helpers.print_info("Verifying results...", False)
C_correct = identity(A=A, B=B)

# Can replace this with np.allclose(A, B)
def areSame(A,B):
    for i in range(8):
        for j in range(4):
            diff = math.fabs(A[i][j] - B[i][j])
            # helpers.print_info("(" + str(i) + ", " + str(j) + ")", False)
            # helpers.print_info("Comparing " + str(B[i][j]) + " to " + str(A[i][j]))
            # helpers.print_info("Difference = " + str(diff))
            if (diff > 0.000001):
                helpers.print_error("Error at position (" + str(i) + ", " + str(j) + "): matrices are not equal! Difference is: " + str(diff), False)
                helpers.print_error(str(B[i][j]) + " should be " + str(A[i][j]), False)
                print()
                return False
    return True
    
helpers.print_info("SDFG result: ", False)
print()
for i in range(8):
    for j in range(4):
        print("%.2f" % C_test[i][j], end=" ")
    print()

print()
print()
helpers.print_info("Correct result: ", False)
for i in range(8):
    for j in range(4):
        print("%.2f" % C_correct[i][j], end=" ")
    print()

if areSame(C_correct, C_test):
    helpers.print_success("The SDFG is correct!", False)
else:
    helpers.print_error("The SDFG is incorrect!", False)