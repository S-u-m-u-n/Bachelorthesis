import sys
import subprocess
import math
from argparse import ArgumentParser
from csv import DictReader
from Schedule import Schedule
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

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')

@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N], alpha: dace.float64, beta: dace.float64):
    return alpha * (A @ B) + beta * C

def create_sdfg(schedule) -> None:
    sdfg = matmul.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_transformations(GPUTransformSDFG)
    sdfg.save('sdfg_start.sdfg')

    #####################################################################
    ### Threadblock Tile
    gemm, state = find_map_by_name(sdfg, "gemm_map")
    # xfutil.tile(state.parent, gemm, True, True, __i0=schedule.thread_block_tile_m, __i1=schedule.thread_block_tile_n, __i2=schedule.load_k)
    divides_evenly = False
    if M % schedule.thread_block_tile_m == 0:
        divides_evenly = True
    StripMining.apply_to(state.parent,
            dict(tiling_type=dace.TilingType.NumberOfTiles,
            tile_size=M / schedule.thread_block_tile_m,
            dim_idx=0,
            divides_evenly=divides_evenly,
            ),
            _map_entry=gemm
    )
    sdfg.save('sdfg_tiled_1.sdfg')
    divides_evenly = False
    if N % schedule.thread_block_tile_n == 0:
        divides_evenly = True
    StripMining.apply_to(state.parent,
            dict(tiling_type=dace.TilingType.NumberOfTiles,
            tile_size=N / schedule.thread_block_tile_n,
            dim_idx=1,
            divides_evenly=divides_evenly,
            ),
            _map_entry=gemm
    )
    sdfg.save('sdfg_tiled_2.sdfg')
    if K % schedule.load_k == 0:
        divides_evenly = True
    StripMining.apply_to(state.parent,
            dict(tiling_type=dace.TilingType.NumberOfTiles,
            tile_size=K / schedule.load_k,
            dim_idx=2,
            divides_evenly=divides_evenly,
            ),
            _map_entry=gemm
    )
    entry_outer, state = find_map_by_param(state.parent, "tile___i0")
    entry_inner, state = find_map_by_param(state.parent, "tile___i1")
    MapCollapse.apply_to(state.parent, _outer_map_entry=entry_outer, _inner_map_entry=entry_inner)
    entry_inner, state = find_map_by_param(state.parent, "__i0")
    entry_inner._map.schedule = dace.ScheduleType.GPU_ThreadBlock

    #####################################################################
    ### local storage (shared memory) for loading threadblock_tiles of A and B
    entry_outer, state = find_map_by_param(state.parent, "tile___i2")
    shared_memory_A = InLocalStorage.apply_to(state.parent, dict(array='_a'), node_a=entry_outer, node_b=entry_inner)
    shared_memory_B = InLocalStorage.apply_to(state.parent, dict(array='_b'), node_a=entry_outer, node_b=entry_inner)
    state.parent.arrays[shared_memory_A.data].storage = dace.StorageType.GPU_Shared
    state.parent.arrays[shared_memory_B.data].storage = dace.StorageType.GPU_Shared
    # Todo: make size constant

    #####################################################################
    ### Warp Tile
    gemm, state = find_map_by_param(state.parent, "__i0")
    xfutil.tile(state.parent, gemm, True, True, __i0=schedule.warp_tile_m, __i1=schedule.warp_tile_n)
    entry_outer, state = find_map_by_param(state.parent, "tile1___i0")
    entry_inner, state = find_map_by_param(state.parent, "tile1___i1")
    sdfg.save('sdfg_tiled_3.sdfg')
    MapCollapse.apply_to(state.parent, _outer_map_entry=entry_outer, _inner_map_entry=entry_inner)
    sdfg.save('sdfg_tiled_4.sdfg')

    #####################################################################
    ### Thread Tile
    gemm, state = find_map_by_param(state.parent, "__i0")
    xfutil.tile(state.parent, gemm, True, True, __i0=schedule.thread_tile_m, __i1=schedule.thread_tile_n)
    entry_outer, state = find_map_by_param(state.parent, "tile2___i0")
    entry_inner, state = find_map_by_param(state.parent, "tile2___i1")
    MapCollapse.apply_to(state.parent, _outer_map_entry=entry_outer, _inner_map_entry=entry_inner)

    #####################################################################
    ### local storage (registers) for loading thread_tiles of A and B
    entry, state = find_map_by_param(state.parent, "__i0")
    # Reorder internal map to "k, i, j"
    xfutil.permute_map(entry, [2, 0, 1])
    # expand the three dimensions of the thread_tile...
    MapExpansion.apply_to(state.parent, map_entry=entry)
    # ...then collapse the inner two dimensions again...
    entry_outer, state = find_map_by_param(state.parent, "__i0")
    entry_inner, state = find_map_by_param(state.parent, "__i1")
    MapCollapse.apply_to(state.parent, _outer_map_entry=entry_outer, _inner_map_entry=entry_inner) 
    # ...and finally apply local storage transformation
    entry_outer, state = find_map_by_param(state.parent, "__i2")
    entry_inner, state = find_map_by_param(state.parent, "__i0")
    InLocalStorage.apply_to(state.parent, dict(array='trans__a'), node_a=entry_outer, node_b=entry_inner)
    InLocalStorage.apply_to(state.parent, dict(array='trans__b'), node_a=entry_outer, node_b=entry_inner)

    #####################################################################
    ### local storage (registers) for loading thread_tiles of C
    map_exit = state.exit_node(entry_outer)
    sdfg.save('sdfg_tiled_5.sdfg')
    outer, state = find_map_by_param(state.parent, "tile2___i0")
    outer_map_exit = state.exit_node(outer)
    AccumulateTransient.apply_to(state.parent, map_exit=map_exit, outer_map_exit=outer_map_exit)
    # Set C tile to zero on allocation
    c_access = next(n for n in state.data_nodes() if n.data == 'trans__c')
    c_access.setzero = True

    # Unroll microkernel maps
    entry, state = find_map_by_param(state.parent, "__i0")
    entry.map.unroll = True

    sdfg.save('sdfg_tiled_4.sdfg')

    #####################################################################
    ### Split K
    if args.split_k and schedule.split_k > 1:
        sdfg.save('sdfg_pre_split_k.sdfg')
        if not args.quiet:
            helpers.print_info('Applying Split K with Split_K = ' + str(schedule.split_k) + " ....", args.colorless)
        entry_outer, state = find_map_by_param(state.parent, "tile___i0")
        entry_inner, state = find_map_by_param(state.parent, "tile___i2")
        MapCollapse.apply_to(state.parent, _outer_map_entry=entry_outer, _inner_map_entry=entry_inner)
        entry, state = find_map_by_param(state.parent, "tile___i2")
        # entry.schedule = dace.ScheduleType.GPU_Device
        divides_evenly = False
        if K % schedule.split_k == 0:
            divides_evenly = True
        entry_new = StripMining.apply_to(state.parent,
                    dict(new_dim_prefix="SPLIT_K",
                    # tiling_type=dace.TilingType.NumberOfTiles,
                    # tile_size=schedule.split_k, # Split K tiles
                    tile_size= K / schedule.split_k, # Split K tiles
                    dim_idx=2, # K dimension
                    divides_evenly=divides_evenly,
                    ),
                    _map_entry=entry
        )
        # entry_new.schedule = dace.ScheduleType.Sequential

        # We need to modify the memlets due to a bug - probably not necessary in the future
        entry, state = find_map_by_param(state.parent, "SPLIT_K_tile___i2")

        current_mapping_x = state.out_edges(entry)[0].data.subset
        current_mapping_y = state.out_edges(entry)[1].data.subset
        state.out_edges(entry)[0].data.subset = Range([
            (current_mapping_x.ndrange()[0][0],
            current_mapping_x.ndrange()[0][1] - schedule.thread_block_tile_m + 1,
            current_mapping_x.ndrange()[0][2]),
            (current_mapping_x.ndrange()[1][0],
            current_mapping_x.ndrange()[1][1] - schedule.load_k + 1,
            current_mapping_x.ndrange()[1][2])
        ])

        state.out_edges(entry)[1].data.subset = Range([
            (current_mapping_y.ndrange()[0][0],
             current_mapping_y.ndrange()[0][1] - schedule.load_k + 1,
             current_mapping_y.ndrange()[0][2]),
            (current_mapping_y.ndrange()[1][0],
            current_mapping_y.ndrange()[1][1] - schedule.thread_block_tile_n + 1,
            current_mapping_y.ndrange()[1][2])
        ])
        # expand the three dimensions of the thread_tile...
        entry, state = find_map_by_param(state.parent, "tile___i0")
        MapExpansion.apply_to(state.parent, map_entry=entry)
        # ...then collapse the first two dimensions again...
        entry_outer, state = find_map_by_param(state.parent, "tile___i0")
        entry_inner, state = find_map_by_param(state.parent, "tile___i1")
        MapCollapse.apply_to(state.parent, _outer_map_entry=entry_outer, _inner_map_entry=entry_inner) 

        if not args.quiet:
            helpers.print_success("Successfully applied Split K.", args.colorless)
        # Todo: Modify tasklet
        # Todo: make a reduction in the end
        #     MapReduceFusion.apply_to(state.parent)

    #####################################################################
    ### SWIZZLE_thread_block
    if args.swizzle_thread_blocks and schedule.SWIZZLE_thread_block > 1:
        sdfg.save('sdfg_pre_swizzle_thread_block.sdfg', args.colorless)
        if not args.quiet:
            helpers.print_info('Applying SWIZZLE_thread_block with SWIZZLE_thread_block = ' + str(schedule.SWIZZLE_thread_block) + " ....")
        entry, state = find_map_by_param(state.parent, "tile___i2")
        def SWIZZLE_x(x):
            return x // schedule.SWIZZLE_thread_block # // stands for floor division
        def SWIZZLE_y(y, x):
            return (y * schedule.SWIZZLE_thread_block) + (x % schedule.SWIZZLE_thread_block)
        # ... apply SWIZZLE_thread_block transformations
        current_mapping_x = state.out_edges(entry)[0].data.subset
        current_mapping_y = state.out_edges(entry)[1].data.subset
        print(current_mapping_x)
        print(current_mapping_y)
        print()
        print("Thread block grid before swizzling:")
        for x in range (0, math.ceil(M / schedule.thread_block_tile_m)):
            print("-" * 9 * math.ceil(M / schedule.thread_block_tile_m) + "-")
            for y in range (0, math.ceil(N / schedule.thread_block_tile_n)):
                print("| (" + str(x) + ", " + str(y) + ") ", end="")
            print("|")
        print("-" * 9 * math.ceil(M / schedule.thread_block_tile_m) + "-")

        print("Thread block grid after swizzling:")
        for x in range (0, math.ceil(M / schedule.thread_block_tile_m)):
            print("-" * 9 * math.ceil(M / schedule.thread_block_tile_m) + "-")
            for y in range (0, math.ceil(N / schedule.thread_block_tile_n)):
                print("| (" + str(SWIZZLE_x(x)) + ", " + str(SWIZZLE_y(y, x)) + ") ", end="")
            print("|")
        print("-" * 9 * math.ceil(M / schedule.thread_block_tile_m) + "-")

        old_id_x = current_mapping_x.ndrange()[0][0] / schedule.thread_block_tile_m
        new_id_x = SWIZZLE_x(old_id_x) * schedule.thread_block_tile_m
        print("SWIZZLE: " + str(old_id_x) + " is remapped to " + str(new_id_x / schedule.thread_block_tile_m))
        old_id_y = current_mapping_y.ndrange()[1][0] / schedule.thread_block_tile_n
        new_id_y = SWIZZLE_y(old_id_y, old_id_x) * schedule.thread_block_tile_n
        print("SWIZZLE: " + str(old_id_y) + " is remapped to " + str(new_id_y  / schedule.thread_block_tile_n))

        state.out_edges(entry)[0].data.subset = Range([
            (new_id_x,
            new_id_x + schedule.thread_block_tile_m - 1,
            current_mapping_x.ndrange()[0][2]),
            (current_mapping_x.ndrange()[1][0],
            current_mapping_x.ndrange()[1][1],
            current_mapping_x.ndrange()[1][2])
        ])

        state.out_edges(entry)[1].data.subset = Range([
            (current_mapping_y.ndrange()[0][0],
             current_mapping_y.ndrange()[0][1],
             current_mapping_y.ndrange()[0][2]),
            (new_id_y,
            new_id_y + schedule.thread_block_tile_n - 1,
            current_mapping_y.ndrange()[1][2])
        ])
        if not args.quiet:
            helpers.print_success("Successfully applied thread block SWIZZLE.", args.colorless)
    
    #####################################################################
    ### SWIZZLE_thread_tile
    # class BitwiseAnd(sy.Function):
    #     nargs = 2
    #     @classmethod
    #     def eval(cls, x, y):
    #         print("Evaluting: " + str(x) + " & " + str(y))
    #         if x == 0 or y == 0:
    #             return 1
    #         return 0
    
    # class BitwiseOr(sy.Function):
    #     nargs = 2
    #     @classmethod
    #     def eval(cls, x, y):
    #         if x == 1 or y == 1:
    #             return 0
    #         return 1

    # class RightShift(sy.Function):
    #     nargs = 2
    #     @classmethod
    #     def eval(cls, x, y):
    #         for i in range(0, y):
    #             x /= 2
    #         return x

    if args.swizzle_threads and schedule.SWIZZLE_thread_tile == True:
        sdfg.save('sdfg_pre_swizzle_thread_tile.sdfg')
        if not args.quiet:
            helpers.print_info('Applying SWIZZLE_thread_tile with SWIZZLE_thread_tile = ' + str(schedule.SWIZZLE_thread_tile) + " ....", args.colorless)
        entry, state = find_map_by_param(state.parent, "__i2")
        warp_tile_width = math.ceil(schedule.warp_tile_n / schedule.thread_tile_n)
        warp_tile_height = math.ceil(schedule.warp_tile_m / schedule.thread_tile_m)
        print(warp_tile_width)
        print(warp_tile_height)

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

        # we want to remove the warp offset (tile1___i0 and tile1___i1 in this case), because the thread_tile swizzling should be independent of the warp
        old_id_x = (current_mapping_x.ndrange()[0][0] - warp_x) / schedule.thread_tile_m
        old_id_y = (current_mapping_y.ndrange()[1][0] - warp_y) / schedule.thread_tile_n
        old_id = warp_tile_height * old_id_x + old_id_y
        print(old_id)
        new_id_x = SWIZZLE_x(old_id)
        print("SWIZZLE: " + str(old_id_x) + " is remapped to " + str(new_id_x))
        new_id_y = SWIZZLE_y(old_id)
        print("SWIZZLE: " + str(old_id_y) + " is remapped to " + str(new_id_y))

        state.out_edges(entry)[0].data.subset = Range([
            (warp_x + new_id_x,
            warp_x + new_id_x + schedule.thread_tile_m - 1,
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
            warp_y + new_id_y + schedule.thread_tile_n - 1,
            current_mapping_y.ndrange()[0][2])
        ])
        # print(state.out_edges(entry)[1].data.subset)
        if not args.quiet:
            helpers.print_success("Successfully applied thread SWIZZLE.", args.colorless)

    # #####################################################################
    # ### Vectorization
    if args.vectorization and schedule.load_k > 1:
        sdfg.save('sdfg_pre_vectorization.sdfg')
        # 128 bits maximum
        if not args.quiet:
            helpers.print_info('Applying Vectorization....', args.colorless)
        if schedule.load_k == 2:
            vector_length = 2
        elif schedule.load_k >= 4:
            vector_length = 2

        # state.parent.apply_transformations_repeated(Vectorization, dict(vector_len=vector_length, preamble=False, postamble=False))
        entry, state = find_map_by_param(state.parent, "__i0")
        Vectorization.apply_to(state.parent,
                        dict(vector_len=vector_length, preamble=False, postamble=False),
                        _map_entry=entry,
                        _tasklet=state.out_edges(entry)[0].dst,
                        _map_exit=state.exit_node(entry))
                        
        # Vectorization.apply_to(state.parent,
        #                 dict(vector_len=vector_length, preamble=False, postamble=False),
        #                 _map_entry=entry,
        #                 _tasklet=state.out_edges(entry)[1].dst,
        #                 _map_exit=state.exit_node(entry))
        if not args.quiet:
            helpers.print_success("Successfully applied vectorization.", args.colorless)
   
    #####################################################################
    ### Double Buffering (on shared memory)
    if args.double_buffering and schedule.double_buffering == True:
        sdfg.save('sdfg_pre_double_buffering.sdfg')
        if not args.quiet:
            helpers.print_info('Applying Double Buffering....', args.colorless)
        entry, state = find_map_by_param(state, "tile___i2")
        DoubleBuffering.apply_to(state.parent, _map_entry=entry, _transient=shared_memory_A)
        if not args.quiet:
            helpers.print_success("Successfully applied double buffering.", args.colorless)

    sdfg.save('sdfg_final.sdfg')
    if not args.quiet:
        helpers.print_info('Compiling sdfg...', args.colorless)
    csdfg = sdfg.compile()
    if not args.quiet:
        helpers.print_success("Successfully compiled SDFG.", args.colorless)
    return csdfg

#####################################################################
### Main function
if __name__ == "__main__":
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
                        dest='verification',
                        help="Verify results. Default: True",
                        action="store_false",
                        default=True)
    parser.add_argument('--all-optimizations',
                        dest='all_optimizations',
                        help="Use all possible optimizations",
                        action="store_true",
                        default=False)
    parser.add_argument('--split-k',
                        dest='split_k',
                        help="Use Split K",
                        action="store_true",
                        default=False)
    parser.add_argument('--swizzle-thread-blocks',
                        dest='swizzle_thread_blocks',
                        help="Use swizzle on the thread blocks",
                        action="store_true",
                        default=False)
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

    if args.all_optimizations:
        args.split_k = True
        args.swizzle_thread_blocks = True
        args.swizzle_threads = True
        args.vectorization = True
        args.double_buffering = True
    if args.precision == '64':
        np_dtype = np.float64
    else:
        helpers.print_error("Bit precisions different from 64 are currently unsupported", args.colorless)
        raise NotImplementedError

    if not args.quiet:
        helpers.print_info("Using this dace: " + str(dace.__file__), args.colorless)
    else:
        warnings.filterwarnings('ignore', category=UserWarning)

    # Define set of possible values for schedule generator
    load_k_possible = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    threadtiles_possible = [1, 2, 4, 8]
    # threadtiles_possible = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    M=np.int32(args.M)
    N=np.int32(args.N)
    K=np.int32(args.K)
    # A = np.random.rand(M, K).astype(np_dtype)
    # B = np.random.rand(K, N).astype(np_dtype)
    # C = np.zeros((M, N)).astype(np_dtype)
    alpha = np.float64(args.alpha)
    beta = np.float64(args.beta)

    if args.version == 'dace':
        ########################################################
        # 2. Find best schedule
        if not args.quiet:
            helpers.print_info("Phase 2/3: Finding best schedule...", args.colorless)
        # schedule = find_best_schedule(load_k_possible, threadtiles_possible, device.registers_per_warp, device.registers_per_thread_block, device.threads_per_warp, device.warps_per_SM, device.SMs, device.total_compute_cores)
        # best_schedule = find_best_schedule(load_k_possible, threadtiles_possible)

        best_schedule = Schedule(load_k=8, thread_tile_m=8, thread_tile_n=8, thread_tile_k=8, warp_tile_m=64, warp_tile_n=32,
                        thread_block_tile_m=64, thread_block_tile_n=32, thread_block_tile_k=640,
                        SWIZZLE_thread_block=1, SWIZZLE_thread_tile=True, split_k=2)
        if not args.quiet:
            helpers.print_success("Found best schedule!", args.colorless)
            print(best_schedule)

        ########################################################
        # 3. Create sdfg
        if not args.quiet:
            helpers.print_info("Phase 3/3: Creating SDFG...", args.colorless)
        
        csdfg = create_sdfg(best_schedule)

    elif args.version == 'cublas':
        dace.libraries.blas.default_implementation = 'cuBLAS'
    else:
        helpers.print_error("Invalid usage of --version parameter!", args.colorless)
        exit(-1)

    for i in range(args.repetitions):
        A = np.random.rand(M, K).astype(np_dtype)
        B = np.random.rand(K, N).astype(np_dtype)
        C = np.zeros((M, N)).astype(np_dtype)
        if args.version == 'cublas':
            C_test = matmul(A=A, B=B, C=C, alpha=alpha, beta=beta)
        else:
            C_test = csdfg(A=A, B=B, C=C, alpha=alpha, beta=beta, M=M, N=N, K=K)

        if args.verification:
            helpers.print_info("Verifying results...", args.colorless)
            C_correct = matmul(A=A, B=B, C=C, alpha=alpha, beta=beta)

            # Can replace this with np.allclose(A, B)
            def areSame(A,B):
                for i in range(M):
                    for j in range(N):
                        diff = math.fabs(A[i][j] - B[i][j])
                        # helpers.print_info("(" + str(i) + ", " + str(j) + ")", args.colorless)
                        # helpers.print_info("Comparing " + str(B[i][j]) + " to " + str(A[i][j]))
                        # helpers.print_info("Difference = " + str(diff))
                        if (diff > 0.000001):
                            helpers.print_error("Error at position (" + str(i) + ", " + str(j) + "): matrices are not equal! Difference is: " + str(diff), args.colorless)
                            helpers.print_error(str(B[i][j]) + " should be " + str(A[i][j]), args.colorless)
                            print()
                            return False
                return True
            
            helpers.print_info("SDFG result: ", args.colorless)
            print()
            for i in range(16):
                for j in range(16):
                    print("%.2f" % C_test[i][j], end=" ")
                print()

            print()
            print()
            helpers.print_info("Correct result: ", args.colorless)
            for i in range(16):
                for j in range(16):
                    print("%.2f" % C_correct[i][j], end=" ")
                print()

            if areSame(C_correct, C_test):
                helpers.print_success("The SDFG is correct!", args.colorless)
            else:
                helpers.print_error("The SDFG is incorrect!", args.colorless)
