import dace
import sys
import subprocess
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
alpha = dace.symbol('alpha')
beta = dace.symbol('beta')

@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N], alpha: dace.float64, beta: dace.float64):
    return alpha * (A @ B) + beta * C



# Finds and returns the best schedule
def find_best_schedule(load_k_possible, threadtiles_possible, registers_per_warp, registers_per_thread_block, warps_per_SM, SMs):
    best_schedule = Schedule()

    for load_k in load_k_possible:
        for thread_tile_m in threadtiles_possible:
            for thread_tile_n in threadtiles_possible:
                for warp_tile_m in range(thread_tile_m, registers_per_warp, thread_tile_m):
                    for warp_tile_n in range(thread_tile_n, registers_per_warp, thread_tile_n):
                        for thread_block_m in range(warp_tile_m, registers_per_thread_block, warp_tile_m):
                            for thread_block_n in range(warp_tile_n, registers_per_thread_block, warp_tile_n):
                                for split_k in range(1, SMs * warps_per_SM * 2):
                                    schedule = Schedule(load_k, thread_tile_m, thread_tile_n, warp_tile_m, warp_tile_n, thread_block_m, thread_block_n, split_k)
                                    if not fulfills_constraints(schedule):
                                        continue

                                    if schedule > best_schedule:
                                        best_schedule = schedule
    return best_schedule


class Schedule:
    def __init__(self, load_k = 0, thread_tile_m = 0, thread_tile_n = 0, warp_tile_m = 0, warp_tile_n = 0, thread_block_m = 0, thread_block_n = 0, thread_block_k = 0, split_k = 0, double_buffering = True, swizzle = 1):
        self.load_k = load_k
        self.double_buffering = True
        self.swizzle = 1
        self.thread_tile_m = thread_tile_m
        self.thread_tile_n = thread_tile_n
        self.warp_tile_m = warp_tile_m
        self.warp_tile_n = warp_tile_n
        self.thread_block_m = thread_block_m
        self.thread_block_n = thread_block_n
        self.thread_block_k = thread_block_k
        self.split_k = split_k

    def __gt__(self, schedule2):
        # 1. Compare number of CUDA cores used (larger is better)
        # Calculate number of threads used
        # 2. Compare communication volume (smaller is better)
        if self.global_communication_volume() < schedule2.global_communication_volume():
            return True
        elif self.global_communication_volume() > schedule2.global_communication_volume():
            return False
        if self.shared_communication_volume() < schedule2.shared_communication_volume():
            return True
        elif self.shared_communication_volume() > schedule2.shared_communication_volume():
            return False
        # 3. Compare split_k (smaller is better)
        if(self.split_k < schedule2.split_k):
            return True
        elif(self.split_k > schedule2.split_k):
            return False
        # 4. Compare thread_tile_n (larger is better)
        if self.thread_tile_n > schedule2.thread_tile_n:
            return True
        elif self.thread_tile_n < schedule2.thread_tile_n:
            return False

    def __str__(self):
        return """Scheduler with the following parameters:
        load_k: %d
        thread_tile_m: %d
        thread_tile_n: %d
        warp_tile_m: %d
        warp_tile_n: %d
        thread_block_m: %d
        thread_block_n: %d
        thread_block_k: %d
        split_k: %d
        """ % (self.load_k, self.thread_tile_m, self.thread_tile_n, self.warp_tile_m, self.warp_tile_n, self.thread_block_m, self.thread_block_n, self.thread_block_k, self.split_k)

    def global_communication_volume(self):
        volume_A_global = self.thread_block_m * self.thread_block_k
        volume_B_global = self.thread_block_n * self.thread_block_k
        volume_C_global = self.thread_block_m * self.thread_block_n
        if beta != 0:
            volume_C_global *= 2
        total_num_thread_blocks = (M * N * K) / (self.thread_block_m * self.thread_block_n * self.thread_block_k)
        return (volume_A_global + volume_B_global + volume_C_global) * total_num_thread_blocks

    def shared_communication_volume(self):
        volume_A_shared = self.warp_tile_m * self.thread_block_k
        volume_B_shared = self.warp_tile_n * self.thread_block_k
        return (volume_A_shared + volume_B_shared) * warps_per_SM * SMs

# TODO: check constraints
def fulfills_constraints(schedule):
    return True


def create_sdfg(schedule):
    sdfg = matmul.to_sdfg()
    # sdfg.save('matmul_cosma.sdfg')
    sdfg.expand_library_nodes()
    sdfg.apply_transformations(GPUTransformSDFG)
    gemm, state = find_map_by_name(sdfg, "gemm_map")
    xfutil.tile(state.parent, gemm, True, True, __i0 = schedule.thread_block_m, __i1 = schedule.thread_block_n, __i2 = schedule.thread_block_k)
    # Swizzle
    # Split K
    # Double Buffering
    # Warp Tile
    # Vectorization
    # xfutil.tile(state.parent, gemm, True, True, __i0 = schedule.thread_block_m, __i1 = schedule.thread_block_n, __i2 = schedule.thread_block_k)
    sdfg.apply_transformations(MapCollapse)
    sdfg.save('matmul_cosma.sdfg')
    sdfg.compile()


#####################################################################
# Main function

if __name__ == "__main__":
    # Define set of possible values for schedule generator
    load_k_possible = [8, 4, 2, 1]
    threadtiles_possible = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # Hardcoded for V100
    registers_per_thread_block = 65536 / (sys.getsizeof(dace.float64()) / 4)
    registers_per_warp = registers_per_thread_block # Why??
    warps_per_SM = 2
    threads_per_warp = 32
    SMs = 80

    output = subprocess.run(["./getDeviceInfo"])
    print(output)

    ### Find best schedule
    # schedule = find_best_schedule(load_k_possible, threadtiles_possible, registers_per_warp, registers_per_thread_block, warps_per_SM, SMs)

    ### Create sdfg
    # create_sdfg(schedule)