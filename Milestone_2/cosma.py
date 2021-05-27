import dace
import sys
import subprocess
import math
from argparse import ArgumentParser
from csv import DictReader
import numpy as np
from tqdm import tqdm
from dace.transformation.interstate import GPUTransformSDFG, StateFusion
from dace.transformation.dataflow import MapTiling, InLocalStorage, MapExpansion, MapCollapse
from dace.transformation.optimizer import Optimizer
from dace.transformation import helpers as xfutil
import helpers


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

parser = ArgumentParser()
parser.add_argument("-v", "--verbose", dest='verbose', help="explain what is being done",
                    action="store_true", default=False)
parser.add_argument("-c", "--colorless", dest='colorless',
                    help="does not print colors, useful when writing to a file", action="store_true", default=False)
parser.add_argument("-g", "--gpu_type", dest='gpu_type',
                    help="use this if you want to specify the gpu type (\"NVIDIA\" or \"AMD\") instead of determining it during runtime", action="store", default="AUTO")
args = parser.parse_args()
if args.verbose:
    print(args)


@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N], alpha: dace.float64, beta: dace.float64):
    return alpha * (A @ B) + beta * C


M = 128
N = 128
K = 128

# Finds and returns the best schedule
def find_best_schedule(load_k_possible, threadtiles_possible):
    best_schedule = Schedule()

    for load_k in tqdm(load_k_possible, desc="load_k", position=0, leave=False, ncols=80):
        for thread_tile_m in tqdm(threadtiles_possible, desc="thread_tile_m", position=1, leave=False, ncols=80):
            for thread_tile_n in tqdm(threadtiles_possible, desc="thread_tile_n", position=2, leave=False, ncols=80):
                for thread_tile_k in tqdm(threadtiles_possible, desc="thread_tile_k", position=3, leave=False, ncols=80):
                    for warp_tile_m in tqdm(range(thread_tile_m, device.registers_per_warp, thread_tile_m), desc="warp_tile_m", position=4, leave=False, ncols=80):
                        for warp_tile_n in tqdm(range(thread_tile_n, device.registers_per_warp, thread_tile_n), desc="warp_tile_n", position=5, leave=False, ncols=80):
                            for thread_block_m in tqdm(range(warp_tile_m, device.registers_per_thread_block, warp_tile_m), desc="thread_block_m", position=6, leave=False, ncols=80):
                                for thread_block_n in tqdm(range(warp_tile_n, device.registers_per_thread_block, warp_tile_n), desc="thread_block_n", position=7, leave=False, ncols=80):
                                    for split_k in tqdm(range(1, device.SMs * device.warps_per_SM * 2), desc="split_k", position=8, leave=False, ncols=80):
                                        schedule = Schedule(load_k, thread_tile_m, thread_tile_n, warp_tile_m, warp_tile_n,
                                                            thread_block_m, thread_block_n, split_k)
                                        # print(schedule)
                                        if not fulfills_constraints(schedule):
                                            continue

                                        if schedule > best_schedule:
                                            best_schedule = schedule
    return best_schedule


class Schedule:
    def __init__(self, load_k=1, thread_tile_m=1, thread_tile_n=1, thread_tile_k=1, warp_tile_m=1, warp_tile_n=1, thread_block_m=1, thread_block_n=1, thread_block_k=1, split_k=1, double_buffering=True, swizzle=1):
        self.load_k = load_k
        self.thread_tile_m = thread_tile_m
        self.thread_tile_n = thread_tile_n
        self.thread_tile_k = thread_tile_k
        self.warp_tile_m = warp_tile_m
        self.warp_tile_n = warp_tile_n
        self.thread_block_m = thread_block_m
        self.thread_block_n = thread_block_n
        self.thread_block_k = thread_block_k
        self.split_k = split_k
        self.double_buffering = double_buffering
        self.swizzle = swizzle

    def __gt__(self, schedule2):
        # 1. Compare number of CUDA cores used (larger is better)
        # For now, we calculate the number of threads used instead
        if self.num_threads_used() > schedule2.num_threads_used():
            return True
        elif self.num_threads_used() < schedule2.num_threads_used():
            return False
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
        double_buffering: %d
        swizzle: %d
        """ % (self.load_k, self.thread_tile_m, self.thread_tile_n, self.warp_tile_m, self.warp_tile_n, self.thread_block_m, self.thread_block_n, self.thread_block_k, self.split_k, self.double_buffering, self.swizzle)

    def num_threads_used(self):
        numTilesM = math.ceil(M / dace.float64(self.thread_tile_m))
        numTilesN = math.ceil(N / dace.float64(self.thread_tile_n))
        numTilesK = math.ceil(K / dace.float64(self.thread_tile_k))
        threads_used_full = (numTilesM - 1) * (numTilesN - 1) * (numTilesK - 1) * min(
            device.warps_per_SM, numTilesM * numTilesN * numTilesK) * device.threads_per_warp  # What is total_P??

        M_Overflow = self.thread_block_m * numTilesM - M
        N_Overflow = self.thread_block_n * numTilesN - N

        M_Threads = math.ceil(
            (self.thread_block_m - M_Overflow) / dace.float64(self.thread_tile_m))
        N_Threads = math.ceil(
            (self.thread_block_n - N_Overflow) / dace.float64(self.thread_tile_n))

        M_Leftover = self.thread_block_m / self.thread_tile_m - M_Threads
        N_Leftover = self.thread_block_n / self.thread_tile_n - N_Threads

        threads_used_top = 1 * (numTilesN - 1) * numTilesK * min(device.warps_per_SM * device.threads_per_warp,
                                                                 numTilesM * numTilesN * numTilesK * device.threads_per_warp - M_Leftover * (self.thread_block_n / self.thread_tile_n))  # What is total_P??
        threads_used_bottom = (numTilesM - 1) * 1 * numTilesK * min(device.warps_per_SM * device.threads_per_warp,
                                                                    numTilesM * numTilesN * numTilesK * device.threads_per_warp - N_Leftover * (self.thread_block_m / self.thread_tile_m))  # What is total_P??
        threads_used_top_right = 1 * 1 * numTilesK * min(device.warps_per_SM * device.threads_per_warp,
                                                         numTilesM * numTilesN * numTilesK * device.threads_per_warp - N_Leftover * (self.thread_block_m / self.thread_tile_m) - M_Leftover * (self.thread_block_n / self.thread_tile_n) + N_Leftover * M_Leftover)  # What is total_P??

        total_threads_used = threads_used_full + threads_used_top + \
            threads_used_bottom + threads_used_top_right

        return min(total_threads_used, device.total_cuda_cores)

    def global_communication_volume(self):
        volume_A_global = self.thread_block_m * self.thread_block_k
        volume_B_global = self.thread_block_n * self.thread_block_k
        volume_C_global = self.thread_block_m * self.thread_block_n
        if beta != 0:
            volume_C_global *= 2
        total_num_thread_blocks = (
            M * N * K) / (self.thread_block_m * self.thread_block_n * self.thread_block_k)
        return (volume_A_global + volume_B_global + volume_C_global) * total_num_thread_blocks

    def shared_communication_volume(self):
        volume_A_shared = self.warp_tile_m * self.thread_block_k
        volume_B_shared = self.warp_tile_n * self.thread_block_k
        return (volume_A_shared + volume_B_shared) * device.warps_per_SM * device.SMs


def fulfills_constraints(schedule):
    # Todo: check constraints
    return True


def create_sdfg(schedule):
    sdfg = matmul.to_sdfg()
    # sdfg.save('matmul_cosma.sdfg')
    sdfg.expand_library_nodes()
    sdfg.apply_transformations(GPUTransformSDFG)
    gemm, state = find_map_by_name(sdfg, "gemm_map")
    # Threadblock Tile
    xfutil.tile(state.parent, gemm, True, True, __i0=schedule.thread_block_m,
                __i1=schedule.thread_block_n, __i2=schedule.thread_block_k)
    # Warp Tile
    xfutil.tile(state.parent, gemm, True, True,
                __i0=schedule.warp_tile_m, __i1=schedule.warp_tile_n)
    # Swizzle
    # Split K
    # Double Buffering
    # Vectorization
    sdfg.apply_transformations(MapCollapse)
    sdfg.save('matmul_cosma.sdfg')
    sdfg.compile()

#####################################################################
# Query functions

def queryNVIDIA():
    helpers.print_info("Querying NVIDIA device info...", args.colorless)
    if args.verbose:
        getDeviceInfo_NVIDIA = subprocess.run(["./getDeviceInfo_NVIDIA"])
    else:
        getDeviceInfo_NVIDIA = subprocess.run(
            ["./getDeviceInfo_NVIDIA"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    if getDeviceInfo_NVIDIA.returncode == 0:
        helpers.print_success("Successfully read NVIDIA device Info", args.colorless)
        # import device_data as device
        return True
    else:
        helpers.print_warning(
            "No CUDA Capable GPU found", args.colorless)
        # import TeslaV100_data as device
        return False

def queryAMD():
    helpers.print_info("Querying AMD device info...", args.colorless)
    if args.verbose:
        getDeviceInfo_AMD = subprocess.run(["./getDeviceInfo_AMD"])
    else:
        getDeviceInfo_AMD = subprocess.run(
            ["./getDeviceInfo_AMD"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    if getDeviceInfo_AMD.returncode == 0:
        helpers.print_success("Successfully read AMD device Info", args.colorless)
        # import device_data as device
        return True
    else:
        helpers.print_warning(
            "No AMD GPU found", args.colorless)
        # import TeslaV100_data as device
        # helpers.print_info("Using default GPU: Tesla V100")
        return False

#####################################################################
# Main function

if __name__ == "__main__":
    # Define set of possible values for schedule generator
    load_k_possible = [1, 2, 4, 8]
    threadtiles_possible = [1, 2, 4, 8, 16]
    # threadtiles_possible = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    ########################################################
    # 1. Get Device Properties or use default (Tesla V100)
    default_device_data = open('device_data.py', 'w')
    default_device_data.write("""Name = "Tesla V100-PCIE-32GB"
SMs = 80
warps_per_SM = 2
threads_per_warp = 32
registers_per_thread_block = 65536
registers_per_warp = 65536
total_cuda_cores = 5120
cuda_capability_version = 7.0""")
    default_device_data.close()
    
    helpers.print_info("Phase 1/3: Querying device info...", args.colorless)
    if args.gpu_type == "AUTO":
        if queryNVIDIA() == False:
            helpers.print_info("No CUDA-capable GPU found, looking for an AMD GPU...", args.colorless)
            queryAMD()
    elif args.gpu_type == "NVIDIA":
        queryNVIDIA()
    elif args.gpu_type == "AMD":
        queryAMD()

    import device_data as device

    helpers.print_info(
        "Using the following GPU for the schedule generator: ", args.colorless)
    helpers.print_device_info(device, args.colorless)

    device.registers_per_thread_block = int(device.registers_per_thread_block /
                                            (sys.getsizeof(dace.float64()) / 4))
    device.registers_per_warp = int(device.registers_per_warp /
                                    (sys.getsizeof(dace.float64()) / 4))

    ########################################################
    # 2. Find best schedule
    helpers.print_info("Phase 2/3: Finding best schedule...", args.colorless)
    # schedule = find_best_schedule(load_k_possible, threadtiles_possible, device.registers_per_warp, device.registers_per_thread_block, device.threads_per_warp, device.warps_per_SM, device.SMs, device.total_cuda_cores)
    schedule = find_best_schedule(load_k_possible, threadtiles_possible)
    print(schedule)

    ########################################################
    # 3. Create sdfg
    helpers.print_info("Phase 3/3: Creating SDFG...", args.colorless)
    # create_sdfg(schedule)

    # 4. Convert to HiP code (if needed)

