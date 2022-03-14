class Schedule:
    def __init__(self, load_k=1, thread_tile_m=1, thread_tile_n=1, thread_tile_k=1, warp_tile_m=1, warp_tile_n=1, thread_block_tile_m=1, thread_block_tile_n=1, thread_block_tile_k=1, split_k=1, SWIZZLE_thread_block=1, SWIZZLE_thread_tile=False):
        self.load_k = load_k
        self.thread_tile_m = thread_tile_m
        self.thread_tile_n = thread_tile_n
        self.thread_tile_k = thread_tile_k
        self.warp_tile_m = warp_tile_m
        self.warp_tile_n = warp_tile_n
        self.thread_block_tile_m = thread_block_tile_m
        self.thread_block_tile_n = thread_block_tile_n
        self.thread_block_tile_k = thread_block_tile_k
        self.split_k = split_k
        self.SWIZZLE_thread_block = SWIZZLE_thread_block
        self.SWIZZLE_thread_tile = SWIZZLE_thread_tile

    def __gt__(self, schedule2):
        # 1. Compare number of compute (CUDA) cores used (larger is better)
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
        thread_block_tile_m: %d
        thread_block_tile_n: %d
        thread_block_tile_k: %d
        split_k: %d
        SWIZZLE_thread_block: %d
        """ % (self.load_k, self.thread_tile_m, self.thread_tile_n, self.warp_tile_m, self.warp_tile_n, self.thread_block_tile_m, self.thread_block_tile_n, self.thread_block_tile_k, self.split_k, self.SWIZZLE_thread_block)

    def num_threads_used(self):
        numTilesM = math.ceil(M / dace.float64(self.thread_tile_m))
        numTilesN = math.ceil(N / dace.float64(self.thread_tile_n))
        numTilesK = math.ceil(K / dace.float64(self.thread_tile_k))
        threads_used_full = (numTilesM - 1) * (numTilesN - 1) * (numTilesK - 1) * min(
            device.warps_per_SM, numTilesM * numTilesN * numTilesK) * device.threads_per_warp  # What is total_P??

        M_Overflow = self.thread_block_tile_m * numTilesM - M
        N_Overflow = self.thread_block_tile_n * numTilesN - N

        M_Threads = math.ceil(
            (self.thread_block_tile_m - M_Overflow) / dace.float64(self.thread_tile_m))
        N_Threads = math.ceil(
            (self.thread_block_tile_n - N_Overflow) / dace.float64(self.thread_tile_n))

        M_Leftover = self.thread_block_tile_m / self.thread_tile_m - M_Threads
        N_Leftover = self.thread_block_tile_n / self.thread_tile_n - N_Threads

        threads_used_top = 1 * (numTilesN - 1) * numTilesK * min(device.warps_per_SM * device.threads_per_warp,
                                                                 numTilesM * numTilesN * numTilesK * device.threads_per_warp - M_Leftover * (self.thread_block_tile_n / self.thread_tile_n))  # What is total_P??
        threads_used_bottom = (numTilesM - 1) * 1 * numTilesK * min(device.warps_per_SM * device.threads_per_warp,
                                                                    numTilesM * numTilesN * numTilesK * device.threads_per_warp - N_Leftover * (self.thread_block_tile_m / self.thread_tile_m))  # What is total_P??
        threads_used_top_right = 1 * 1 * numTilesK * min(device.warps_per_SM * device.threads_per_warp,
                                                         numTilesM * numTilesN * numTilesK * device.threads_per_warp - N_Leftover * (self.thread_block_tile_m / self.thread_tile_m) - M_Leftover * (self.thread_block_tile_n / self.thread_tile_n) + N_Leftover * M_Leftover)  # What is total_P??

        total_threads_used = threads_used_full + threads_used_top + \
            threads_used_bottom + threads_used_top_right

        return min(total_threads_used, device.total_compute_cores)

    def global_communication_volume(self):
        volume_A_global = self.thread_block_tile_m * self.thread_block_tile_k
        volume_B_global = self.thread_block_tile_n * self.thread_block_tile_k
        volume_C_global = self.thread_block_tile_m * self.thread_block_tile_n
        if beta != 0:
            volume_C_global *= 2
        total_num_thread_blocks = (
            M * N * K) / (self.thread_block_tile_m * self.thread_block_tile_n * self.thread_block_tile_k)
        return (volume_A_global + volume_B_global + volume_C_global) * total_num_thread_blocks

    def shared_communication_volume(self):
        volume_A_shared = self.warp_tile_m * self.thread_block_tile_k
        volume_B_shared = self.warp_tile_n * self.thread_block_tile_k
        return (volume_A_shared + volume_B_shared) * device.warps_per_SM * device.SMs