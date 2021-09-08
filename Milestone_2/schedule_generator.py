# Finds and returns the best schedule
def find_best_schedule(load_k_possible, threadtiles_possible):
    best_schedule = Schedule()

    for load_k in tqdm(load_k_possible, desc="load_k", position=0, leave=False, ncols=80):
        for thread_tile_m in tqdm(threadtiles_possible, desc="thread_tile_m", position=1, leave=False, ncols=80):
            for thread_tile_n in tqdm(threadtiles_possible, desc="thread_tile_n", position=2, leave=False, ncols=80):
                for thread_tile_k in tqdm(threadtiles_possible, desc="thread_tile_k", position=3, leave=False, ncols=80):
                    for warp_tile_m in tqdm(range(thread_tile_m, device.registers_per_warp, thread_tile_m), desc="warp_tile_m", position=4, leave=False, ncols=80):
                        for warp_tile_n in tqdm(range(thread_tile_n, device.registers_per_warp, thread_tile_n), desc="warp_tile_n", position=5, leave=False, ncols=80):
                            for thread_block_tile_m in tqdm(range(warp_tile_m, device.registers_per_thread_block, warp_tile_m), desc="thread_block_tile_m", position=6, leave=False, ncols=80):
                                for thread_block_tile_n in tqdm(range(warp_tile_n, device.registers_per_thread_block, warp_tile_n), desc="thread_block_tile_n", position=7, leave=False, ncols=80):
                                    for split_k in tqdm(range(1, device.SMs * device.warps_per_SM * 2), desc="split_k", position=8, leave=False, ncols=80):
                                        schedule = Schedule(load_k, thread_tile_m, thread_tile_n, warp_tile_m, warp_tile_n,
                                                            thread_block_tile_m, thread_block_tile_n, split_k)
                                        # print(schedule)
                                        if not fulfills_constraints(schedule):
                                            continue

                                        if schedule > best_schedule:
                                            best_schedule = schedule
    return best_schedule

def fulfills_constraints(schedule):
    # check constraints
    return True