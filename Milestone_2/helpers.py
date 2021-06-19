def print_success(text, colorless = False):
    if colorless:
        print("[Sucess] ", text)
    else:
        print("\033[01m\033[32m[Success] \033[0m", text)

def print_warning(text, colorless = False):
    if colorless:
        print("[Warning] ", text)
    else:
        print("\033[01m\033[33m[Warning] \033[0m", text)

def print_error(text, colorless = False):
    if colorless:
        print("[Error] ", text)
    else:
        print("\033[01m\033[31m[Error] \033[0m", text)

def print_info(text, colorless = False):
    if colorless:
        print("[Info] ", text)
    else:
        print("\033[01m\033[36m[Info] \033[0m", text)

def print_device_info(device, colorless = False):
    if colorless:
        print("============== GPU Info ==============")
        print(">> Name: ", device.Name)
        print(">> SMs: ", device.SMs)
        print(">> threads_per_warp: ", device.threads_per_warp)
        print(">> warps_per_SM: ", device.warps_per_SM)
        print(">> registers_per_thread_block: ", device.registers_per_thread_block)
        print(">> registers_per_warp:", device.registers_per_warp)
        print(">> total_cuda_cores:", device.total_compute_cores)
        print(">> cuda_capability_version:", device.capability_version)
        print("======================================")
    else:
        print("\033[01m\033[36m============== GPU Info ==============\033[0m")
        print("\033[01m\033[36m>> Name:\033[0m", device.Name)
        print("\033[01m\033[36m>> SMs:\033[0m", device.SMs)
        print("\033[01m\033[36m>> threads_per_warp:\033[0m", device.threads_per_warp)
        print("\033[01m\033[36m>> warps_per_SM:\033[0m", device.warps_per_SM)
        print("\033[01m\033[36m>> registers_per_thread_block:\033[0m", device.registers_per_thread_block)
        print("\033[01m\033[36m>> registers_per_warp:\033[0m", device.registers_per_warp)
        print("\033[01m\033[36m>> total_cuda_cores:\033[0m", device.total_compute_cores)
        print("\033[01m\033[36m>> cuda_capability_version:\033[0m", device.capability_version)
        print("\033[01m\033[36m======================================\033[0m")