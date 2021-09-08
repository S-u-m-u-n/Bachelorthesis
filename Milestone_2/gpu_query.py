#####################################################################
# Query functions

def queryNVIDIA():
    if not args.quiet:
        helpers.print_info("Querying NVIDIA device info...", args.colorless)
    if args.verbose:
        getDeviceInfo_NVIDIA = subprocess.run(["./getDeviceInfo_NVIDIA"])
    else:
        getDeviceInfo_NVIDIA = subprocess.run(
            ["./getDeviceInfo_NVIDIA"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    if getDeviceInfo_NVIDIA.returncode == 0:
        if not args.quiet:
            helpers.print_success("Successfully read NVIDIA device Info", args.colorless)
        return True
    else:
        if not args.quiet:
            helpers.print_warning("No CUDA Capable GPU found", args.colorless)
        return False

def queryAMD():
    if not args.quiet:
        helpers.print_info("Querying AMD device info...", args.colorless)
    if args.verbose:
        getDeviceInfo_AMD = subprocess.run(["./getDeviceInfo_AMD"])
    else:
        getDeviceInfo_AMD = subprocess.run(
            ["./getDeviceInfo_AMD"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    if getDeviceInfo_AMD.returncode == 0:
        if not args.quiet:
            helpers.print_success("Successfully read AMD device Info", args.colorless)
        return True
    else:
        if not args.quiet:
            helpers.print_warning("No AMD GPU found", args.colorless)
        return False

    ########################################################
    # 1. Get Device Properties or use default (Tesla V100)
    default_device_data = open('device_data.py', 'w')
    default_device_data.write("""Name = "Tesla V100-PCIE-32GB"
SMs = 80
warps_per_SM = 2
threads_per_warp = 32
registers_per_thread_block = 65536
registers_per_warp = 65536
total_compute_cores = 5120
capability_version = 7.0""")
    default_device_data.close()

    if not args.quiet:
        helpers.print_info("Phase 1/3: Querying device info...", args.colorless)

    if args.gpu_type == "NVIDIA":
        queryNVIDIA()
    elif args.gpu_type == "AMD":
        # os.environ['DACE_compiler_cuda_backend'] = 'hip'
        dace.Config.set('DACE_compiler_cuda_backend', value='hip')
        queryAMD()
    elif args.gpu_type != "default":
        helpers.print_error("Invalid usage of -g parameter!")
        exit(-1)

    import device_data as device

    if not args.quiet:
        helpers.print_info(
            "Using the following GPU for the schedule generator: ", args.colorless)
        helpers.print_device_info(device, args.colorless)

    device.registers_per_thread_block = int(device.registers_per_thread_block /
                                            (sys.getsizeof(dace.float64()) / 4))
    device.registers_per_warp = int(device.registers_per_warp /
                                    (sys.getsizeof(dace.float64()) / 4))