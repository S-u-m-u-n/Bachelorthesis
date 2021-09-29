import subprocess
from argparse import ArgumentParser
import helpers

parser = ArgumentParser()
parser.add_argument("-t", "--test", type=int, dest='test', nargs="?", required=True)
parser.add_argument("-r", "--repetitions", type=int, dest='repetitions', nargs="?", default=10)
parser.add_argument("-p", "--path", type=str, dest='path', nargs="?", required=True)
args = parser.parse_args()

helpers.print_info("Running performance tests...", False)

### (1024 x 1024) x (1024 x 1024)
if args.test == 1:
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (1024 x 1024) x (1024 x 1024)", False)
    helpers.print_info("=" * 20, False)
    subprocess.run(["mkdir", "-p", "./performance_test_results/1024_1024/" + str(args.path)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/unoptimized.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/vectorization.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--vectorization", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/swizzled_threads.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/swizzled_threads_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--swizzle-threads", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/vectorization_swizzled_threads.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--vectorization", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/vectorization_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--vectorization", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/vectorization_swizzled_threads_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--vectorization", "--swizzle-threads", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--double-buffering", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_vectorization.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--double-buffering", "--vectorization", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_swizzled_threads.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--double-buffering", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_swizzled_threads_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--swizzle-threads", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_vectorization_swizzled_threads.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--double-buffering", "--vectorization", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_vectorization_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--double-buffering", "--vectorization", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_vectorization_swizzled_threads_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--double-buffering", "--vectorization", "--swizzle-threads", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/cublas.csv", "python", "./cublas.py", "-M=1024", "-N=1024", "-K=1024", "--repetitions=" + str(args.repetitions)])
    # subprocess.run(["sed", "-i", "'1d;2d;3d'", "./performance_test_results/1024_1024/cublas.csv"]) # Delete first three lines, which are filled with nvprof information
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/1024_1024/cutlass.csv", "/home/jacobsi/cutlass/build/tools/profiler/cutlass_profiler", "--verification-enabled=false", "--warmup-iterations=0" "--operation=Gemm", "--m=1024", "--n=1024", "--k=1024", "--cta_m=128", "--cta_n=64", "--A=f64:column", "--B=f64:column", "--profiling-iterations=" + str(args.repetitions - 1)])
    subprocess.run(["sed", "-i", "'1d;2d;3d'", "./performance_test_results/1024_1024/cutlass.csv"]) # Delete first three lines, which are filled with nvprof information
    subprocess.run(["sed", "-i", "'1d;2d;3d'", "./performance_test_results/1024_1024/" + str(args.path) + "/*.csv"]) # Delete first three lines, which are filled with nvprof information

### (4096 x 4096) x (4096 x 4096)
if args.test == 2:
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (4096 x 4096) x (4096 x 4096)", False)
    helpers.print_info("=" * 20, False)
    subprocess.run(["mkdir", "-p", "./performance_test_results/4096_4096/" + str(args.path)])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/unoptimized.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--repetitions=" + str(args.repetitions)])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/vectorization.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--vectorization", "--repetitions=" + str(args.repetitions)])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/swizzled_threads.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/vectorization_swizzled_threads.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--vectorization", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    # # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--double-buffering", "--repetitions=" + str(args.repetitions)])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_vectorization.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--double-buffering", "--vectorization", "--repetitions=" + str(args.repetitions)])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_swizzled_threads.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--double-buffering", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_vectorization_swizzled_threads.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--double-buffering", "--vectorization", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])

    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/unoptimized.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/vectorization.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--vectorization", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/swizzled_threads.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/swizzled_threads_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--swizzle-threads", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/vectorization_swizzled_threads.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--vectorization", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/vectorization_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--vectorization", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/vectorization_swizzled_threads_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--vectorization", "--swizzle-threads", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--double-buffering", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_vectorization.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--double-buffering", "--vectorization", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_swizzled_threads.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--double-buffering", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_swizzled_threads_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--swizzle-threads", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_vectorization_swizzled_threads.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--double-buffering", "--vectorization", "--swizzle-threads", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_vectorization_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--double-buffering", "--vectorization", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_vectorization_swizzled_threads_swizzled_thread_blocks.csv", "python", "./sdfg_api.py", "-M=4096", "-N=4096", "-K=4096", "--double-buffering", "--vectorization", "--swizzle-threads", "--swizzle-thread-blocks", "--repetitions=" + str(args.repetitions)])


    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/cublas.csv", "python", "./cublas.py", "-M=4096", "-N=4096", "-K=4096", "--repetitions=" + str(args.repetitions)])
    # subprocess.run(["sed", "-i", "'1d;2d;3d'", "./performance_test_results/4096_4096/cublas.csv"]) # Delete first three lines, which are filled with nvprof information
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "./performance_test_results/4096_4096/cutlass.csv", "/home/jacobsi/cutlass/build/tools/profiler/cutlass_profiler", "--verification-enabled=false", "--warmup-iterations=0" "--operation=Gemm", "--m=4096", "--n=4096", "--k=4096", "--cta_m=128", "--cta_n=64", "--A=f64:column", "--B=f64:column", "--profiling-iterations=" + str(args.repetitions - 1)])
    subprocess.run(["sed", "-i", "'1d;2d;3d'", "./performance_test_results/4096_4096/cutlass.csv"]) # Delete first three lines, which are filled with nvprof information

    subprocess.run(["sed", "-i", "'1d;2d;3d'", "./performance_test_results/4096_4096/" + str(args.path) + "/*.csv"]) # Delete first three lines, which are filled with nvprof information

### (1024 x 8192) x (8192 x 1024)
# if args.test == 3:
    # helpers.print_info("=" * 20, False)
    # helpers.print_info("===== (1024 x 8192) x (8192 x 1024)", False)
    # helpers.print_info("=" * 20, False)
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "performance_test_results/1024_8192_8192_1024/" + str(args.path) + "/1024_8192_8192_1024_unoptimized.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=dace", "--skip-verification", "--gpu-type=default", "--repetitions=5"])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "performance_test_results/1024_8192_8192_1024/" + str(args.path) + "/1024_8192_8192_1024_optimized.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=dace", "--skip-verification", "--gpu-type=default", "--repetitions=5"])
    # subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "performance_test_results/1024_8192_8192_1024/" + str(args.path) + "/1024_8192_8192_1024_cublas.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--repetitions=5"])

helpers.print_success("Performance tests finished.", False)
