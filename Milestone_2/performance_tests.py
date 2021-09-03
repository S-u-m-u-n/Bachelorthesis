import subprocess
import helpers
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--test", type=int, dest='test', nargs="?", default=1)
# parser.add_argument("-r", "--repetitions", type=int, dest='repetitions', nargs="?", default=1)
args = parser.parse_args()

helpers.print_info("Running performance tests...", False)

### (1024 x 1024) x (1024 x 1024)
if args.test == 1:
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (1024 x 1024) x (1024 x 1024)", False)
    helpers.print_info("=" * 20, False)
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "1024_1024_unoptimized.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--repetitions=5"])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "1024_1024_swizzle.csv", "python", "./sdfg_api.py", "-M=1024", "-N=1024", "-K=1024", "--swizzle-threads", "--repetitions=5"])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "1024_1024_cublas.csv", "python", "./cublas.py", "-M=1024", "-N=1024", "-K=1024", "--repetitions=5"])

### (4096 x 4096) x (4096 x 4096)
if args.test == 2:
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (4096 x 4096) x (4096 x 4096)", False)
    helpers.print_info("=" * 20, False)
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "4096_4096_unoptimized.csv", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--quiet", "--version=dace", "--skip-verification", "--gpu-type=default", "--repetitions=5"])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "4096_4096_optimized.csv", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--quiet", "--version=dace", "--skip-verification", "--gpu-type=default", "--repetitions=5"])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "4096_4096_cublas.csv", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--repetitions=5"])

### (1024 x 8192) x (8192 x 1024)
if args.test == 3:
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (1024 x 8192) x (8192 x 1024)", False)
    helpers.print_info("=" * 20, False)
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "1024_8192_8192_1024_unoptimized.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=dace", "--skip-verification", "--gpu-type=default", "--repetitions=5"])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "1024_8192_8192_1024_optimized.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=dace", "--skip-verification", "--gpu-type=default", "--repetitions=5"])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "1024_8192_8192_1024_cublas.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--repetitions=5"])