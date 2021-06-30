import subprocess
import helpers

helpers.print_info("Running performance tests...", False)

# (1024 x 1024) x (1024 x 1024)
helpers.print_info("=" * 20, False)
helpers.print_info("===== (1024 x 1024) x (1024 x 1024)", False)
helpers.print_info("=" * 20, False)
subprocess.run(["nvprof", "--csv", "--log-file", "1024_1024_unoptimized.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=1024", "--quiet", "--version=unoptimized", "--skip_verification", "--gpu_type=default", "--repetitions=5"])
subprocess.run(["nvprof", "--csv", "--log-file", "1024_1024_optimized.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=1024", "--quiet", "--version=optimize_gpu", "--skip_verification", "--gpu_type=default", "--repetitions=5"])
subprocess.run(["nvprof", "--csv", "--log-file", "1024_1024_cublas.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=1024", "--quiet", "--version=cublas", "--skip_verification", "--gpu_type=default", "--repetitions=5"])

# (4096 x 4096) x (4096 x 4096)
helpers.print_info("=" * 20, False)
helpers.print_info("===== (4096 x 4096) x (4096 x 4096)", False)
helpers.print_info("=" * 20, False)
subprocess.run(["nvprof", "--csv", "--log-file", "4096_4096_unoptimized.csv", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--quiet", "--version=unoptimized", "--skip_verification", "--gpu_type=default", "--repetitions=5"])
subprocess.run(["nvprof", "--csv", "--log-file", "4096_4096_optimized.csv", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--quiet", "--version=optimize_gpu", "--skip_verification", "--gpu_type=default", "--repetitions=5"])
subprocess.run(["nvprof", "--csv", "--log-file", "4096_4096_cublas.csv", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--quiet", "--version=cublas", "--skip_verification", "--gpu_type=default", "--repetitions=5"])

# (1024 x 8192) x (8192 x 1024)
helpers.print_info("=" * 20, False)
helpers.print_info("===== (1024 x 8192) x (8192 x 1024)", False)
helpers.print_info("=" * 20, False)
subprocess.run(["nvprof", "--csv", "--log-file", "1024_8192_8192_1024_unoptimized.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=unoptimized", "--skip_verification", "--gpu_type=default", "--repetitions=5"])
subprocess.run(["nvprof", "--csv", "--log-file", "1024_8192_8192_1024_optimized.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=optimize_gpu", "--skip_verification", "--gpu_type=default", "--repetitions=5"])
subprocess.run(["nvprof", "--csv", "--log-file", "1024_8192_8192_1024_cublas.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=cublas", "--skip_verification", "--gpu_type=default", "--repetitions=5"])