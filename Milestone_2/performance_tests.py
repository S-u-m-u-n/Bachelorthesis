import subprocess
import helpers

# 4096 x 4096
# (1024 x 8192) x (8192 x 1024)
# 1024 x 1024

helpers.print_info("Running performance tests...", False)
helpers.print_info("=" * 20, False)
helpers.print_info("===== (1024 x 1024) x (1024 x 1024)", False)
helpers.print_info("=" * 20, False)
subprocess.run(["nvprof", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=1024", "--quiet", "--version=unoptimized"])
subprocess.run(["nvprof", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=1024", "--quiet", "--version=cublas"])
helpers.print_info("=" * 20, False)
helpers.print_info("===== (4096 x 4096) x (4096 x 4096)", False)
helpers.print_info("=" * 20, False)
subprocess.run(["nvprof", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--quiet", "--version=unoptimized"])
subprocess.run(["nvprof", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--quiet", "--version=cublas"])
helpers.print_info("=" * 20, False)
helpers.print_info("===== (1024 x 8192) x (8192 x 1024)", False)
helpers.print_info("=" * 20, False)
subprocess.run(["nvprof", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=unoptimized"])
subprocess.run(["nvprof", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=cublas"])