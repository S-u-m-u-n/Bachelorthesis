import subprocess
import helpers

# 4096 x 4096
# (1024 x 8192) x (8192 x 1024)
# 1024 x 1024

helpers.print_info("Running performance tests...", False)
subprocess.run(["nvprof python ./cosma.py -M=1024 -N=1024 -K=1024 --version=cublas"])
subprocess.run(["nvprof python ./cosma.py -M=4096 -N=4096 -K=4096 --version=cublas"])
subprocess.run(["nvprof python ./cosma.py -M=1024 -N=1024 -K=8192 --version=cublas"])