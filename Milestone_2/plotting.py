import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import helpers

# ToDo: Use seaborn to plot & calculate CI
sns.set_theme(style="whitegrid")

parser = ArgumentParser()
parser.add_argument("--test", type=int, dest='test', nargs="?", default=1)
args = parser.parse_args()

helpers.print_info("Creating performance plots...", False)

def read_nvprof_data(path_to_csv):
    df = pd.read_csv(path_to_csv, skiprows=3).filter(['Duration', 'Name']).iloc[1:]
    return df[df['Name'].str.contains("Thread_block_grid|dgemm")].filter(['Duration']).reset_index(drop=True).apply(pd.to_numeric, errors='coerce')

### (1024 x 1024) x (1024 x 1024)
if args.test == 1:
    unoptimized_df = read_nvprof_data("./1024_1024_unoptimized.csv")
    swizzled_threads_df = read_nvprof_data("./1024_1024_swizzle.csv")
    # swizzled_threads = read_nvprof_data("./1024_1024_swizzled_threads.csv")
    cublas_df = read_nvprof_data("./1024_1024_cublas.csv")
    # print(unoptimized.head(10))
    # print(swizzled_threads.head(10))
    # print(cublas.head(10))
    combined_df = pd.concat([unoptimized_df, swizzled_threads_df, cublas_df], axis=1)
    combined_df.columns = ["unoptimized", "swizzled_threads", "cublas"]
    # print(combined_df.head(10))

    ax = sns.violinplot(data=combined_df)
    ax.set(xticklabels=["Unoptimized", "Swizzled Threads", "Cublas"], ylabel="Runtime [ms]", title="M = 1024, N = 1024, K = 1024") # , xlabel=""
    plt.show()

### (4096 x 4096) x (4096 x 4096)
if args.test == 2:
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "4096_4096_unoptimized.csv", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--quiet", "--version=dace", "--skip-verification", "--gpu-type=default", "--repetitions=5"])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "4096_4096_optimized.csv", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--quiet", "--version=dace", "--skip-verification", "--gpu-type=default", "--repetitions=5"])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "4096_4096_cublas.csv", "python", "./cosma.py", "-M=4096", "-N=4096", "-K=4096", "--repetitions=5"])

### (1024 x 8192) x (8192 x 1024)
if args.test == 3:
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "1024_8192_8192_1024_unoptimized.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=dace", "--skip-verification", "--gpu-type=default", "--repetitions=5"])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "1024_8192_8192_1024_optimized.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--quiet", "--version=dace", "--skip-verification", "--gpu-type=default", "--repetitions=5"])
    subprocess.run(["nvprof", "--print-gpu-trace", "--csv", "--log-file", "1024_8192_8192_1024_cublas.csv", "python", "./cosma.py", "-M=1024", "-N=1024", "-K=8192", "--repetitions=5"])

helpers.print_success("Performance plots created.", False)