import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import helpers

# ToDo: Use seaborn to plot & calculate CI
sns.set_theme(style="whitegrid")

parser = ArgumentParser()
parser.add_argument("-t", "--test", type=int, dest='test', nargs="?", required=True)
parser.add_argument("-p", "--path", type=str, dest='path', nargs="?", required=True)
parser.add_argument('--precision', type=int, dest='precision', choices=[32, 64], default=64, help="Specify floating precision (32 or 64)")
args = parser.parse_args()

helpers.print_info("Creating performance plots...", False)

def read_nvprof_data(path_to_csv):
    df = pd.read_csv(path_to_csv, skiprows=3)
    flag = False
    if (df['Duration'][0] == 'us'):
        flag = True
    # print(df.iloc[1, 1])
    # if (df[1])
    df = df.filter(['Duration', 'Name']).iloc[1:]

    df = df[df['Name'].str.contains("Thread_block_grid|dgemm|sgemm|cosmaSgemm|cutlass")].filter(['Duration']).reset_index(drop=True).apply(pd.to_numeric, errors='coerce').iloc[1:, :]
    if flag:
        df /= 1000
    return df

if args.precision == 32:
    peak_performance = 14 * 1000 * 1000 * 1000 # TFLOPS/ms
    precision_str = "Single precision: "
else:
    peak_performance = 7 * 1000 * 1000 * 1000 # TFLOPS/ms
    precision_str = "Double precision: "

### (1024 x 1024) x (1024 x 1024)
if args.test == 1:
    base_path = "./performance_test_results/1024_1024_1024_" + str(args.precision) + "bit/"
    path = base_path + str(args.path) +'/'
    ### Without Double Buffering
    unoptimized_df = read_nvprof_data(path + "unoptimized.csv")
    # v_df = read_nvprof_data(path + "vectorization.csv")
    st_df = read_nvprof_data(path + "swizzled_threads.csv")
    stb_df = read_nvprof_data(path + "swizzled_thread_blocks.csv")
    st_stb_df = read_nvprof_data(path + "swizzled_threads_swizzled_thread_blocks.csv")
    # v_st_df = read_nvprof_data(path + "vectorization_swizzled_threads.csv")
    # v_stb_df = read_nvprof_data(path + "vectorization_swizzled_thread_blocks.csv")
    # v_st_stb_df = read_nvprof_data(path + "vectorization_swizzled_threads_swizzled_thread_blocks.csv")
    combined_df = pd.concat([unoptimized_df, st_df, stb_df, st_stb_df], axis=1)
    combined_df.columns = ["u", "st", "stb", "st+stb"]
    fig = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["-", "ST", "STB", "ST+STB"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024") # , xlabel=""
    fig.savefig(path + "comparison.png")
    ### With Double Buffering
    db_df = read_nvprof_data(path + "double_buffering.csv")
    # db_v_df = read_nvprof_data(path + "double_buffering_vectorization.csv")
    db_st_df = read_nvprof_data(path + "double_buffering_swizzled_threads.csv")
    db_stb_df = read_nvprof_data(path + "double_buffering_swizzled_thread_blocks.csv")
    db_st_stb_df = read_nvprof_data(path + "double_buffering_swizzled_threads_swizzled_thread_blocks.csv")

    # db_v_st_df = read_nvprof_data(path + "double_buffering_vectorization_swizzled_threads.csv")
    # db_v_stb_df = read_nvprof_data(path + "double_buffering_vectorization_swizzled_thread_blocks.csv")
    # db_v_st_stb_df = read_nvprof_data(path + "double_buffering_vectorization_swizzled_threads_swizzled_thread_blocks.csv")
    cutlass_df = read_nvprof_data(base_path + "cutlass.csv")
    cublas_df = read_nvprof_data(base_path + "cublas.csv")
    fig_db = plt.figure(figsize=(10,5))
    if args.precision == 32:
        cucosma_df = read_nvprof_data(base_path + "cucosma.csv")
        combined_df_db = pd.concat([db_df, db_st_df, db_stb_df, db_st_stb_df, cutlass_df, cucosma_df, cublas_df], axis=1)
        combined_df_db.columns = ["db", "db+st", "db+stb", "db+st+stb", "cutlass", "cucosma", "cublas"]
        sns.violinplot(data=combined_df_db).set(xticklabels=["-", "ST", "STB", "ST+STB", "CUTLASS", "cuCOSMA", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024 with double buffering") # , xlabel=""
    else:
        combined_df_db = pd.concat([db_df, db_st_df, db_stb_df, db_st_stb_df, cutlass_df, cublas_df], axis=1)
        combined_df_db.columns = ["db", "db+st", "db+stb", "db+st+stb", "cutlass", "cublas"]
        sns.violinplot(data=combined_df_db).set(xticklabels=["-", "ST", "STB", "ST+STB", "CUTLASS", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024 with double buffering") # , xlabel=""
    plt.axhline(1024 * 1024 * (2 * 1024 - 1) / peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    fig_db.savefig(path + "comparison_db.png")

### (4096 x 4096) x (4096 x 4096)
if args.test == 2:
    base_path = "./performance_test_results/4096_4096_4096_" + str(args.precision) + "bit/"
    path = base_path + str(args.path) +'/'
    ### Without Double Buffering
    unoptimized_df = read_nvprof_data(path + "unoptimized.csv")
    # v_df = read_nvprof_data(path + "vectorization.csv")
    st_df = read_nvprof_data(path + "swizzled_threads.csv")
    stb_df = read_nvprof_data(path + "swizzled_thread_blocks.csv")
    st_stb_df = read_nvprof_data(path + "swizzled_threads_swizzled_thread_blocks.csv")
    # v_st_df = read_nvprof_data(path + "vectorization_swizzled_threads.csv")
    # v_stb_df = read_nvprof_data(path + "vectorization_swizzled_thread_blocks.csv")
    # v_st_stb_df = read_nvprof_data(path + "vectorization_swizzled_threads_swizzled_thread_blocks.csv")
    combined_df = pd.concat([unoptimized_df, st_df, stb_df, st_stb_df], axis=1)
    combined_df.columns = ["u", "st", "stb", "st+stb"]
    fig = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["-", "ST", "STB", "ST+STB"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096") # , xlabel=""
    fig.savefig(path + "comparison.png")
    ### With Double Buffering
    db_df = read_nvprof_data(path + "double_buffering.csv")
    db_st_df = read_nvprof_data(path + "double_buffering_swizzled_threads.csv")
    db_stb_df = read_nvprof_data(path + "double_buffering_swizzled_thread_blocks.csv")
    db_st_stb_df = read_nvprof_data(path + "double_buffering_swizzled_threads_swizzled_thread_blocks.csv")
    db_v_st_df = read_nvprof_data(path + "double_buffering_vectorization_swizzled_threads.csv")
    db_v_st_stb_df = read_nvprof_data(path + "double_buffering_vectorization_swizzled_threads_swizzled_thread_blocks.csv")
    cutlass_df = read_nvprof_data(base_path + "cutlass.csv")
    cublas_df = read_nvprof_data(base_path + "cublas.csv")
    fig_db = plt.figure(figsize=(10,5))

    if args.precision == 32:
        cucosma_df = read_nvprof_data(base_path + "cucosma.csv")
        combined_df_db = pd.concat([db_df, db_st_df, db_stb_df, db_st_stb_df, db_v_st_df, db_v_st_stb_df, cutlass_df, cucosma_df, cublas_df], axis=1)
        combined_df_db.columns = ["db", "db+st", "db+stb", "db+st+stb", "db_v_st_df", "db_v_st_stb_df", "cutlass", "cucosma", "cublas"]
        sns.violinplot(data=combined_df_db).set(xticklabels=["-", "ST", "STB", "ST+STB", "V+ST", "V+ST+STB", "CUTLASS", "cuCOSMA", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096 with double buffering") # , xlabel=""
    else:
        combined_df_db = pd.concat([db_df, db_st_df, db_stb_df, db_st_stb_df, db_v_st_df, db_v_st_stb_df, cutlass_df, cublas_df], axis=1)
        combined_df_db.columns = ["db", "db+st", "db+stb", "db+st+stb", "db_v_st_df", "db_v_st_stb_df", "cutlass", "cublas"]
        sns.violinplot(data=combined_df_db).set(xticklabels=["-", "ST", "STB", "ST+STB", "V+ST", "V+ST+STB", "CUTLASS", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096 with double buffering") # , xlabel=""

    plt.axhline(4096 * 4096 * (2 * 4096 - 1) / peak_performance, linestyle='--', label="Peak Performance")
    # plt.tick_params(axis='x', which='major', labelsize=3)
    plt.legend()
    fig_db.savefig(path + "comparison_db.png")

### (1024 x 8192) x (8192 x 1024)
if args.test == 3:
    base_path = "./performance_test_results/1024_1024_8192_" + str(args.precision) + "bit/"
    path = base_path + str(args.path) +'/'
    split_k_1 = read_nvprof_data(path + "split_k_1.csv")
    split_k_2 = read_nvprof_data(path + "split_k_2.csv")
    # split_k_3 = read_nvprof_data(path + "split_k_3.csv")
    split_k_4 = read_nvprof_data(path + "split_k_4.csv")
    # split_k_5 = read_nvprof_data(path + "split_k_5.csv")
    # split_k_6 = read_nvprof_data(path + "split_k_6.csv")
    # split_k_7 = read_nvprof_data(path + "split_k_7.csv")
    split_k_8 = read_nvprof_data(path + "split_k_8.csv")
    # split_k_9 = read_nvprof_data(path + "split_k_9.csv")
    # split_k_10 = read_nvprof_data(path + "split_k_10.csv")
    # split_k_11 = read_nvprof_data(path + "split_k_11.csv")
    # split_k_12 = read_nvprof_data(path + "split_k_12.csv")
    # split_k_13 = read_nvprof_data(path + "split_k_13.csv")
    # split_k_14 = read_nvprof_data(path + "split_k_14.csv")
    # split_k_15 = read_nvprof_data(path + "split_k_15.csv")
    split_k_16 = read_nvprof_data(path + "split_k_16.csv")

    cublas_df = read_nvprof_data(base_path + "cublas.csv")
    fig = plt.figure(figsize=(10,5))


    if args.precision == 32:
        cucosma_df = read_nvprof_data(base_path + "cucosma.csv")
        cutlass_df = read_nvprof_data(base_path + "cutlass.csv")
        combined_df_db = pd.concat([split_k_1, split_k_2, split_k_4, split_k_8, split_k_16, cutlass_df, cucosma_df, cublas_df], axis=1)
        combined_df_db.columns = ["-", "2", "4", "8", "16", "cutlass", "cucosma", "cublas"]
        sns.violinplot(data=combined_df_db).set(xticklabels=["-", "2", "4", "8", "16", "CUTLASS", "cuCOSMA", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 8192 with double buffering", xlabel="Split K")
    else:
        combined_df_db = pd.concat([split_k_1, split_k_2, split_k_4, split_k_8, split_k_16, cublas_df], axis=1)
        combined_df_db.columns = ["-", "2", "4", "8", "16", "cublas"]
        sns.violinplot(data=combined_df_db).set(xticklabels=["-", "2", "4", "8", "16", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 8192 with double buffering", xlabel="Split K")

    plt.axhline(1024 * 1024 * (2 * 8192 - 1) / peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    fig.savefig(path + "comparison.png")

### (256 x 10240) x (10240 x 256)
if args.test == 4:
    base_path = "./performance_test_results/256_256_10240_" + str(args.precision) + "bit/"
    path = base_path + str(args.path) +'/'
    split_k_1 = read_nvprof_data(path + "split_k_1.csv")
    split_k_2 = read_nvprof_data(path + "split_k_2.csv")
    split_k_4 = read_nvprof_data(path + "split_k_4.csv")
    split_k_5 = read_nvprof_data(path + "split_k_5.csv")
    split_k_8 = read_nvprof_data(path + "split_k_8.csv")
    split_k_10 = read_nvprof_data(path + "split_k_10.csv")
    split_k_16 = read_nvprof_data(path + "split_k_16.csv")
    split_k_20 = read_nvprof_data(path + "split_k_20.csv")
    split_k_40 = read_nvprof_data(path + "split_k_40.csv")

    cublas_df = read_nvprof_data(base_path + "cublas.csv")
    fig = plt.figure(figsize=(10,5))

    if args.precision == 32:
        cucosma_df = read_nvprof_data(base_path + "cucosma.csv")
        cutlass_df = read_nvprof_data(base_path + "cutlass.csv")
        combined_df_db = pd.concat([split_k_1, split_k_2, split_k_4, split_k_5, split_k_8, split_k_10, split_k_16, split_k_20, split_k_40, cutlass_df, cucosma_df, cublas_df], axis=1)
        combined_df_db.columns = ["-", "2", "4", "5", "8", "10", "16", "20", "40", "cutlass", "cucosma", "cublas"]
        sns.violinplot(data=combined_df_db).set(xticklabels=["-", "2", "4", "5", "8", "10", "16", "20", "40", "CUTLASS", "cuCOSMA", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 8192 with double buffering", xlabel="Split K")
    else:
        combined_df_db = pd.concat([split_k_1, split_k_2, split_k_4, split_k_5, split_k_8, split_k_10, split_k_16, split_k_20, split_k_40, cublas_df], axis=1)
        combined_df_db.columns = ["-", "2", "4", "5", "8", "10", "16", "20", "40", "cublas"]
        sns.violinplot(data=combined_df_db).set(xticklabels=["-", "2", "4", "5", "8", "10", "16", "20", "40", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 256, N = 256, K = 10240 with double buffering", xlabel="Split K")

    plt.axhline(256 * 256 * (2 * 10240 - 1) / peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    fig.savefig(path + "comparison.png")

    # fig2, (sk1, sk2, sk4, sk5, sk8, sk10, sk16, sk20, sk40, cublas) = plt.subplots(1, 10, constrained_layout=True, sharey=True)
    # sk1.plot(split_k_1)
    # sk2.plot(split_k_2)
    # sk4.plot(split_k_4)
    # sk5.plot(split_k_5)
    # sk8.plot(split_k_8)
    # sk10.plot(split_k_10)
    # sk16.plot(split_k_16)
    # sk20.plot(split_k_20)
    # sk40.plot(split_k_40)
    # cublas.plot(cublas_df)
    # fig2.savefig(path + "comparison2.png")


### (1024 x 1024) x (1024 x 1024)
if args.test == 5:
    base_path = "./performance_test_results/1024_1024_1024_isolated_optimizations" + str(args.precision) + "bit/"
    path = base_path + str(args.path) +'/'
    ### Without Double Buffering
    unoptimized_df = read_nvprof_data(path + "unoptimized.csv")
    st_df = read_nvprof_data(path + "swizzled_threads.csv")
    stb_df = read_nvprof_data(path + "swizzled_thread_blocks.csv")
    db_df = read_nvprof_data(path + "double_buffering.csv") / 1000

    swizzle_threads = pd.concat([unoptimized_df, st_df], axis=1)
    swizzle_threads.columns = ["u", "st"]
    fig_swizzle_threads = plt.figure()
    sns.violinplot(data=swizzle_threads).set(xticklabels=["Unoptimized", "Swizzled Threads"], ylabel="Runtime [ms]", title="M = 1024, N = 1024, K = 1024") # , xlabel=""
    fig_swizzle_threads.savefig(path + "1024_1024_1024_isolated_optimizations_swizzle_threads.png")
    
    swizzle_thread_blocks = pd.concat([unoptimized_df, stb_df], axis=1)
    swizzle_thread_blocks.columns = ["u", "stb"]
    fig_swizzle_thread_blocks = plt.figure()
    sns.violinplot(data=swizzle_thread_blocks).set(xticklabels=["Unoptimized", "Swizzled Thread Blocks"], ylabel="Runtime [ms]", title="M = 1024, N = 1024, K = 1024") # , xlabel=""
    fig_swizzle_thread_blocks.savefig(path + "1024_1024_1024_isolated_optimizations_swizzle_thread_blocks.png")

    db = pd.concat([unoptimized_df, db_df], axis=1)
    db.columns = ["u", "df"]
    fig_db = plt.figure()
    sns.violinplot(data=db).set(xticklabels=["Unoptimized", "Double Buffering"], ylabel="Runtime [ms]", title="M = 1024, N = 1024, K = 1024") # , xlabel=""
    fig_db.savefig(path + "1024_1024_1024_isolated_optimizations_db.png")

helpers.print_success("Performance plots created.", False)