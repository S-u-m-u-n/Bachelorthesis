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
args = parser.parse_args()

helpers.print_info("Creating performance plots...", False)

def read_nvprof_data(path_to_csv):
    df = pd.read_csv(path_to_csv, skiprows=3).filter(['Duration', 'Name']).iloc[1:]
    return df[df['Name'].str.contains("Thread_block_grid|dgemm")].filter(['Duration']).reset_index(drop=True).apply(pd.to_numeric, errors='coerce')

### (1024 x 1024) x (1024 x 1024)
if args.test == 1:
    ### Without Double Buffering
    unoptimized_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/unoptimized.csv")
    v_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/vectorization.csv")
    st_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/swizzled_threads.csv")
    stb_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/swizzled_thread_blocks.csv")
    st_stb_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/swizzled_threads_swizzled_thread_blocks.csv")
    v_st_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/vectorization_swizzled_threads.csv")
    v_stb_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/vectorization_swizzled_thread_blocks.csv")
    v_st_stb_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/vectorization_swizzled_threads_swizzled_thread_blocks.csv")
    combined_df = pd.concat([unoptimized_df, v_df, st_df, stb_df, st_stb_df, v_st_df, v_stb_df, v_st_stb_df], axis=1)
    combined_df.columns = ["u", "v", "st", "stb", "v_st"]
    fig = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["Unoptimized", "Vectorization", "Swizzled_Threads", "Vectorization+Swizzled_Threads"], ylabel="Runtime [ms]", title="M = 1024, N = 1024, K = 1024") # , xlabel=""
    fig.savefig("./performance_test_results/1024_1024/" + str(args.path) + "/1024_1024_comparison.png")
    ### With Double Buffering
    db_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering.csv")
    db_v_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_vectorization.csv") / 1000
    db_st_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_swizzled_threads.csv")
    db_stb_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_swizzled_thread_blocks.csv")
    db_st_stb_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_swizzled_threads_swizzled_thread_blocks.csv")
    db_v_st_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_vectorization_swizzled_threads.csv") / 1000
    db_v_stb_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_vectorization_swizzled_thread_blocks.csv") / 1000
    db_v_st_stb_df = read_nvprof_data("./performance_test_results/1024_1024/" + str(args.path) + "/double_buffering_vectorization_swizzled_threads_swizzled_thread_blocks.csv") / 1000
    cutlass_df = read_nvprof_data("./performance_test_results/1024_1024/cutlass.csv")
    cublas_df = read_nvprof_data("./performance_test_results/1024_1024/cublas.csv")
    combined_df_db = pd.concat([db_df, db_v_df, db_st_df, db_stb_df, db_st_stb_df, db_v_st_df, db_v_stb_df, db_v_st_stb_df, cutlass_df, cublas_df], axis=1)
    combined_df_db.columns = ["db", "db_v", "db_st", "db_v_st", "cutlass", "cublas"]
    fig_db = plt.figure()
    sns.violinplot(data=combined_df_db).set(xticklabels=["DB", "DB+V", "DB+ST", "DB+V+ST", "CUTLASS", "cuBLAS"], ylabel="Runtime [ms]", title="M = 1024, N = 1024, K = 1024 with double buffering") # , xlabel=""
    fig_db.savefig("./performance_test_results/1024_1024/" + str(args.path) + "/1024_1024_comparison_db.png")

### (4096 x 4096) x (4096 x 4096)
if args.test == 2:
    ### Without Double Buffering
    unoptimized_df = read_nvprof_data("./performance_test_results/4096_4096/" + str(args.path) + "/unoptimized.csv")
    v_df = read_nvprof_data("./performance_test_results/4096_4096/" + str(args.path) + "/vectorization.csv")
    st_df = read_nvprof_data("./performance_test_results/4096_4096/" + str(args.path) + "/swizzled_threads.csv")
    stb_df = read_nvprof_data("./performance_test_results/4096_4096/" + str(args.path) + "/swizzled_thread_blocks.csv")
    v_st_df = read_nvprof_data("./performance_test_results/4096_4096/" + str(args.path) + "/vectorization_swizzled_threads.csv")
    combined_df = pd.concat([unoptimized_df, v_df, st_df, v_st_df], axis=1)
    combined_df.columns = ["u", "v", "st", "stb", "v_st"]
    fig = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["Unoptimized", "Vectorization", "Swizzled_Threads", "Vectorization+Swizzled_Threads"], ylabel="Runtime [ms]", title="M = 4096, N = 4096, K = 4096") # , xlabel=""
    fig.savefig("./performance_test_results/4096_4096/" + str(args.path) + "/4096_4096_comparison.png")
    ### With Double Buffering
    db_df = read_nvprof_data("./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering.csv")
    db_v_df = read_nvprof_data("./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_vectorization.csv")
    db_st_df = read_nvprof_data("./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_swizzled_threads.csv")
    db_v_st_df = read_nvprof_data("./performance_test_results/4096_4096/" + str(args.path) + "/double_buffering_vectorization_swizzled_threads.csv")
    cutlass_df = read_nvprof_data("./performance_test_results/4096_4096/cutlass.csv")
    cublas_df = read_nvprof_data("./performance_test_results/4096_4096/cublas.csv")
    combined_df_db = pd.concat([db_df, db_v_df, db_st_df, db_v_st_df, cutlass_df, cublas_df], axis=1)
    combined_df_db.columns = ["db", "db_v", "db_st", "db_v_st", "cutlass", "cublas"]
    fig_db = plt.figure()
    sns.violinplot(data=combined_df_db).set(xticklabels=["DB", "DB+V", "DB+ST", "DB+V+ST", "CUTLASS", "cuBLAS"], ylabel="Runtime [ms]", title="M = 4096, N = 4096, K = 4096 with double buffering") # , xlabel=""
    fig_db.savefig("./performance_test_results/4096_4096/" + str(args.path) + "/4096_4096_comparison_db.png")

### (1024 x 8192) x (8192 x 1024)
if args.test == 3:
    base_path = "./performance_test_results/1024_8192_1024/"
    path = base_path + str(args.path)
    split_k_1 = read_nvprof_data(path + "/split_k_1.csv")
    split_k_2 = read_nvprof_data(path + "/split_k_2.csv")
    split_k_3 = read_nvprof_data(path + "/split_k_3.csv")
    split_k_4 = read_nvprof_data(path + "/split_k_4.csv")
    split_k_5 = read_nvprof_data(path + "/split_k_5.csv")
    split_k_6 = read_nvprof_data(path + "/split_k_6.csv")
    split_k_7 = read_nvprof_data(path + "/split_k_7.csv")
    split_k_8 = read_nvprof_data(path + "/split_k_8.csv")
    split_k_9 = read_nvprof_data(path + "/split_k_9.csv")
    split_k_10 = read_nvprof_data(path + "/split_k_10.csv")
    split_k_11 = read_nvprof_data(path + "/split_k_11.csv")
    split_k_12 = read_nvprof_data(path + "/split_k_12.csv")
    split_k_13 = read_nvprof_data(path + "/split_k_13.csv")
    split_k_14 = read_nvprof_data(path + "/split_k_14.csv")
    split_k_15 = read_nvprof_data(path + "/split_k_15.csv")
    split_k_16 = read_nvprof_data(path + "/split_k_16.csv")

    cublas_df = read_nvprof_data(base_path + "cublas.csv")
    combined_df_db = pd.concat([split_k_1, split_k_2, split_k_3, split_k_4, split_k_5, split_k_6, split_k_7, split_k_8, split_k_9, split_k_10, split_k_11, split_k_12, split_k_13, split_k_14, split_k_15, split_k_16, cublas_df], axis=1)
    combined_df_db.columns = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "cublas"]
    fig_db = plt.figure()
    sns.violinplot(data=combined_df_db).set(xticklabels=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "cuBLAS"], ylabel="Runtime [ms]", title="Split K performance: M = 1024, N = 1024, K = 8192", xlabel="Split K") # , xlabel=""
    fig_db.savefig(path + "/comparison.png")

### (256 x 10240) x (10240 x 256)
if args.test == 4:
    base_path = "./performance_test_results/256_10240_256/"
    path = base_path + str(args.path)
    split_k_1 = read_nvprof_data(path + "/split_k_1.csv")
    split_k_2 = read_nvprof_data(path + "/split_k_2.csv")
    split_k_4 = read_nvprof_data(path + "/split_k_4.csv")
    split_k_5 = read_nvprof_data(path + "/split_k_5.csv")
    split_k_8 = read_nvprof_data(path + "/split_k_8.csv")
    split_k_10 = read_nvprof_data(path + "/split_k_10.csv")
    split_k_16 = read_nvprof_data(path + "/split_k_16.csv")
    split_k_20 = read_nvprof_data(path + "/split_k_20.csv")
    split_k_40 = read_nvprof_data(path + "/split_k_40.csv")

    cublas_df = read_nvprof_data(base_path + "cublas.csv")
    combined_df_db = pd.concat([split_k_1, split_k_2, split_k_4, split_k_5, split_k_8, split_k_10, split_k_16, split_k_20, split_k_40, cublas_df], axis=1)
    combined_df_db.columns = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "cublas"]
    fig_db = plt.figure()
    sns.violinplot(data=combined_df_db).set(xticklabels=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "cuBLAS"], ylabel="Runtime [ms]", title="Split K performance: M = 256, N = 256, K = 10240", xlabel="Split K") # , xlabel=""
    fig_db.savefig(path + "/comparison.png")

helpers.print_success("Performance plots created.", False)