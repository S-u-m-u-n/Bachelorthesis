import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser
import helpers

sns.set_theme(style="whitegrid")

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, dest='repetitions', nargs="?", default=200)
parser.add_argument("-p", "--path", type=str, dest='path', nargs="?", default="./performance_test_results/final/")
parser.add_argument("-t", "--test", type=int, dest='test', choices=[1, 2, 3, 4], required=True)
args = parser.parse_args()

helpers.print_info("Creating performance plots...", False)

def read_nvprof_data(path_to_csv, warmup_already_removed=False):
    df = pd.read_csv(path_to_csv, skiprows=3)
    

    
    flag = False
    if (df['Duration'][0] == 'us'):
        flag = True
    # print(df.iloc[1, 1])
    # if (df[1])
    df = df.filter(['Duration', 'Name']).iloc[1:]

    if warmup_already_removed:
        start = 0
    else:
        start = args.repetitions / 2

    df = df[df['Name'].str.contains("Thread_block_grid|dgemm|sgemm|cosmaSgemm|cutlass")].filter(['Duration']).reset_index(drop=True).apply(pd.to_numeric, errors='coerce').iloc[int(start):, :]
    if flag:
        df /= 1000

    # print(df['Duration'])
    return df
    # return df['Duration']



### (1024 x 1024) x (1024 x 1024)
def eval_1024_1024(precision):
    path = str(args.path) + "1024_1024_1024_" + str(precision) + "bit/"

    best_avg_perf = 99999999999
    best_name = "empty"
    best = []
    for file in os.listdir(path):
        if file.endswith('csv') and not file.startswith('cu'):
            tmp = read_nvprof_data(path + str(file))
            avg_perf = tmp['Duration'].mean()
            if avg_perf < best_avg_perf:
                best_avg_perf = avg_perf
                best_name = str(file)
                best = tmp

    helpers.print_info("Best average performance: " + str(best_avg_perf), False)
    helpers.print_info("From file: " + str(best_name), False)

    peak_performance = 1024 * 1024 * (2 * 1024 - 1) / (7 * 1000 * 1000 * 1000) # OPS/(FLOPS/ms) = ms
    if precision == 32:
        peak_performance = peak_performance * 2
        precision_str = "Single precision: "
    else:
        precision_str = "Double precision: "

    # Single optimization or no optimizations (5 + 1):
    unoptimized = read_nvprof_data(path + "unoptimized.csv").values
    _st = read_nvprof_data(path + "_st.csv").values
    _stb = read_nvprof_data(path + "_stb.csv").values
    _rev = read_nvprof_data(path + "_rev.csv").values
    _col = read_nvprof_data(path + "_col.csv").values
    _dbr = read_nvprof_data(path + "_dbr.csv").values

    # print(unoptimized.values)

    combined_df = pd.concat([unoptimized, _st, _stb, _rev, _col, _dbr], axis=1)
    combined_df.columns = ["-", "st", "stb", "st+stb", "as", "sdf"]
    fig_1 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["-", "ST", "STB", "REV", "COL", "DBR"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024") # , xlabel=""
    fig_1.savefig(path + "comparison_0_1.png")

    # Two optimizations (10):
    _st_stb = read_nvprof_data(path + "_st_stb.csv").values
    _st_rev = read_nvprof_data(path + "_st_rev.csv").values
    _st_col = read_nvprof_data(path + "_st_col.csv").values
    _st_dbr = read_nvprof_data(path + "_st_dbr.csv").values

    _stb_rev = read_nvprof_data(path + "_stb_rev.csv").values
    _stb_col = read_nvprof_data(path + "_stb_col.csv").values
    _stb_dbr = read_nvprof_data(path + "_stb_dbr.csv").values
    
    _rev_col = read_nvprof_data(path + "_rev_col.csv").values
    _rev_dbr = read_nvprof_data(path + "_rev_dbr.csv").values

    _col_dbr = read_nvprof_data(path + "_col_dbr.csv").values

    combined_df = pd.concat([_st_stb, _st_rev, _st_col, _st_dbr, _stb_rev, _stb_col, _stb_dbr, _rev_col, _rev_dbr, _col_dbr], axis=1)
    fig_2 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB", "ST+REV", "ST+COL", "ST+DBR", "STB+REV", "STB+COL", "STB+DBR", "REV+COL", "REV+DBR", "COL+DBR"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024") # , xlabel=""
    fig_2.savefig(path + "comparison_2.png")


    # Three optimizations (10):
    _st_stb_rev = read_nvprof_data(path + "_st_stb_rev.csv")
    _st_stb_col = read_nvprof_data(path + "_st_stb_col.csv")
    _st_stb_dbr = read_nvprof_data(path + "_st_stb_dbr.csv")

    _st_rev_col = read_nvprof_data(path + "_st_rev_col.csv")
    _st_rev_dbr = read_nvprof_data(path + "_st_rev_dbr.csv")

    _st_col_dbr = read_nvprof_data(path + "_st_col_dbr.csv")

    _stb_rev_col = read_nvprof_data(path + "_stb_rev_col.csv")
    _stb_rev_dbr = read_nvprof_data(path + "_stb_rev_dbr.csv")

    _stb_col_dbr = read_nvprof_data(path + "_stb_col_dbr.csv")
    _rev_col_dbr = read_nvprof_data(path + "_rev_col_dbr.csv")

    combined_df = pd.concat([_st_stb_rev, _st_stb_col, _st_stb_dbr, _st_rev_col, _st_rev_dbr, _st_col_dbr, _stb_rev_col, _stb_rev_dbr, _stb_col_dbr, _rev_col_dbr], axis=1)
    fig_3 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB+REV", "ST+STB+COL", "ST+STB+DBR", "ST+REV+COL", "ST+REV+DBR", "ST+COL+DBR", "STB+REV+COL", "STB+REV+DBR", "STB+COL+DBR", "REV+COL+DBR"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024") # , xlabel=""
    fig_3.savefig(path + "comparison_3.png")


    # Four or five optimizations (5 + 1):
    _st_stb_rev_col = read_nvprof_data(path + "_st_stb_rev_col.csv")
    _st_stb_rev_dbr = read_nvprof_data(path + "_st_stb_rev_dbr.csv")

    _st_stb_col_dbr = read_nvprof_data(path + "_st_stb_col_dbr.csv")
    _st_rev_col_dbr = read_nvprof_data(path + "_st_rev_col_dbr.csv")

    _stb_rev_col_dbr = read_nvprof_data(path + "_stb_rev_col_dbr.csv")

    _st_stb_rev_col_dbr = read_nvprof_data(path + "_st_stb_rev_col_dbr.csv")

    combined_df = pd.concat([_st_stb_rev_col, _st_stb_rev_dbr, _st_stb_col_dbr, _st_rev_col_dbr, _stb_rev_col_dbr, _st_stb_rev_col_dbr], axis=1)
    fig_4 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB+REV+COL", "ST+STB+REV+DBR", "ST+STB+COL+DBR", "ST+REV+COL+DBR", "STB+REV+COL+DBR", "ST+STB+REV+COL+DBR"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024") # , xlabel=""
    fig_4.savefig(path + "comparison_4.png")


    ##########################################################################################################################################
    ######################################################       Double Buffering       ######################################################
    ##########################################################################################################################################
    # Single optimization or no optimizations (6 + 1):
    dbs = read_nvprof_data(path + "dbs.csv")
    dbs_st = read_nvprof_data(path + "dbs_st.csv")
    dbs_stb = read_nvprof_data(path + "dbs_stb.csv")
    dbs_rev = read_nvprof_data(path + "dbs_rev.csv")
    dbs_col = read_nvprof_data(path + "dbs_col.csv")
    dbs_dbr = read_nvprof_data(path + "dbs_dbr.csv")
    dbs_npo = read_nvprof_data(path + "dbs_npo.csv")

    combined_df = pd.concat([dbs, dbs_st, dbs_stb, dbs_rev, dbs_col, dbs_dbr, dbs_npo], axis=1)
    # combined_df.columns = ["-", "st", "stb", "st+stb"]
    dbs_fig_1 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["-", "ST", "STB", "REV", "COL", "DBR", "NPO"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024 with shared memory double buffering") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    dbs_fig_1.savefig(path + "dbs_comparison_0_1.png")

    # Two optimizations (15):
    dbs_st_stb = read_nvprof_data(path + "dbs_st_stb.csv")
    dbs_st_rev = read_nvprof_data(path + "dbs_st_rev.csv")
    dbs_st_col = read_nvprof_data(path + "dbs_st_col.csv")
    dbs_st_dbr = read_nvprof_data(path + "dbs_st_dbr.csv")
    dbs_st_npo = read_nvprof_data(path + "dbs_st_npo.csv")

    dbs_stb_rev = read_nvprof_data(path + "dbs_stb_rev.csv")
    dbs_stb_col = read_nvprof_data(path + "dbs_stb_col.csv")
    dbs_stb_dbr = read_nvprof_data(path + "dbs_stb_dbr.csv")
    dbs_stb_npo = read_nvprof_data(path + "dbs_stb_npo.csv")
    
    dbs_rev_col = read_nvprof_data(path + "dbs_rev_col.csv")
    dbs_rev_dbr = read_nvprof_data(path + "dbs_rev_dbr.csv")
    dbs_rev_npo = read_nvprof_data(path + "dbs_rev_npo.csv")

    dbs_col_dbr = read_nvprof_data(path + "dbs_col_dbr.csv")
    dbs_col_npo = read_nvprof_data(path + "dbs_col_npo.csv")

    dbs_dbr_npo = read_nvprof_data(path + "dbs_dbr_npo.csv")


    combined_df = pd.concat([dbs_st_stb, dbs_st_rev, dbs_st_col, dbs_st_dbr, dbs_st_npo, dbs_stb_rev, dbs_stb_col, dbs_stb_dbr, dbs_stb_npo, dbs_rev_col, dbs_rev_dbr, dbs_rev_npo, dbs_col_dbr, dbs_col_npo, dbs_dbr_npo], axis=1)
    dbs_fig_2 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB", "ST+REV", "ST+COL", "ST+DBR", "ST+NPO", "STB+REV", "STB+COL", "STB+DBR", "STB+NPO", "REV+COL", "REV+DBR", "COL+DBR", "COL+NPO", "DBR+NPO"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024 with shared memory double buffering") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    dbs_fig_2.savefig(path + "dbs_comparison_2.png")


    # Three optimizations (20):
    dbs_st_stb_rev = read_nvprof_data(path + "dbs_st_stb_rev.csv")
    dbs_st_stb_col = read_nvprof_data(path + "dbs_st_stb_col.csv")
    dbs_st_stb_dbr = read_nvprof_data(path + "dbs_st_stb_dbr.csv")
    dbs_st_stb_npo = read_nvprof_data(path + "dbs_st_stb_npo.csv")

    dbs_st_rev_col = read_nvprof_data(path + "dbs_st_rev_col.csv")
    dbs_st_rev_dbr = read_nvprof_data(path + "dbs_st_rev_dbr.csv")
    dbs_st_rev_npo = read_nvprof_data(path + "dbs_st_rev_npo.csv")

    dbs_st_col_dbr = read_nvprof_data(path + "dbs_st_col_dbr.csv")
    dbs_st_col_npo = read_nvprof_data(path + "dbs_st_col_npo.csv")

    dbs_st_dbr_npo = read_nvprof_data(path + "dbs_st_dbr_npo.csv")

    dbs_stb_rev_col = read_nvprof_data(path + "dbs_stb_rev_col.csv")
    dbs_stb_rev_dbr = read_nvprof_data(path + "dbs_stb_rev_dbr.csv")
    dbs_stb_rev_npo = read_nvprof_data(path + "dbs_stb_rev_npo.csv")

    dbs_stb_col_dbr = read_nvprof_data(path + "dbs_stb_col_dbr.csv")
    dbs_stb_col_npo = read_nvprof_data(path + "dbs_stb_col_npo.csv")

    dbs_stb_dbr_npo = read_nvprof_data(path + "dbs_stb_dbr_npo.csv")

    dbs_rev_col_dbr = read_nvprof_data(path + "dbs_rev_col_dbr.csv")
    dbs_rev_col_npo = read_nvprof_data(path + "dbs_rev_col_npo.csv")

    dbs_rev_dbr_npo = read_nvprof_data(path + "dbs_rev_dbr_npo.csv")

    dbs_col_dbr_npo = read_nvprof_data(path + "dbs_col_dbr_npo.csv")


    combined_df = pd.concat([dbs_st_stb_rev, dbs_st_stb_col, dbs_st_stb_dbr, dbs_st_stb_npo, dbs_st_rev_col, dbs_st_rev_dbr, dbs_st_rev_npo, dbs_st_col_dbr, dbs_st_col_npo, dbs_st_dbr_npo,
                            dbs_stb_rev_col, dbs_stb_rev_dbr, dbs_stb_rev_npo, dbs_stb_col_dbr, dbs_stb_col_npo, dbs_stb_dbr_npo, dbs_rev_col_dbr, dbs_rev_col_npo, dbs_rev_dbr_npo, dbs_col_dbr_npo], axis=1)
    dbs_fig_3 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB+REV", "ST+STB+COL", "ST+STB+DBR", "ST+STB+NPO", "ST+REV+COL", "ST+REV+DBR", "ST+REV+NPO", "ST+COL+DBR", "ST+COL+NPO", "ST+DBR+NPO",
                                                    "STB+REV+COL", "STB+REV+DBR", "STB_REV_NPO", "STB+COL+DBR", "STB+COL+NPO", "STB+DBR+NPO", "REV+COL+DBR", "REV+COL+NPO", "REV+DBR+NPO", "COL+DBR+NPO"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024 with shared memory double buffering") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    dbs_fig_3.savefig(path + "dbs_comparison_3.png")


    # Four optimizations (15):
    dbs_st_stb_rev_col = read_nvprof_data(path + "dbs_st_stb_rev_col.csv")
    dbs_st_stb_rev_dbr = read_nvprof_data(path + "dbs_st_stb_rev_dbr.csv")
    dbs_st_stb_rev_npo = read_nvprof_data(path + "dbs_st_stb_rev_npo.csv")

    dbs_st_stb_col_dbr = read_nvprof_data(path + "dbs_st_stb_col_dbr.csv")
    dbs_st_stb_col_npo = read_nvprof_data(path + "dbs_st_stb_col_npo.csv")

    dbs_st_stb_dbr_npo = read_nvprof_data(path + "dbs_st_stb_dbr_npo.csv")

    dbs_st_rev_col_dbr = read_nvprof_data(path + "dbs_st_rev_col_dbr.csv")
    dbs_st_rev_col_npo = read_nvprof_data(path + "dbs_st_rev_col_npo.csv")

    dbs_st_rev_dbr_npo = read_nvprof_data(path + "dbs_st_rev_dbr_npo.csv")
    
    dbs_st_col_dbr_npo = read_nvprof_data(path + "dbs_st_col_dbr_npo.csv")

    dbs_stb_rev_col_dbr = read_nvprof_data(path + "dbs_stb_rev_col_dbr.csv")
    dbs_stb_rev_col_npo = read_nvprof_data(path + "dbs_stb_rev_col_npo.csv")

    dbs_stb_col_dbr_npo = read_nvprof_data(path + "dbs_stb_col_dbr_npo.csv")

    dbs_stb_rev_dbr_npo = read_nvprof_data(path + "dbs_stb_rev_dbr_npo.csv")

    dbs_rev_col_dbr_npo = read_nvprof_data(path + "dbs_rev_col_dbr_npo.csv")


    combined_df = pd.concat([dbs_st_stb_rev_col, dbs_st_stb_rev_dbr, dbs_st_stb_rev_npo, dbs_st_stb_col_dbr, dbs_st_stb_col_npo,
                            dbs_st_stb_dbr_npo, dbs_st_rev_col_dbr, dbs_st_rev_col_npo, dbs_st_rev_dbr_npo, dbs_st_col_dbr_npo,
                            dbs_stb_rev_col_dbr, dbs_stb_rev_col_npo, dbs_stb_col_dbr_npo, dbs_stb_rev_dbr_npo, dbs_rev_col_dbr_npo], axis=1)
    dbs_fig_4 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB+REV+COL", "ST+STB+REV+DBR", "ST+STB+REV+NPO", "ST+STB+COL+DBR", "ST+STB+COL+NPO",
                                                    "ST+STB+DBR+NPO", "ST+REV+COL+DBR", "ST+REV+COL+NPO", "ST+REV+DBR+NPO", "ST+COL+DBR+NPO",
                                                    "STB+REV+COL+DBR", "STB+REV+COL+NPO", "STB+COL+DBR+NPO", "STB+REV+DBR+NPO", "REV+COL+DBR+NPO"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024 with shared memory double buffering") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    dbs_fig_4.savefig(path + "dbs_comparison_4.png")

    # Five or six optimizations (6 + 1):
    dbs_st_stb_rev_col_dbr = read_nvprof_data(path + "dbs_st_stb_rev_col_dbr.csv")
    dbs_st_stb_rev_col_npo = read_nvprof_data(path + "dbs_st_stb_rev_col_npo.csv")
    dbs_st_stb_rev_dbr_npo = read_nvprof_data(path + "dbs_st_stb_rev_dbr_npo.csv")
    dbs_st_stb_col_dbr_npo = read_nvprof_data(path + "dbs_st_stb_col_dbr_npo.csv")
    dbs_st_rev_col_dbr_npo = read_nvprof_data(path + "dbs_st_rev_col_dbr_npo.csv")
    dbs_stb_rev_col_dbr_npo = read_nvprof_data(path + "dbs_stb_rev_col_dbr_npo.csv")
    
    dbs_st_stb_rev_col_dbr_npo = read_nvprof_data(path + "dbs_st_stb_rev_col_dbr_npo.csv")

    combined_df = pd.concat([dbs_st_stb_rev_col_dbr, dbs_st_stb_rev_col_npo, dbs_st_stb_rev_dbr_npo, dbs_st_stb_col_dbr_npo, dbs_st_rev_col_dbr_npo, dbs_stb_rev_col_dbr_npo, dbs_st_stb_rev_col_dbr_npo], axis=1)
    dbs_fig_5 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB+REV+COL+DBR", "ST+STB+REV+COL+NPO", "ST+STB+REV+DBR+NPO", "ST+STB+COL+DBR+NPO", "ST+REV+COL+DBR+NPO", "STB+REV+COL+DBR+NPO", "ST+STB+REV+COL+DBR+NPO"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024 with shared memory double buffering") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    dbs_fig_5.savefig(path + "dbs_comparison_5.png")

    ###############################################################################
    cublas = read_nvprof_data(path + "cublas.csv")
    fig_best = plt.figure(figsize=(10,5))
    if args.precision == 32:
        cutlass = read_nvprof_data(path + "cutlass.csv", True)
        cucosma = read_nvprof_data(path + "cucosma.csv", True)
        combined_df = pd.concat([best, cutlass, cucosma, cublas], axis=1)
        # combined_df_db.columns = ["db", "db+st", "db+stb", "db+st+stb", "cutlass", "cucosma", "cublas"]
        sns.violinplot(data=combined_df).set(xticklabels=["DaCe", "cuCOSMA", "CUTLASS", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024") # , xlabel=""
    else:
        cutlass = read_nvprof_data(path + "cutlass.csv")
        combined_df_db = pd.concat([best, cutlass, cublas], axis=1)
        # combined_df_db.columns = ["db", "db+st", "db+stb", "db+st+stb", "cutlass", "cublas"]
        sns.violinplot(data=combined_df_db).set(xticklabels=["DaCe", "CUTLASS", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 1024") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    fig_db.savefig(path + "best_comparison.png")

### (4096 x 4096) x (4096 x 4096)
def eval_4096_4096(precision):
    path = str(args.path) + "4096_4096_4096_" + str(precision) + "bit/"

    best_avg_perf = 99999999999
    best_name = "empty"
    best = []
    for file in os.listdir(path):
        if not file.startswith('cu'):
            tmp = read_nvprof_data(path + str(file))
            avg_perf = tmp.mean()
            if avg_perf < best_avg_perf:
                best_avg_perf = avg_perf
                best_name = str(file)
                best = tmp

    helpers.print_info("Best average performance: " + str(best_avg_perf), False)
    helpers.print_info("From file: " + best_name, False)
    print(best)

    peak_performance = 4096 * 4096 * (2 * 4096 - 1) / (7 * 1000 * 1000 * 1000) # OPS/(FLOPS/ms) = ms
    if precision == 32:
        peak_performance = peak_performance * 2
        precision_str = "Single precision: "
    else:
        precision_str = "Double precision: "

    # Single optimization or no optimizations (5 + 1):
    unoptimized = read_nvprof_data(path + "unoptimized.csv")
    _st = read_nvprof_data(path + "_st.csv")
    _stb = read_nvprof_data(path + "_stb.csv")
    _rev = read_nvprof_data(path + "_rev.csv")
    _col = read_nvprof_data(path + "_col.csv")
    _dbr = read_nvprof_data(path + "_dbr.csv")

    combined_df = pd.concat([unoptimized, _st, _stb, _rev, _col, _dbr], axis=1)
    # combined_df.columns = ["-", "st", "stb", "st+stb"]
    fig_1 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["-", "ST", "STB", "REV", "COL", "DBR"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096") # , xlabel=""
    fig_1.savefig(path + "comparison_0_1.png")

    # Two optimizations (10):
    _st_stb = read_nvprof_data(path + "_st_stb.csv")
    _st_rev = read_nvprof_data(path + "_st_rev.csv")
    _st_col = read_nvprof_data(path + "_st_col.csv")
    _st_dbr = read_nvprof_data(path + "_st_dbr.csv")

    _stb_rev = read_nvprof_data(path + "_stb_rev.csv")
    _stb_col = read_nvprof_data(path + "_stb_col.csv")
    _stb_dbr = read_nvprof_data(path + "_stb_dbr.csv")
    
    _rev_col = read_nvprof_data(path + "_rev_col.csv")
    _rev_dbr = read_nvprof_data(path + "_rev_dbr.csv")

    _col_dbr = read_nvprof_data(path + "_col_dbr.csv")

    combined_df = pd.concat([_st_stb, _st_rev, _st_col, _st_dbr, _stb_rev, _stb_col, _stb_dbr, _rev_col, _rev_dbr, _col_dbr], axis=1)
    fig_2 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB", "ST+REV", "ST+COL", "ST+DBR", "STB+REV", "STB+COL", "STB+DBR", "REV+COL", "REV+DBR", "COL+DBR"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096") # , xlabel=""
    fig_2.savefig(path + "comparison_2.png")


    # Three optimizations (10):
    _st_stb_rev = read_nvprof_data(path + "_st_stb_rev.csv")
    _st_stb_col = read_nvprof_data(path + "_st_stb_col.csv")
    _st_stb_dbr = read_nvprof_data(path + "_st_stb_dbr.csv")

    _st_rev_col = read_nvprof_data(path + "_st_rev_col.csv")
    _st_rev_dbr = read_nvprof_data(path + "_st_rev_dbr.csv")

    _st_col_dbr = read_nvprof_data(path + "_st_col_dbr.csv")

    _stb_rev_col = read_nvprof_data(path + "_stb_rev_col.csv")
    _stb_rev_dbr = read_nvprof_data(path + "_stb_rev_dbr.csv")

    _stb_col_dbr = read_nvprof_data(path + "_stb_col_dbr.csv")
    _rev_col_dbr = read_nvprof_data(path + "_rev_col_dbr.csv")

    combined_df = pd.concat([_st_stb_rev, _st_stb_col, _st_stb_dbr, _st_rev_col, _st_rev_dbr, _st_col_dbr, _stb_rev_col, _stb_rev_dbr, _stb_col_dbr, _rev_col_dbr], axis=1)
    fig_3 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB+REV", "ST+STB+COL", "ST+STB+DBR", "ST+REV+COL", "ST+REV+DBR", "ST+COL+DBR", "STB+REV+COL", "STB+REV+DBR", "STB+COL+DBR", "REV+COL+DBR"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096") # , xlabel=""
    fig_3.savefig(path + "comparison_3.png")


    # Four or five optimizations (5 + 1):
    _st_stb_rev_col = read_nvprof_data(path + "_st_stb_rev_col.csv")
    _st_stb_rev_dbr = read_nvprof_data(path + "_st_stb_rev_dbr.csv")

    _st_stb_col_dbr = read_nvprof_data(path + "_st_stb_col_dbr.csv")
    _st_rev_col_dbr = read_nvprof_data(path + "_st_rev_col_dbr.csv")

    _stb_rev_col_dbr = read_nvprof_data(path + "_stb_rev_col_dbr.csv")

    _st_stb_rev_col_dbr = read_nvprof_data(path + "_st_stb_rev_col_dbr.csv")

    combined_df = pd.concat([_st_stb_rev_col, _st_stb_rev_dbr, _st_stb_col_dbr, _st_rev_col_dbr, _stb_rev_col_dbr, _st_stb_rev_col_dbr], axis=1)
    fig_4 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB+REV+COL", "ST+STB+REV+DBR", "ST+STB+COL+DBR", "ST+REV+COL+DBR", "STB+REV+COL+DBR", "ST+STB+REV+COL+DBR"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096") # , xlabel=""
    fig_4.savefig(path + "comparison_4.png")


    ##########################################################################################################################################
    ######################################################       Double Buffering       ######################################################
    ##########################################################################################################################################
    # Single optimization or no optimizations (6 + 1):
    dbs = read_nvprof_data(path + "dbs.csv")
    dbs_st = read_nvprof_data(path + "dbs_st.csv")
    dbs_stb = read_nvprof_data(path + "dbs_stb.csv")
    dbs_rev = read_nvprof_data(path + "dbs_rev.csv")
    dbs_col = read_nvprof_data(path + "dbs_col.csv")
    dbs_dbr = read_nvprof_data(path + "dbs_dbr.csv")
    dbs_npo = read_nvprof_data(path + "dbs_npo.csv")

    combined_df = pd.concat([dbs, dbs_st, dbs_stb, dbs_rev, dbs_col, dbs_dbr, dbs_npo], axis=1)
    # combined_df.columns = ["-", "st", "stb", "st+stb"]
    dbs_fig_1 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["-", "ST", "STB", "REV", "COL", "DBR", "NPO"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096 with shared memory double buffering") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    dbs_fig_1.savefig(path + "dbs_comparison_0_1.png")

    # Two optimizations (15):
    dbs_st_stb = read_nvprof_data(path + "dbs_st_stb.csv")
    dbs_st_rev = read_nvprof_data(path + "dbs_st_rev.csv")
    dbs_st_col = read_nvprof_data(path + "dbs_st_col.csv")
    dbs_st_dbr = read_nvprof_data(path + "dbs_st_dbr.csv")
    dbs_st_npo = read_nvprof_data(path + "dbs_st_npo.csv")

    dbs_stb_rev = read_nvprof_data(path + "dbs_stb_rev.csv")
    dbs_stb_col = read_nvprof_data(path + "dbs_stb_col.csv")
    dbs_stb_dbr = read_nvprof_data(path + "dbs_stb_dbr.csv")
    dbs_stb_npo = read_nvprof_data(path + "dbs_stb_npo.csv")
    
    dbs_rev_col = read_nvprof_data(path + "dbs_rev_col.csv")
    dbs_rev_dbr = read_nvprof_data(path + "dbs_rev_dbr.csv")
    dbs_rev_npo = read_nvprof_data(path + "dbs_rev_npo.csv")

    dbs_col_dbr = read_nvprof_data(path + "dbs_col_dbr.csv")
    dbs_col_npo = read_nvprof_data(path + "dbs_col_npo.csv")

    dbs_dbr_npo = read_nvprof_data(path + "dbs_dbr_npo.csv")


    combined_df = pd.concat([dbs_st_stb, dbs_st_rev, dbs_st_col, dbs_st_dbr, dbs_st_npo, dbs_stb_rev, dbs_stb_col, dbs_stb_dbr, dbs_stb_npo, dbs_rev_col, dbs_rev_dbr, dbs_rev_npo, dbs_col_dbr, dbs_col_npo, dbs_dbr_npo], axis=1)
    dbs_fig_2 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB", "ST+REV", "ST+COL", "ST+DBR", "ST+NPO", "STB+REV", "STB+COL", "STB+DBR", "STB+NPO", "REV+COL", "REV+DBR", "COL+DBR", "COL+NPO", "DBR+NPO"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096 with shared memory double buffering") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    dbs_fig_2.savefig(path + "dbs_comparison_2.png")


    # Three optimizations (20):
    dbs_st_stb_rev = read_nvprof_data(path + "dbs_st_stb_rev.csv")
    dbs_st_stb_col = read_nvprof_data(path + "dbs_st_stb_col.csv")
    dbs_st_stb_dbr = read_nvprof_data(path + "dbs_st_stb_dbr.csv")
    dbs_st_stb_npo = read_nvprof_data(path + "dbs_st_stb_npo.csv")

    dbs_st_rev_col = read_nvprof_data(path + "dbs_st_rev_col.csv")
    dbs_st_rev_dbr = read_nvprof_data(path + "dbs_st_rev_dbr.csv")
    dbs_st_rev_npo = read_nvprof_data(path + "dbs_st_rev_npo.csv")

    dbs_st_col_dbr = read_nvprof_data(path + "dbs_st_col_dbr.csv")
    dbs_st_col_npo = read_nvprof_data(path + "dbs_st_col_npo.csv")

    dbs_st_dbr_npo = read_nvprof_data(path + "dbs_st_dbr_npo.csv")

    dbs_stb_rev_col = read_nvprof_data(path + "dbs_stb_rev_col.csv")
    dbs_stb_rev_dbr = read_nvprof_data(path + "dbs_stb_rev_dbr.csv")
    dbs_stb_rev_npo = read_nvprof_data(path + "dbs_stb_rev_npo.csv")

    dbs_stb_col_dbr = read_nvprof_data(path + "dbs_stb_col_dbr.csv")
    dbs_stb_col_npo = read_nvprof_data(path + "dbs_stb_col_npo.csv")

    dbs_stb_dbr_npo = read_nvprof_data(path + "dbs_stb_dbr_npo.csv")

    dbs_rev_col_dbr = read_nvprof_data(path + "dbs_rev_col_dbr.csv")
    dbs_rev_col_npo = read_nvprof_data(path + "dbs_rev_col_npo.csv")

    dbs_rev_dbr_npo = read_nvprof_data(path + "dbs_rev_dbr_npo.csv")

    dbs_col_dbr_npo = read_nvprof_data(path + "dbs_col_dbr_npo.csv")


    combined_df = pd.concat([dbs_st_stb_rev, dbs_st_stb_col, dbs_st_stb_dbr, dbs_st_stb_npo, dbs_st_rev_col, dbs_st_rev_dbr, dbs_st_rev_npo, dbs_st_col_dbr, dbs_st_col_npo, dbs_st_dbr_npo,
                            dbs_stb_rev_col, dbs_stb_rev_dbr, dbs_stb_rev_npo, dbs_stb_col_dbr, dbs_stb_col_npo, dbs_stb_dbr_npo, dbs_rev_col_dbr, dbs_rev_col_npo, dbs_rev_dbr_npo, dbs_col_dbr_npo], axis=1)
    dbs_fig_3 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB+REV", "ST+STB+COL", "ST+STB+DBR", "ST+STB+NPO", "ST+REV+COL", "ST+REV+DBR", "ST+REV+NPO", "ST+COL+DBR", "ST+COL+NPO", "ST+DBR+NPO",
                                                    "STB+REV+COL", "STB+REV+DBR", "STB_REV_NPO", "STB+COL+DBR", "STB+COL+NPO", "STB+DBR+NPO", "REV+COL+DBR", "REV+COL+NPO", "REV+DBR+NPO", "COL+DBR+NPO"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096 with shared memory double buffering") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    dbs_fig_3.savefig(path + "dbs_comparison_3.png")


    # Four optimizations (15):
    dbs_st_stb_rev_col = read_nvprof_data(path + "dbs_st_stb_rev_col.csv")
    dbs_st_stb_rev_dbr = read_nvprof_data(path + "dbs_st_stb_rev_dbr.csv")
    dbs_st_stb_rev_npo = read_nvprof_data(path + "dbs_st_stb_rev_npo.csv")

    dbs_st_stb_col_dbr = read_nvprof_data(path + "dbs_st_stb_col_dbr.csv")
    dbs_st_stb_col_npo = read_nvprof_data(path + "dbs_st_stb_col_npo.csv")

    dbs_st_stb_dbr_npo = read_nvprof_data(path + "dbs_st_stb_dbr_npo.csv")

    dbs_st_rev_col_dbr = read_nvprof_data(path + "dbs_st_rev_col_dbr.csv")
    dbs_st_rev_col_npo = read_nvprof_data(path + "dbs_st_rev_col_npo.csv")

    dbs_st_rev_dbr_npo = read_nvprof_data(path + "dbs_st_rev_dbr_npo.csv")
    
    dbs_st_col_dbr_npo = read_nvprof_data(path + "dbs_st_col_dbr_npo.csv")

    dbs_stb_rev_col_dbr = read_nvprof_data(path + "dbs_stb_rev_col_dbr.csv")
    dbs_stb_rev_col_npo = read_nvprof_data(path + "dbs_stb_rev_col_npo.csv")

    dbs_stb_col_dbr_npo = read_nvprof_data(path + "dbs_stb_col_dbr_npo.csv")

    dbs_stb_rev_dbr_npo = read_nvprof_data(path + "dbs_stb_rev_dbr_npo.csv")

    dbs_rev_col_dbr_npo = read_nvprof_data(path + "dbs_rev_col_dbr_npo.csv")


    combined_df = pd.concat([dbs_st_stb_rev_col, dbs_st_stb_rev_dbr, dbs_st_stb_rev_npo, dbs_st_stb_col_dbr, dbs_st_stb_col_npo,
                            dbs_st_stb_dbr_npo, dbs_st_rev_col_dbr, dbs_st_rev_col_npo, dbs_st_rev_dbr_npo, dbs_st_col_dbr_npo,
                            dbs_stb_rev_col_dbr, dbs_stb_rev_col_npo, dbs_stb_col_dbr_npo, dbs_stb_rev_dbr_npo, dbs_rev_col_dbr_npo], axis=1)
    dbs_fig_4 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB+REV+COL", "ST+STB+REV+DBR", "ST+STB+REV+NPO", "ST+STB+COL+DBR", "ST+STB+COL+NPO",
                                                    "ST+STB+DBR+NPO", "ST+REV+COL+DBR", "ST+REV+COL+NPO", "ST+REV+DBR+NPO", "ST+COL+DBR+NPO",
                                                    "STB+REV+COL+DBR", "STB+REV+COL+NPO", "STB+COL+DBR+NPO", "STB+REV+DBR+NPO", "REV+COL+DBR+NPO"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096 with shared memory double buffering") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    dbs_fig_4.savefig(path + "dbs_comparison_4.png")

    # Five or six optimizations (6 + 1):
    dbs_st_stb_rev_col_dbr = read_nvprof_data(path + "dbs_st_stb_rev_col_dbr.csv")
    dbs_st_stb_rev_col_npo = read_nvprof_data(path + "dbs_st_stb_rev_col_npo.csv")
    dbs_st_stb_rev_dbr_npo = read_nvprof_data(path + "dbs_st_stb_rev_dbr_npo.csv")
    dbs_st_stb_col_dbr_npo = read_nvprof_data(path + "dbs_st_stb_col_dbr_npo.csv")
    dbs_st_rev_col_dbr_npo = read_nvprof_data(path + "dbs_st_rev_col_dbr_npo.csv")
    dbs_stb_rev_col_dbr_npo = read_nvprof_data(path + "dbs_stb_rev_col_dbr_npo.csv")
    
    dbs_st_stb_rev_col_dbr_npo = read_nvprof_data(path + "dbs_st_stb_rev_col_dbr_npo.csv")

    combined_df = pd.concat([dbs_st_stb_rev_col_dbr, dbs_st_stb_rev_col_npo, dbs_st_stb_rev_dbr_npo, dbs_st_stb_col_dbr_npo, dbs_st_rev_col_dbr_npo, dbs_stb_rev_col_dbr_npo, dbs_st_stb_rev_col_dbr_npo], axis=1)
    dbs_fig_5 = plt.figure()
    sns.violinplot(data=combined_df).set(xticklabels=["ST+STB+REV+COL+DBR", "ST+STB+REV+COL+NPO", "ST+STB+REV+DBR+NPO", "ST+STB+COL+DBR+NPO", "ST+REV+COL+DBR+NPO", "STB+REV+COL+DBR+NPO", "ST+STB+REV+COL+DBR+NPO"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096 with shared memory double buffering") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    dbs_fig_5.savefig(path + "dbs_comparison_5.png")

    ###############################################################################
    cublas = read_nvprof_data(path + "cublas.csv")
    fig_best = plt.figure(figsize=(10,5))
    if args.precision == 32:
        cutlass = read_nvprof_data(path + "cutlass.csv", True)
        cucosma = read_nvprof_data(path + "cucosma.csv", True)
        combined_df = pd.concat([best, cutlass, cucosma, cublas], axis=1)
        # combined_df_db.columns = ["db", "db+st", "db+stb", "db+st+stb", "cutlass", "cucosma", "cublas"]
        sns.violinplot(data=combined_df).set(xticklabels=["DaCe", "cuCOSMA", "CUTLASS", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096") # , xlabel=""
    else:
        cutlass = read_nvprof_data(path + "cutlass.csv")
        combined_df_db = pd.concat([best, cutlass, cublas], axis=1)
        # combined_df_db.columns = ["db", "db+st", "db+stb", "db+st+stb", "cutlass", "cublas"]
        sns.violinplot(data=combined_df_db).set(xticklabels=["DaCe", "CUTLASS", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096") # , xlabel=""
    plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    plt.legend()
    fig_db.savefig(path + "best_comparison.png")


### (1024 x 8192) x (8192 x 1024)
def eval_1024_8192_1024(precision):
    path = str(args.path) + "1024_1024_8192_" + str(precision) + "bit/"

    best_avg_perf = 99999999999
    best_name = "empty"
    best = []
    for file in os.listdir(path):
        if not file.startswith('cu'):
            tmp = read_nvprof_data(path + str(file))
            avg_perf = tmp.mean()
            if avg_perf < best_avg_perf:
                best_avg_perf = avg_perf
                best_name = str(file)
                best = tmp

    helpers.print_info("Best average performance: " + best_avg_perf, False)
    helpers.print_info("From file: " + best_name, False)


    peak_performance = 1024 * 1024 * (2 * 8192 - 1) / (7 * 1000 * 1000 * 1000) # OPS/(FLOPS/ms) = ms
    if precision == 32:
        peak_performance = peak_performance * 2
        precision_str = "Single precision: "
    else:
        precision_str = "Double precision: "


    # split_k_1 = read_nvprof_data(path + "split_k_1.csv")
    # split_k_2 = read_nvprof_data(path + "split_k_2.csv")
    # split_k_4 = read_nvprof_data(path + "split_k_4.csv")
    # split_k_8 = read_nvprof_data(path + "split_k_8.csv")
    # split_k_16 = read_nvprof_data(path + "split_k_16.csv")

    ###############################################################################
    # cutlass = read_nvprof_data(base_path + "cutlass.csv")
    # cublas = read_nvprof_data(base_path + "cublas.csv")
    # fig_best = plt.figure(figsize=(10,5))
    # if args.precision == 32:
    #     cucosma = read_nvprof_data(base_path + "cucosma.csv")
    #     combined_df = pd.concat([best, cutlass, cucosma, cublas], axis=1)
    #     # combined_df_db.columns = ["db", "db+st", "db+stb", "db+st+stb", "cutlass", "cucosma", "cublas"]
    #     sns.violinplot(data=combined_df).set(xticklabels=["DaCe", "cuCOSMA", "CUTLASS", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096") # , xlabel=""
    # else:
    #     combined_df_db = pd.concat([best, cutlass, cublas], axis=1)
    #     # combined_df_db.columns = ["db", "db+st", "db+stb", "db+st+stb", "cutlass", "cublas"]
    #     sns.violinplot(data=combined_df_db).set(xticklabels=["DaCe", "CUTLASS", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 4096, N = 4096, K = 4096") # , xlabel=""

    # # cublas_df = read_nvprof_data(base_path + "cublas.csv")
    # fig = plt.figure(figsize=(10,5))
    # if args.precision == 32:
    #     cucosma_df = read_nvprof_data(base_path + "cucosma.csv")
    #     cutlass_df = read_nvprof_data(base_path + "cutlass.csv")
    #     combined_df_db = pd.concat([split_k_1, split_k_2, split_k_4, split_k_8, split_k_16, cutlass_df, cucosma_df, cublas_df], axis=1)
    #     combined_df_db.columns = ["-", "2", "4", "8", "16", "cutlass", "cucosma", "cublas"]
    #     sns.violinplot(data=combined_df_db).set(xticklabels=["-", "2", "4", "8", "16", "CUTLASS", "cuCOSMA", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 8192 with double buffering", xlabel="Split K")
    # else:
    #     combined_df_db = pd.concat([split_k_1, split_k_2, split_k_4, split_k_8, split_k_16, cublas_df], axis=1)
    #     combined_df_db.columns = ["-", "2", "4", "8", "16", "cublas"]
    #     sns.violinplot(data=combined_df_db).set(xticklabels=["-", "2", "4", "8", "16", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 8192 with double buffering", xlabel="Split K")

    # plt.axhline(peak_performance, linestyle='--', label="Peak Performance")
    # plt.legend()
    # fig.savefig(path + "comparison.png")

### (256 x 10240) x (10240 x 256)
def eval_256_10240_256(precision):
    path = str(args.path) + "256_256_10240_" + str(precision) + "bit/"

    best_avg_perf = 99999999999
    best_name = "empty"
    best = []
    for file in os.listdir(path):
        if not file.startswith('cu'):
            tmp = read_nvprof_data(path + str(file))
            avg_perf = tmp.mean()
            if avg_perf < best_avg_perf:
                best_avg_perf = avg_perf
                best_name = str(file)
                best = tmp

    helpers.print_info("Best average performance: " + best_avg_perf, False)
    helpers.print_info("From file: " + best_name, False)


    peak_performance = 256 * 256 * (2 * 10240 - 1) / (7 * 1000 * 1000 * 1000) # OPS/(FLOPS/ms) = ms
    if precision == 32:
        peak_performance = peak_performance * 2
        precision_str = "Single precision: "
    else:
        precision_str = "Double precision: "

    # split_k_1 = read_nvprof_data(path + "split_k_1.csv")
    # split_k_2 = read_nvprof_data(path + "split_k_2.csv")
    # split_k_4 = read_nvprof_data(path + "split_k_4.csv")
    # split_k_5 = read_nvprof_data(path + "split_k_5.csv")
    # split_k_8 = read_nvprof_data(path + "split_k_8.csv")
    # split_k_10 = read_nvprof_data(path + "split_k_10.csv")
    # split_k_16 = read_nvprof_data(path + "split_k_16.csv")
    # split_k_20 = read_nvprof_data(path + "split_k_20.csv")
    # split_k_40 = read_nvprof_data(path + "split_k_40.csv")

    # cublas_df = read_nvprof_data(base_path + "cublas.csv")
    # fig = plt.figure(figsize=(10,5))

    # if args.precision == 32:
    #     cucosma_df = read_nvprof_data(base_path + "cucosma.csv")
    #     cutlass_df = read_nvprof_data(base_path + "cutlass.csv")
    #     combined_df_db = pd.concat([split_k_1, split_k_2, split_k_4, split_k_5, split_k_8, split_k_10, split_k_16, split_k_20, split_k_40, cutlass_df, cucosma_df, cublas_df], axis=1)
    #     combined_df_db.columns = ["-", "2", "4", "5", "8", "10", "16", "20", "40", "cutlass", "cucosma", "cublas"]
    #     sns.violinplot(data=combined_df_db).set(xticklabels=["-", "2", "4", "5", "8", "10", "16", "20", "40", "CUTLASS", "cuCOSMA", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 1024, N = 1024, K = 8192 with double buffering", xlabel="Split K")
    # else:
    #     combined_df_db = pd.concat([split_k_1, split_k_2, split_k_4, split_k_5, split_k_8, split_k_10, split_k_16, split_k_20, split_k_40, cublas_df], axis=1)
    #     combined_df_db.columns = ["-", "2", "4", "5", "8", "10", "16", "20", "40", "cublas"]
    #     sns.violinplot(data=combined_df_db).set(xticklabels=["-", "2", "4", "5", "8", "10", "16", "20", "40", "cuBLAS"], ylabel="Runtime [ms]", title=precision_str + "M = 256, N = 256, K = 10240 with double buffering", xlabel="Split K")

    # plt.axhline(256 * 256 * (2 * 10240 - 1) / peak_performance, linestyle='--', label="Peak Performance")
    # plt.legend()
    # fig.savefig(path + "comparison.png")

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


if args.test == 1:
    eval_1024_1024(32)
    eval_1024_1024(64)
    helpers.print_success("Performance evaluation finished.", False)
elif args.test == 2:
    eval_4096_4096(32)
    eval_4096_4096(64)
    helpers.print_success("Performance evaluation finished.", False)
elif args.test == 3:
    eval_1024_8192_1024(32)
    eval_1024_8192_1024(64)
    helpers.print_success("Performance evaluation finished.", False)
elif args.test == 4:
    eval_256_10240_256(32)
    eval_256_10240_256(64)
    helpers.print_success("Performance evaluation finished.", False)
else:
    helpers.print_error("Invalid test number.", False)