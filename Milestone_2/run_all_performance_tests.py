import subprocess
import helpers
import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, dest='repetitions', nargs="?", default=200)
parser.add_argument("-p", "--path", type=str, dest='path', nargs="?", default="/home/jacobsi/Bachelorthesis/Milestone_2/performance_test_results/final/")
parser.add_argument("-t", "--test", type=int, dest='test', choices=[1, 2, 3, 4, 12, 34], required=True)
args = parser.parse_args()

# helpers.print_info("Running performance tests...", False)

nvprof_options = ["nvprof", "--print-gpu-trace", "--csv", "--log-file"]
global_python_options = ["python3", "./sdfg_api_v4.py", "--repetitions=" + str(args.repetitions)]

optimizations = {
        # Single optimization (5):
        "_st": ["--swizzle-threads"],
        "_stb": ["--swizzle-thread-blocks", "2"],
        "_rev": ["--reverse-k"],
        "_col": ["--shared-A-column"],
        "_dbr": ["--double-buffering-register"],


        # Two optimizations (10):
        "_st_stb": ["--swizzle-threads", "--swizzle-thread-blocks", "2"],
        "_st_rev": ["--swizzle-threads", "--reverse-k"],
        "_st_col": ["--swizzle-threads", "--shared-A-column"],
        "_st_dbr": ["--swizzle-threads", "--double-buffering-register"],

        "_stb_rev": ["--swizzle-thread-blocks", "2", "--reverse-k"],
        "_stb_col": ["--swizzle-thread-blocks", "2", "--shared-A-column"],
        "_stb_dbr": ["--swizzle-thread-blocks", "2", "--double-buffering-register"],

        "_rev_col": ["--reverse-k", "--shared-A-column"],
        "_rev_dbr": ["--reverse-k", "--double-buffering-register"],

        "_col_dbr": ["--shared-A-column", "--double-buffering-register"],


        # Three optimizations (10):
        "_st_stb_rev": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k"],
        "_st_stb_col": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--shared-A-column"],
        "_st_stb_dbr": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--double-buffering-register"],

        "_st_rev_col": ["--swizzle-threads", "--reverse-k", "--shared-A-column"],
        "_st_rev_dbr": ["--swizzle-threads", "--reverse-k", "--double-buffering-register"],

        "_st_col_dbr": ["--swizzle-threads", "--shared-A-column", "--double-buffering-register"],

        "_stb_rev_col": ["--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column"],
        "_stb_rev_dbr": ["--swizzle-thread-blocks", "2", "--reverse-k", "--double-buffering-register"],

        "_stb_col_dbr": ["--swizzle-thread-blocks", "2", "--shared-A-column", "--double-buffering-register"],

        "_rev_col_dbr": ["--reverse-k", "--shared-A-column", "--double-buffering-register"],


        # Four optimizations (5):
        "_st_stb_rev_col": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column"],
        "_st_stb_rev_dbr": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k", "--double-buffering-register"],

        "_st_stb_col_dbr": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--shared-A-column", "--double-buffering-register"],
        "_st_rev_col_dbr": ["--swizzle-threads", "--reverse-k", "--shared-A-column", "--double-buffering-register"],

        "_stb_rev_col_dbr": ["--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column", "--double-buffering-register"],


        # Five optimizations (1):
        "_st_stb_rev_col_dbr": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column", "--double-buffering-register"],
}

optimizations_dbs = {
        # Single optimization (6):
        "_st": ["--swizzle-threads"],
        "_stb": ["--swizzle-thread-blocks", "2"],
        "_rev": ["--reverse-k"],
        "_col": ["--shared-A-column"],
        "_dbr": ["--double-buffering-register"],
        "_npo": ["--nested-pointer-offset"],

        # Two optimizations (15):
        "_st_stb": ["--swizzle-threads", "--swizzle-thread-blocks", "2"],
        "_st_rev": ["--swizzle-threads", "--reverse-k"],
        "_st_col": ["--swizzle-threads", "--shared-A-column"],
        "_st_dbr": ["--swizzle-threads", "--double-buffering-register"],
        "_st_npo": ["--swizzle-threads", "--nested-pointer-offset"],

        "_stb_rev": ["--swizzle-thread-blocks", "2", "--reverse-k"],
        "_stb_col": ["--swizzle-thread-blocks", "2", "--shared-A-column"],
        "_stb_dbr": ["--swizzle-thread-blocks", "2", "--double-buffering-register"],
        "_stb_npo":  ["--swizzle-thread-blocks", "2", "--nested-pointer-offset"],

        "_rev_col": ["--reverse-k", "--shared-A-column"],
        "_rev_dbr": ["--reverse-k", "--double-buffering-register"],
        "_rev_npo": ["--reverse-k", "--nested-pointer-offset"],

        "_col_dbr": ["--shared-A-column", "--double-buffering-register"],
        "_col_npo": ["--shared-A-column", "--nested-pointer-offset"],
        
        "_dbr_npo": ["--double-buffering-register", "--nested-pointer-offset"],


        # Three optimizations (20):
        "_st_stb_rev": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k"],
        "_st_stb_col": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--shared-A-column"],
        "_st_stb_dbr": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--double-buffering-register"],
        "_st_stb_npo": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--nested-pointer-offset"],

        "_st_rev_col": ["--swizzle-threads", "--reverse-k", "--shared-A-column"],
        "_st_rev_dbr": ["--swizzle-threads", "--reverse-k", "--double-buffering-register"],
        "_st_rev_npo": ["--swizzle-threads", "--reverse-k", "--nested-pointer-offset"],

        "_st_col_dbr": ["--swizzle-threads", "--shared-A-column", "--double-buffering-register"],
        "_st_col_npo": ["--swizzle-threads", "--shared-A-column", "--nested-pointer-offset"],

        "_st_dbr_npo": ["--swizzle-threads", "--double-buffering-register", "--nested-pointer-offset"],

        "_stb_rev_col": ["--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column"],
        "_stb_rev_dbr": ["--swizzle-thread-blocks", "2", "--reverse-k", "--double-buffering-register"],
        "_stb_rev_npo": ["--swizzle-thread-blocks", "2", "--reverse-k", "--nested-pointer-offset"],

        "_stb_col_dbr": ["--swizzle-thread-blocks", "2", "--shared-A-column", "--double-buffering-register"],
        "_stb_col_npo": ["--swizzle-thread-blocks", "2", "--shared-A-column", "--nested-pointer-offset"],

        "_stb_dbr_npo": ["--swizzle-thread-blocks", "2", "--double-buffering-register", "--nested-pointer-offset"],

        "_rev_col_dbr": ["--reverse-k", "--shared-A-column", "--double-buffering-register"],
        "_rev_col_npo": ["--reverse-k", "--shared-A-column", "--nested-pointer-offset"],

        "_rev_dbr_npo": ["--reverse-k", "--double-buffering-register", "--nested-pointer-offset"],

        "_col_dbr_npo": ["--shared-A-column", "--double-buffering-register", "--nested-pointer-offset"],


        # Four optimizations (15):
        "_st_stb_rev_col": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column"],
        "_st_stb_rev_dbr": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k", "--double-buffering-register"],
        "_st_stb_rev_npo": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k", "--nested-pointer-offset"],

        "_st_stb_col_dbr": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--shared-A-column", "--double-buffering-register"],
        "_st_stb_col_npo": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--shared-A-column", "--nested-pointer-offset"],

        "_st_stb_dbr_npo": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--double-buffering-register", "--nested-pointer-offset"],

        "_st_rev_col_dbr": ["--swizzle-threads", "--reverse-k", "--shared-A-column", "--double-buffering-register"],
        "_st_rev_col_npo": ["--swizzle-threads", "--reverse-k", "--shared-A-column", "--nested-pointer-offset"],

        "_st_rev_dbr_npo": ["--swizzle-threads", "--reverse-k", "--double-buffering-register", "--nested-pointer-offset"],

        "_st_col_dbr_npo": ["--swizzle-threads", "--shared-A-column", "--double-buffering-register", "--nested-pointer-offset"],

        "_stb_rev_col_dbr": ["--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column", "--double-buffering-register"],
        "_stb_rev_col_npo": ["--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column", "--nested-pointer-offset"],

        "_stb_col_dbr_npo": ["--swizzle-thread-blocks", "2", "--shared-A-column", "--double-buffering-register", "--nested-pointer-offset"],

        "_stb_rev_dbr_npo": ["--swizzle-thread-blocks", "2", "--reverse-k", "--double-buffering-register", "--nested-pointer-offset"],
        
        "_rev_col_dbr_npo": ["--reverse-k", "--shared-A-column", "--double-buffering-register", "--nested-pointer-offset"],

        # Five optimizations (6):
        "_st_stb_rev_col_dbr": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column", "--double-buffering-register"],
        "_st_stb_rev_col_npo": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column", "--nested-pointer-offset"],
        "_st_stb_rev_dbr_npo": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k", "--double-buffering-register", "--nested-pointer-offset"],
        "_st_stb_col_dbr_npo": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--shared-A-column", "--double-buffering-register", "--nested-pointer-offset"],
        "_st_rev_col_dbr_npo": ["--swizzle-threads", "--reverse-k", "--shared-A-column", "--double-buffering-register", "--nested-pointer-offset"],
        "_stb_rev_col_dbr_npo": ["--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column", "--double-buffering-register", "--nested-pointer-offset"],
        
        # Six optimizations (1):
        "_st_stb_rev_col_dbr_npo": ["--swizzle-threads", "--swizzle-thread-blocks", "2", "--reverse-k", "--shared-A-column", "--double-buffering-register", "--nested-pointer-offset"],
}


### (1024 x 1024) x (1024 x 1024)
def run_1024_1024(precision):
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (1024 x 1024) x (1024 x 1024)" + str(precision) + "bit =====", False)
    helpers.print_info("=" * 20, False)
    python_options = global_python_options + ["-M=1024", "-N=1024", "-K=1024", "--precision=" + str(precision)]
    path = str(args.path) + "1024_1024_1024_" + str(precision) + "bit/"
    subprocess.run(["mkdir", "-p", path])

    subprocess.run(nvprof_options + [path + "unoptimized.csv"] + python_options)
    subprocess.run(nvprof_options + [path + "dbs.csv"] + python_options + ["--double-buffering-shared"])

    for result_path, optims in optimizations.items():
        helpers.print_info("Running: " + str(result_path), False)
        subprocess.run(nvprof_options + [path + result_path + ".csv"] + python_options + optims)

    for k, v in optimizations_dbs.items():
        helpers.print_info("Running: dbs" + str(result_path), False)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + ".csv"] + python_options + ["--double-buffering-shared"] + optims)

### (4096 x 4096) x (4096 x 4096)
def run_4096_4096(precision):
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (4096 x 4096) x (4096 x 4096)" + str(precision) + "bit =====", False)
    helpers.print_info("=" * 20, False)
    python_options = global_python_options + ["-M=4096", "-N=4096", "-K=4096", "--precision=" + str(precision)]
    path = str(args.path) + "4096_4096_4096_" + str(precision) + "bit/"
    subprocess.run(["mkdir", "-p", path])

    subprocess.run(nvprof_options + [path + "unoptimized.csv"] + python_options)
    subprocess.run(nvprof_options + [path + "dbs.csv"] + python_options + ["--double-buffering-shared"])

    for result_path, optims in optimizations.items():
        helpers.print_info("Running: " + str(result_path), False)
        subprocess.run(nvprof_options + [path + result_path + ".csv"] + python_options + optims)

    for result_path, optims in optimizations_dbs.items():
        helpers.print_info("Running: dbs" + str(result_path), False)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + ".csv"] + python_options + ["--double-buffering-shared"] + optims)
   
## (1024 x 8192) x (8192 x 1024)
def run_1024_8192_1024(precision):
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (1024 x 8192) x (8192 x 1024)" + str(precision) + "bit =====", False)
    helpers.print_info("=" * 20, False)
    python_options = global_python_options + ["-M=1024", "-N=1024", "-K=8192", "--double-buffering-shared", "--precision=" + str(precision)]
    path = str(args.path) + "1024_1024_8192_" + str(precision) + "bit/"
    subprocess.run(["mkdir", "-p", path])

    subprocess.run(nvprof_options + [path + "dbs_split_k_1.csv"] + python_options)
    subprocess.run(nvprof_options + [path + "dbs_split_k_2.csv"] + python_options + ["--split-k", "2"])
    subprocess.run(nvprof_options + [path + "dbs_split_k_4.csv"] + python_options + ["--split-k", "4"])
    subprocess.run(nvprof_options + [path + "dbs_split_k_8.csv"] + python_options + ["--split-k", "8"])
    subprocess.run(nvprof_options + [path + "dbs_split_k_16.csv"] + python_options + ["--split-k", "16"])

    for result_path, optims in optimizations_dbs.items():
        helpers.print_info("Running: dbs" + str(result_path), False)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_1.csv"] + python_options + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_2.csv"] + python_options + ["--split-k", "2"] + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_4.csv"] + python_options + ["--split-k", "4"] + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_8.csv"] + python_options + ["--split-k", "8"] + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_16.csv"] + python_options + ["--split-k", "16"] + optims)

## (256 x 10240) x (10240 x 256)
def run_256_10240_256(precision):
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (256 x 10240) x (10240 x 256)" + str(precision) + "bit =====", False)
    helpers.print_info("=" * 20, False)
    python_options = global_python_options + ["-M=256", "-N=256", "-K=10240", "--double-buffering-shared", "--precision=" + str(precision)]
    path = str(args.path) + "256_256_10240_" + str(precision) + "bit/"
    subprocess.run(["mkdir", "-p", path])

    subprocess.run(nvprof_options + [path + "dbs_split_k_1.csv"] + python_options)
    subprocess.run(nvprof_options + [path + "dbs_split_k_2.csv"] + python_options + ["--split-k", "2"])
    subprocess.run(nvprof_options + [path + "dbs_split_k_4.csv"] + python_options + ["--split-k", "4"])
    subprocess.run(nvprof_options + [path + "dbs_split_k_5.csv"] + python_options + ["--split-k", "5"])
    subprocess.run(nvprof_options + [path + "dbs_split_k_8.csv"] + python_options + ["--split-k", "8"])
    subprocess.run(nvprof_options + [path + "dbs_split_k_10.csv"] + python_options + ["--split-k", "10"])
    subprocess.run(nvprof_options + [path + "dbs_split_k_16.csv"] + python_options + ["--split-k", "16"])
    subprocess.run(nvprof_options + [path + "dbs_split_k_20.csv"] + python_options + ["--split-k", "20"])
    subprocess.run(nvprof_options + [path + "dbs_split_k_40.csv"] + python_options + ["--split-k", "40"])

    for result_path, optims in optimizations_dbs.items():
        helpers.print_info("Running: dbs" + str(result_path), False)

        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_1.csv"] + python_options + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_2.csv"] + python_options + ["--split-k", "2"] + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_4.csv"] + python_options + ["--split-k", "4"] + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_5.csv"] + python_options + ["--split-k", "5"] + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_8.csv"] + python_options + ["--split-k", "8"] + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_10.csv"] + python_options + ["--split-k", "10"] + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_16.csv"] + python_options + ["--split-k", "16"] + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_20.csv"] + python_options + ["--split-k", "20"] + optims)
        subprocess.run(nvprof_options + [path + "dbs" + result_path + "_split_k_40.csv"] + python_options + ["--split-k", "40"] + optims)

if args.test == 1:
    run_1024_1024(32)
    run_1024_1024(64)
    helpers.print_success("Performance tests finished.", False)
elif args.test == 2:
    run_4096_4096(32)
    run_4096_4096(64)
    helpers.print_success("Performance tests finished.", False)
elif args.test == 3:
    run_1024_8192_1024(32)
    run_1024_8192_1024(64)
    helpers.print_success("Performance tests finished.", False)
elif args.test == 4:
    run_256_10240_256(32)
    run_256_10240_256(64)
    helpers.print_success("Performance tests finished.", False)
elif args.test == 12:
    run_1024_1024(32)
    run_1024_1024(64)
    run_4096_4096(32)
    run_4096_4096(64)
    helpers.print_success("Performance tests finished.", False)
elif args.test == 34:
    run_1024_8192_1024(32)
    run_1024_8192_1024(64)
    run_256_10240_256(32)
    run_256_10240_256(64)
    helpers.print_success("Performance tests finished.", False)
else:
    helpers.print_error("Invalid test number.", False)




