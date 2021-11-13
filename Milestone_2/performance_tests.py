import subprocess
import helpers
import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-t", "--test", type=int, dest='test', nargs="?", required=True)
parser.add_argument("-r", "--repetitions", type=int, dest='repetitions', nargs="?", default=10)
parser.add_argument("-p", "--path", type=str, dest='path', nargs="?", required=True)
parser.add_argument('--precision', type=int, dest='precision', choices=[32, 64], required=True, help="Specify floating precision (32 or 64)")
args = parser.parse_args()

helpers.print_info("Running performance tests...", False)

nvprof_options = ["nvprof", "--print-gpu-trace", "--csv", "--log-file"]
# nvprof --print-gpu-trace --csv --log-file vectorized_64_db_st.csv
python_options = ["python", "./sdfg_api_v2.py", "--precision=" + str(args.precision), "--repetitions=" + str(args.repetitions)]
cublas_options = ["python", "./cublas.py", "--precision=" + str(args.precision), "--repetitions=" + str(args.repetitions)]

cutlass_options = ["/home/jacobsi/cutlass/build/tools/profiler/cutlass_profiler", "--verification-enabled=false", "--warmup-iterations=0", "--operation=gemm", "--profiling-iterations=" + str(args.repetitions - 1)]
# nvprof --print-gpu-trace --csv --log-file cutlass_test.csv /home/jacobsi/cutlass/build/tools/profiler/cutlass_profiler --verification-enabled=false --warmup-iterations=0 --operation=gemm --m=1024 --n=1024 --k=1024 --cta_m=128 --cta_n=64 --A=f64:column --B=f64:column --profiling-iterations=10
if args.precision == 64:
    cutlass_options += ["--cta_m=128", "--cta_n=64", "--A=f64:column", "--B=f64:column"]
else:
    cutlass_options += ["--cta_m=64", "--cta_n=64", "--A=f32:column", "--B=f32:column"]


### (1024 x 1024) x (1024 x 1024)
if args.test == 1:
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (1024 x 1024) x (1024 x 1024)", False)
    helpers.print_info("=" * 20, False)
    python_options += ["-M=1024", "-N=1024", "-K=1024"]
    cublas_options += ["-M=1024", "-N=1024", "-K=1024"]
    cutlass_options += ["--m=1024", "--n=1024", "--k=1024"]
    base_path = "./performance_test_results/1024_1024_1024_" + str(args.precision) + "bit/"
    path = base_path + str(args.path) +'/'
    subprocess.run(["mkdir", "-p", path])
    subprocess.run(nvprof_options + [path + "unoptimized.csv"] + python_options)
    subprocess.run(nvprof_options + [path + "swizzled_threads.csv"] + python_options + ["--swizzle-threads"])
    subprocess.run(nvprof_options + [path + "swizzled_thread_blocks.csv"] + python_options + ["--swizzle-thread-blocks", "2"])
    subprocess.run(nvprof_options + [path + "swizzled_threads_swizzled_thread_blocks.csv"] + python_options + ["--swizzle-threads", "--swizzle-thread-blocks", "2"])
    # subprocess.run(nvprof_options + [path + "vectorization.csv"] + python_options + ["--vectorization"])
    # subprocess.run(nvprof_options + [path + "vectorization_swizzled_threads.csv"] + python_options + ["--vectorization", "--swizzle-threads"])
    # subprocess.run(nvprof_options + [path + "vectorization_swizzled_thread_blocks.csv"] + python_options + ["--vectorization", "--swizzle-thread-blocks", "2"])
    # subprocess.run(nvprof_options + [path + "vectorization_swizzled_threads_swizzled_thread_blocks.csv"] + python_options + ["--vectorization", "--swizzle-threads", "--swizzle-thread-blocks", "2"])
    subprocess.run(nvprof_options + [path + "double_buffering.csv"] + python_options + ["--double-buffering"])
    subprocess.run(nvprof_options + [path + "double_buffering_swizzled_threads.csv"] + python_options + ["--double-buffering", "--swizzle-threads"])
    subprocess.run(nvprof_options + [path + "double_buffering_swizzled_thread_blocks.csv"] + python_options + ["--swizzle-thread-blocks", "2"])
    subprocess.run(nvprof_options + [path + "double_buffering_swizzled_threads_swizzled_thread_blocks.csv"] + python_options + ["--swizzle-threads", "--swizzle-thread-blocks", "2"])
    
    # subprocess.run(nvprof_options + [path + "double_buffering_vectorization.csv"] + python_options + ["--double-buffering", "--vectorization"])
    # subprocess.run(nvprof_options + [path + "double_buffering_vectorization_swizzled_threads.csv"] + python_options + ["--double-buffering", "--vectorization", "--swizzle-threads"])
    # subprocess.run(nvprof_options + [path + "double_buffering_vectorization_swizzled_thread_blocks.csv"] + python_options + ["--double-buffering", "--vectorization", "--swizzle-thread-blocks", "2"])
    # subprocess.run(nvprof_options + [path + "double_buffering_vectorization_swizzled_threads_swizzled_thread_blocks.csv"] + python_options + ["--double-buffering", "--vectorization", "--swizzle-threads", "--swizzle-thread-blocks", "2"])
    subprocess.run(nvprof_options + [base_path + "cublas.csv"] + cublas_options)
    subprocess.run(nvprof_options + [base_path + "cutlass.csv"] + cutlass_options)

### (4096 x 4096) x (4096 x 4096)
if args.test == 2:
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (4096 x 4096) x (4096 x 4096)", False)
    helpers.print_info("=" * 20, False)
    python_options += ["-M=4096", "-N=4096", "-K=4096"]
    cublas_options += ["-M=4096", "-N=4096", "-K=4096"]
    cutlass_options += ["--m=4096", "--n=4096", "--k=4096"]
    base_path = "./performance_test_results/4096_4096_4096_" + str(args.precision) + "bit/"
    path = base_path + str(args.path) +'/'
    subprocess.run(["mkdir", "-p", path])
    subprocess.run(nvprof_options + [path + "unoptimized.csv"] + python_options)
    subprocess.run(nvprof_options + [path + "swizzled_threads.csv"] + python_options + ["--swizzle-threads"])
    subprocess.run(nvprof_options + [path + "swizzled_thread_blocks.csv"] + python_options + ["--swizzle-thread-blocks", "2"])
    subprocess.run(nvprof_options + [path + "swizzled_threads_swizzled_thread_blocks.csv"] + python_options + ["--swizzle-threads", "--swizzle-thread-blocks", "2"])
    # subprocess.run(nvprof_options + [path + "vectorization.csv"] + python_options + ["--vectorization"])
    # subprocess.run(nvprof_options + [path + "vectorization_swizzled_threads.csv"] + python_options + ["--vectorization", "--swizzle-threads"])
    # subprocess.run(nvprof_options + [path + "vectorization_swizzled_thread_blocks.csv"] + python_options + ["--vectorization", "--swizzle-thread-blocks", "2"])
    # subprocess.run(nvprof_options + [path + "vectorization_swizzled_threads_swizzled_thread_blocks.csv"] + python_options + ["--vectorization", "--swizzle-threads", "--swizzle-thread-blocks", "2"])
    subprocess.run(nvprof_options + [path + "double_buffering.csv"] + python_options + ["--double-buffering"])
    subprocess.run(nvprof_options + [path + "double_buffering_swizzled_threads.csv"] + python_options + ["--double-buffering", "--swizzle-threads"])
    subprocess.run(nvprof_options + [path + "double_buffering_swizzled_thread_blocks.csv"] + python_options + ["--swizzle-thread-blocks", "2"])
    subprocess.run(nvprof_options + [path + "double_buffering_swizzled_threads_swizzled_thread_blocks.csv"] + python_options + ["--swizzle-threads", "--swizzle-thread-blocks", "2"])
    # subprocess.run(nvprof_options + [path + "double_buffering_vectorization.csv"] + python_options + ["--double-buffering", "--vectorization"])
    # subprocess.run(nvprof_options + [path + "double_buffering_vectorization_swizzled_threads.csv"] + python_options + ["--double-buffering", "--vectorization", "--swizzle-threads"])
    # subprocess.run(nvprof_options + [path + "double_buffering_vectorization_swizzled_thread_blocks.csv"] + python_options + ["--double-buffering", "--vectorization", "--swizzle-thread-blocks", "2"])
    # subprocess.run(nvprof_options + [path + "double_buffering_vectorization_swizzled_threads_swizzled_thread_blocks.csv"] + python_options + ["--double-buffering", "--vectorization", "--swizzle-threads", "--swizzle-thread-blocks", "2"])
    subprocess.run(nvprof_options + [base_path + "cublas.csv"] + cublas_options)
    subprocess.run(nvprof_options + [base_path + "cutlass.csv"] + cutlass_options)
   

## (1024 x 8192) x (8192 x 1024)
if args.test == 3:
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (1024 x 8192) x (8192 x 1024)", False)
    helpers.print_info("=" * 20, False)
    python_options += ["-M=1024", "-N=1024", "-K=8192", "--double-buffering", "--swizzle-threads"]
    # python_options += ["-M=1024", "-N=1024", "-K=8192", "--double-buffering", "--swizzle-threads", "--swizzle-thread-blocks", "2"]
    cublas_options += ["-M=1024", "-N=1024", "-K=8192"]
    base_path = "./performance_test_results/1024_1024_8192_" + str(args.precision) + "bit/"
    path = base_path + str(args.path) +'/'
    subprocess.run(["mkdir", "-p", path])
    subprocess.run(nvprof_options + [path + "split_k_1.csv"] + python_options)
    subprocess.run(nvprof_options + [path + "split_k_2_3d.csv"] + python_options + ["--split-k-3d", "--split-k", "2"])
    subprocess.run(nvprof_options + [path + "split_k_2_seq.csv"] + python_options + ["--split-k-seq", "--split-k", "2"])
    # subprocess.run(nvprof_options + [path + "split_k_3.csv"] + python_options + ["--split-k", "3"])
    subprocess.run(nvprof_options + [path + "split_k_4_3d.csv"] + python_options + ["--split-k-3d", "--split-k", "4"])
    subprocess.run(nvprof_options + [path + "split_k_4_seq.csv"] + python_options + ["--split-k-seq", "--split-k", "4"])
    # subprocess.run(nvprof_options + [path + "split_k_5.csv"] + python_options + ["--split-k", "5"])
    # subprocess.run(nvprof_options + [path + "split_k_6.csv"] + python_options + ["--split-k", "6"])
    # subprocess.run(nvprof_options + [path + "split_k_7.csv"] + python_options + ["--split-k", "7"])
    subprocess.run(nvprof_options + [path + "split_k_8_3d.csv"] + python_options + ["--split-k-3d", "--split-k", "8"])
    subprocess.run(nvprof_options + [path + "split_k_8_seq.csv"] + python_options + ["--split-k-seq", "--split-k", "8"])
    # subprocess.run(nvprof_options + [path + "split_k_9.csv"] + python_options + ["--split-k", "9"])
    # subprocess.run(nvprof_options + [path + "split_k_10.csv"] + python_options + ["--split-k", "10"])
    # subprocess.run(nvprof_options + [path + "split_k_11.csv"] + python_options + ["--split-k", "11"])
    # subprocess.run(nvprof_options + [path + "split_k_12.csv"] + python_options + ["--split-k", "12"])
    # subprocess.run(nvprof_options + [path + "split_k_13.csv"] + python_options + ["--split-k", "13"])
    # subprocess.run(nvprof_options + [path + "split_k_14.csv"] + python_options + ["--split-k", "14"])
    # subprocess.run(nvprof_options + [path + "split_k_15.csv"] + python_options + ["--split-k", "15"])
    subprocess.run(nvprof_options + [path + "split_k_16_3d.csv"] + python_options + ["--split-k-3d", "--split-k", "16"])
    subprocess.run(nvprof_options + [path + "split_k_16_seq.csv"] + python_options + ["--split-k-seq", "--split-k", "16"])
    subprocess.run(nvprof_options + [base_path + "cublas.csv"] + cublas_options)
    # subprocess.run(nvprof_options + [base_path + "cutlass.csv"] + cutlass_options)
    # subprocess.run(["sed", "-i", "'1d;2d;3d'", base_path + "cublas.csv"]) # Delete first three lines, which are filled with nvprof information
    # subprocess.run(["sed", "-i", "'1d;2d;3d'", base_path + "cutlass.csv"]) # Delete first three lines, which are filled with nvprof information
    # subprocess.run(["sed", "-i", "'1d;2d;3d'", path + "*.csv"]) # Delete first three lines, which are filled with nvprof information

## (256 x 10240) x (10240 x 256)
if args.test == 4:
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (256 x 10240) x (10240 x 256)", False)
    helpers.print_info("=" * 20, False)
    python_options += ["-M=256", "-N=256", "-K=10240", "--double-buffering", "--swizzle-threads", "--swizzle-thread-blocks", "2"]
    cublas_options += ["-M=256", "-N=256", "-K=10240"]
    # cutlass_options += ["--m=256", "--n=256", "--k=10240"]
    base_path = "./performance_test_results/256_256_10240_" + str(args.precision) + "bit/"
    path = base_path + str(args.path) +'/'
    subprocess.run(["mkdir", "-p", path])
    subprocess.run(nvprof_options + [path + "split_k_1.csv"] + python_options)
    subprocess.run(nvprof_options + [path + "split_k_2.csv"] + python_options + ["--split-k", "2"])
    # subprocess.run(nvprof_options + [path + "split_k_3.csv"] + python_options + ["--split-k", "3"])
    subprocess.run(nvprof_options + [path + "split_k_4.csv"] + python_options + ["--split-k", "4"])
    subprocess.run(nvprof_options + [path + "split_k_5.csv"] + python_options + ["--split-k", "5"])
    # subprocess.run(nvprof_options + [path + "split_k_6.csv"] + python_options + ["--split-k", "6"])
    # subprocess.run(nvprof_options + [path + "split_k_7.csv"] + python_options + ["--split-k", "7"])
    subprocess.run(nvprof_options + [path + "split_k_8.csv"] + python_options + ["--split-k", "8"])
    # subprocess.run(nvprof_options + [path + "split_k_9.csv"] + python_options + ["--split-k", "9"])
    subprocess.run(nvprof_options + [path + "split_k_10.csv"] + python_options + ["--split-k", "10"])
    # subprocess.run(nvprof_options + [path + "split_k_11.csv"] + python_options + ["--split-k", "11"])
    # subprocess.run(nvprof_options + [path + "split_k_12.csv"] + python_options + ["--split-k", "12"])
    # subprocess.run(nvprof_options + [path + "split_k_13.csv"] + python_options + ["--split-k", "13"])
    # subprocess.run(nvprof_options + [path + "split_k_14.csv"] + python_options + ["--split-k", "14"])
    # subprocess.run(nvprof_options + [path + "split_k_15.csv"] + python_options + ["--split-k", "15"])
    subprocess.run(nvprof_options + [path + "split_k_16.csv"] + python_options + ["--split-k", "16"])
    # subprocess.run(nvprof_options + [path + "split_k_17.csv"] + python_options + ["--split-k", "17"])
    # subprocess.run(nvprof_options + [path + "split_k_18.csv"] + python_options + ["--split-k", "18"])
    # subprocess.run(nvprof_options + [path + "split_k_19.csv"] + python_options + ["--split-k", "19"])
    subprocess.run(nvprof_options + [path + "split_k_20.csv"] + python_options + ["--split-k", "20"])
    # subprocess.run(nvprof_options + [path + "split_k_21.csv"] + python_options + ["--split-k", "21"])
    # subprocess.run(nvprof_options + [path + "split_k_22.csv"] + python_options + ["--split-k", "22"])
    # subprocess.run(nvprof_options + [path + "split_k_23.csv"] + python_options + ["--split-k", "23"])
    # subprocess.run(nvprof_options + [path + "split_k_24.csv"] + python_options + ["--split-k", "24"])
    # subprocess.run(nvprof_options + [path + "split_k_25.csv"] + python_options + ["--split-k", "25"])
    # subprocess.run(nvprof_options + [path + "split_k_26.csv"] + python_options + ["--split-k", "26"])
    # subprocess.run(nvprof_options + [path + "split_k_27.csv"] + python_options + ["--split-k", "27"])
    # subprocess.run(nvprof_options + [path + "split_k_28.csv"] + python_options + ["--split-k", "28"])
    # subprocess.run(nvprof_options + [path + "split_k_29.csv"] + python_options + ["--split-k", "29"])
    # subprocess.run(nvprof_options + [path + "split_k_30.csv"] + python_options + ["--split-k", "30"])
    # subprocess.run(nvprof_options + [path + "split_k_31.csv"] + python_options + ["--split-k", "31"])
    # subprocess.run(nvprof_options + [path + "split_k_32.csv"] + python_options + ["--split-k", "32"])
    # subprocess.run(nvprof_options + [path + "split_k_33.csv"] + python_options + ["--split-k", "33"])
    # subprocess.run(nvprof_options + [path + "split_k_34.csv"] + python_options + ["--split-k", "34"])
    # subprocess.run(nvprof_options + [path + "split_k_35.csv"] + python_options + ["--split-k", "35"])
    # subprocess.run(nvprof_options + [path + "split_k_36.csv"] + python_options + ["--split-k", "36"])
    # subprocess.run(nvprof_options + [path + "split_k_37.csv"] + python_options + ["--split-k", "37"])
    # subprocess.run(nvprof_options + [path + "split_k_38.csv"] + python_options + ["--split-k", "38"])
    # subprocess.run(nvprof_options + [path + "split_k_39.csv"] + python_options + ["--split-k", "39"])
    subprocess.run(nvprof_options + [path + "split_k_40.csv"] + python_options + ["--split-k", "40"])
    # subprocess.run(nvprof_options + [path + "split_k_41.csv"] + python_options + ["--split-k", "41"])
    # subprocess.run(nvprof_options + [path + "split_k_42.csv"] + python_options + ["--split-k", "42"])
    # subprocess.run(nvprof_options + [path + "split_k_43.csv"] + python_options + ["--split-k", "43"])
    # subprocess.run(nvprof_options + [path + "split_k_44.csv"] + python_options + ["--split-k", "44"])
    # subprocess.run(nvprof_options + [path + "split_k_45.csv"] + python_options + ["--split-k", "45"])
    # subprocess.run(nvprof_options + [path + "split_k_46.csv"] + python_options + ["--split-k", "46"])
    # subprocess.run(nvprof_options + [path + "split_k_47.csv"] + python_options + ["--split-k", "47"])
    # subprocess.run(nvprof_options + [path + "split_k_48.csv"] + python_options + ["--split-k", "48"])
    subprocess.run(nvprof_options + [base_path + "cublas.csv"] + cublas_options)
    # subprocess.run(nvprof_options + [base_path + "cutlass.csv"] + cutlass_options)
    # subprocess.run(["sed", "-i", "'1d;2d;3d'", base_path + "cublas.csv"]) # Delete first three lines, which are filled with nvprof information
    # subprocess.run(["sed", "-i", "'1d;2d;3d'", base_path + "cutlass.csv"]) # Delete first three lines, which are filled with nvprof information
    # files = glob.glob(path + "*.csv")
    # subprocess.run(["sed", "-i", "1d;2d;3d"] + files) # Delete first three lines, which are filled with nvprof information

### (1024 x 1024) x (1024 x 1024)
if args.test == 5:
    helpers.print_info("=" * 20, False)
    helpers.print_info("===== (1024 x 1024) x (1024 x 1024)", False)
    helpers.print_info("=" * 20, False)
    python_options += ["-M=1024", "-N=1024", "-K=1024"]
    cublas_options += ["-M=1024", "-N=1024", "-K=1024"]
    cutlass_options += ["--m=1024", "--n=1024", "--k=1024"]
    base_path = "./performance_test_results/1024_1024_1024_isolated_optimizations_" + str(args.precision) + "bit/"
    path = base_path + str(args.path) +'/'
    subprocess.run(["mkdir", "-p", path])
    subprocess.run(nvprof_options + [path + "unoptimized.csv"] + python_options)
    subprocess.run(nvprof_options + [path + "swizzled_threads.csv"] + python_options + ["--swizzle-threads"])
    subprocess.run(nvprof_options + [path + "swizzled_thread_blocks.csv"] + python_options + ["--swizzle-thread-blocks", "2"])
    subprocess.run(nvprof_options + [path + "double_buffering.csv"] + python_options + ["--double-buffering"])

helpers.print_success("Performance tests finished.", False)
