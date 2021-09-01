import dace
import numpy as np
from argparse import ArgumentParser

@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N], alpha: dace.float64, beta: dace.float64):
    C = alpha * (A @ B) + beta * C
    return C

parser = ArgumentParser()
parser.add_argument("-M", type=int, dest='M', nargs="?", default=640)
parser.add_argument("-K", type=int, dest='K', nargs="?", default=640)
parser.add_argument("-N", type=int, dest='N', nargs="?", default=640)
parser.add_argument("--alpha", type=np.float64, dest='alpha', nargs="?", default=1.0)
parser.add_argument("--beta", type=np.float64, dest='beta', nargs="?", default=1.0)
parser.add_argument("-r", "--repetitions", type=int, dest='repetitions', nargs="?", default=1)

M=np.int32(args.M)
N=np.int32(args.N)
K=np.int32(args.K)
alpha = np.float64(args.alpha)
beta = np.float64(args.beta)

dace.libraries.blas.default_implementation = 'cuBLAS'
for i in range(args.repetitions):
    A = np.random.rand(M, K).astype(np_dtype)
    B = np.random.rand(K, N).astype(np_dtype)
    C = np.zeros((M, N)).astype(np_dtype)
    matmul(A=A, B=B, C=C, alpha=alpha, beta=beta)