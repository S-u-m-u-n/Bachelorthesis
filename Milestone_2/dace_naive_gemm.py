import dace
import dace.libraries.blas
import numpy as np
from argparse import ArgumentParser

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')

parser = ArgumentParser()
parser.add_argument("-M", type=int, dest='M', nargs="?", default=640)
parser.add_argument("-K", type=int, dest='K', nargs="?", default=640)
parser.add_argument("-N", type=int, dest='N', nargs="?", default=640)
parser.add_argument("--alpha", type=np.float64, dest='alpha', nargs="?", default=1.0)
parser.add_argument("--beta", type=np.float64, dest='beta', nargs="?", default=1.0)
parser.add_argument("-r", "--repetitions", type=int, dest='repetitions', nargs="?", default=1)
parser.add_argument('-p', '--precision',
                    type=int,
                    dest='precision',
                    choices=[32, 64],
                    default=64,
                    help="Specify floating precision (32 or 64)")
args = parser.parse_args()

if args.precision == 32:
    dtype = dace.float32
    ndtype = np.float32
else:
    dtype = dace.float64
    ndtype = np.float64

M_input = args.M
N_input = args.N
K_input = args.K
alpha = ndtype(args.alpha)
beta = ndtype(args.beta)

@dace.program
def matmul(A: dtype[M, K], B: dtype[K, N], C: dtype[M, N], alpha: dtype, beta: dtype):
    return alpha * (A @ B) + beta * C

dace.libraries.blas.default_implementation = 'Default'
sdfg = matmul.to_sdfg()
sdfg.save('dace_naive_gemm.sdfg')
for i in range(args.repetitions):
    A = np.random.rand(M_input, K_input).astype(ndtype)
    B = np.random.rand(K_input, N_input).astype(ndtype)
    C = np.zeros((M_input, N_input)).astype(ndtype)
    matmul(A=A, B=B, C=C, alpha=alpha, beta=beta)