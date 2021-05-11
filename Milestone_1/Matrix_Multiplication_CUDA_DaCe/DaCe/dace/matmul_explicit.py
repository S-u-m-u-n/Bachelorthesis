import dace
import numpy as np
# from dace.transformation.interstate import GPUTransformSDFG

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')

@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    tmp = dace.define_local([M, N, K], dtype=A.dtype)

    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            a << A[i, k]
            b << B[k, j]
            out >> tmp[i, j, k]

            out = a * b

    dace.reduce(lambda a, b: a + b, tmp, C, axis=2)

A = np.random.rand(128, 128)
B = np.random.rand(128, 128)
C = np.random.rand(128, 128)

matmul(A=A, B=B, C=C)
# sdfg = matmul.to_sdfg(strict=False)
# sdfg.apply_transformations(GPUTransformSDFG)
