import dace
import numpy as np
from dace.transformation.interstate import GPUTransformSDFG

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')

@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N]):
    return A @ B

sdfg = matmul.to_sdfg(strict=False)
sdfg.expand_library_nodes()
sdfg.apply_transformations(GPUTransformSDFG)
sdfg.save('matmul_vanilla.sdfg')
