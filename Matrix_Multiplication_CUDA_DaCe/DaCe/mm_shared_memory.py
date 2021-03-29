import dace
import numpy as np
from dace.transformation.interstate import GPUTransformSDFG, StateFusion
from dace.transformation.dataflow import MapTiling, InLocalStorage, MapExpansion

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')

@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N]):
    return A @ B

sdfg = matmul.to_sdfg(strict=False)
sdfg.expand_library_nodes()
sdfg.apply_transformations(GPUTransformSDFG)
sdfg.apply_transformations(MapTiling, {'tile_sizes': (128, 128, 1)})
sdfg.apply_transformations(InLocalStorage, {})
sdfg.apply_transformations(StateFusion)
sdfg.save('matmul_shared_memory.sdfg')
