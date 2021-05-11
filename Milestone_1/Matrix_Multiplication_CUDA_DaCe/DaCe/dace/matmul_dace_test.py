import dace
from dace.transformation.interstate import GPUTransformSDFG
import numpy as np
import sys

if len(sys.argv) != 5:
    sys.exit('Error: Please provide exactly four arguments.')

rowsA = int(sys.argv[1])
columnsA = int(sys.argv[2])
rowsB = int(sys.argv[3])
columnsB = int(sys.argv[4])

if columnsA != rowsB:
    sys.exit('Error: The inner dimensions of the matrices need to match.')

@dace.program
def matmul(A: dace.float64[rowsA, columnsA], B: dace.float64[rowsB, columnsB]):
    return A @ B

A = np.random.rand(rowsA, columnsA)
B = np.random.rand(rowsB, columnsB)
C = matmul(A, B)
if C.all() != (A @ B).all():
    sys.exit('Error: Result is incorrect.')

print(C)
sdfg = matmul.to_sdfg(strict=False)
# sdfg.apply_transformations(GPUTransformSDFG)
