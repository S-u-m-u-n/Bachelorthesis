import dace
import numpy as np
import sys

if len(sys.argv) != 5:
    raise ValueError('Please provide exactly four arguments.')

@dace.program
def matmul(A, B):
    return A * B

rowsA = int(sys.argv[1])
columnsA = int(sys.argv[2])
rowsB = int(sys.argv[3])
columnsB = int(sys.argv[4])

if columnsA != rowsB:
    sys.exit("Error: The inner dimensions of the matrices don't match.")

A = np.random.rand(rowsA, columnsA)
B = np.random.rand(rowsB, columnsB)

matmul(A, B)
