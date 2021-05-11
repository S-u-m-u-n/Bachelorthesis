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

A = np.random.rand(rowsA, columnsA)
B = np.random.rand(rowsB, columnsB)
C = np.dot(A, B)
# print(A)
# print(B)
# print(C)
if C.all() != np.dot(A, B).all():
    sys.exit('Error: Result is incorrect.')
