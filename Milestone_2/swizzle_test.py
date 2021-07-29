import numpy as np

A = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    [16, 17, 18, 19],
    [20, 21, 22, 23],
    [24, 25, 26, 27],
    [28, 29, 30, 31]
])

def SWIZZLE_x(idx): # LaneIdy
#     # return (idx >> 1) & 7
    # return 2 * (idx >> 2) + (idx & 1)
    return 2 * (idx >> 2) + (idx & 1)
def SWIZZLE_y(idx): # LaneIdx
    # return ((idx & 16) >> 3) | (idx & 1)
    # return 16 * ((idx >> 1) & 1)
    return 16 * ((idx >> 1) & 1)

# print(SWIZZLE_x(0))
# print(SWIZZLE_x(1))
# print(SWIZZLE_x(2))
# print(SWIZZLE_x(3))
# print(SWIZZLE_x(4))
# print(SWIZZLE_x(5))
# print(SWIZZLE_x(6))
# print(SWIZZLE_x(7))

# print(SWIZZLE_y(0))
# print(SWIZZLE_y(1))
# print(SWIZZLE_y(2))
# print(SWIZZLE_y(3))
# print(SWIZZLE_y(4))
# print(SWIZZLE_y(5))
# print(SWIZZLE_y(6))
# print(SWIZZLE_y(7))

# for x in range (0, 8):
#     print("-" * 3 * 8 + "-")
#     for y in range (0, 4):
#         print("| " + str(SWIZZLE_x(x * 4 + y)) + " ", end="")
#     print("|")
# print("-" * 3 * 8 + "-")

# for x in range (0, 8):
#     print("-" * 3 * 8 + "-")
#     for y in range (0, 4):
#         print("| " + str(SWIZZLE_y(x * 4 + y)) + " ", end="")
#     print("|")
# print("-" * 3 * 8 + "-")

for x in range (0, 8):
    # print("-" * 3 * 8 + "-")
    for y in range (0, 4):
        idx = x * 4 + y
        print("old index = " + str(idx))
        print("SWIZZLE_x(" + str(x) + ") = " + str(SWIZZLE_x(idx)))
        print("SWIZZLE_y(" + str(y) + ") = " + str(SWIZZLE_y(idx)))
        print("new index = " + str(SWIZZLE_x(idx) + SWIZZLE_y(idx)))
        print("--")
        # print("| " + str(SWIZZLE_x(x * 4 + y) * 4 + SWIZZLE_y(x * 4 + y)) + " ", end="")
    # print("|")
print("-" * 3 * 8 + "-")

# for x in range (0, 8):
#     print("-" * 3 * 8 + "-")
#     for y in range (0, 4):
#         idx = x * 4 + y
#         # print("| " + str(idx) + " -> " + str(4 * SWIZZLE_x(idx) + SWIZZLE_y(idx)) + " ", end="")
#         print("| " + str(4 * SWIZZLE_x(idx) + SWIZZLE_y(idx)) + " <- " + str(idx) + " ", end="")
#     print("|")
# print("-" * 3 * 8 + "-")

# for x in range (0, 8):
#     print("-" * 3 * 8 + "-")
#     for y in range (0, 4):
#         idx = x * 4 + y
#         print("| " + str(A[SWIZZLE_x(idx)][SWIZZLE_y(idx)]) + " ", end="")
#     print("|")
# print("-" * 3 * 8 + "-")