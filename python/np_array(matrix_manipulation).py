import numpy as np


# 1. 矩阵初始化
# 1.1 零矩阵
A = np.zeros([3, 3], dtype=float)
print(A)
"""
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]"""

# 1.2 单位矩阵
A = np.identity(3)
print(A)
"""
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]"""

# 1.3 对角矩阵
A = np.eye(3, k=1)
print(f"A:\n {A}")
"""
 [[0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 0.]]
"""

A = np.eye(3)

# 2. 矩阵加减
B = np.zeros([3, 3], dtype=float)
num = 1
rows, cols = B.shape
for i in range(rows):
    for j in range(cols):
        B[i, j] = num
        num += 1
print(B)
"""
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]"""

C = A + B
print(C)
"""
[[ 2.  2.  3.]
 [ 4.  6.  6.]
 [ 7.  8. 10.]]"""

# 3. 矩阵乘法
# 3.1 常数 * 矩阵
D = 2 * B
print(D)
"""
[[ 2.  4.  6.]
 [ 8. 10. 12.]
 [14. 16. 18.]]"""

# 3.2 矩阵逐元素相乘 B*C = C*B
D = B * C
print(D)
"""
[[ 2.  4.  9.]
 [16. 30. 36.]
 [49. 64. 90.]]"""

# 3.3 矩阵相乘
D = np.matmul(B, C)
print(D)
"""
[[ 31.  38.  45.]
 [ 70.  86. 102.]
 [109. 134. 159.]]
"""


# 4. 矩阵求逆
# 4.1 逆
E = np.linalg.inv(C)
np.set_printoptions(
    precision=2, threshold=None, linewidth=None, suppress=True, formatter=None
)
print(E)
"""
[[-6.  -2.   3. ]
 [-1.   0.5  0. ]
 [ 5.   1.  -2. ]]"""

# 4.2 伪逆
E = np.linalg.inv(B)
print(E)
""" 逆矩阵不存在
[[ 3.15e+15 -6.31e+15  3.15e+15]
 [-6.31e+15  1.26e+16 -6.31e+15]
 [ 3.15e+15 -6.31e+15  3.15e+15]]"""
E = np.linalg.pinv(B)
print(E)
"""
[[-0.64 -0.17  0.31]
 [-0.06 -0.    0.06]
 [ 0.53  0.17 -0.19]]"""

# 5. 矩阵concatenate
# 5.1 纵向拼接
F = np.concatenate((B, C), axis=0)
print(F)
"""
[[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8.  9.]
 [ 2.  2.  3.]
 [ 4.  6.  6.]
 [ 7.  8. 10.]]"""

# 5.2 横向拼接
F = np.concatenate((B, C), axis=1)
print(F)
"""
[[ 1.  2.  3.  2.  2.  3.]
 [ 4.  5.  6.  4.  6.  6.]
 [ 7.  8.  9.  7.  8. 10.]]"""

# 6. 矩阵取值
G = F[0:2, 0:3]
print(G)
"""
[1. 2. 3.]
 [4. 5. 6.]]"""

# 7. 矩阵除法

H = F / 2
print(H)
"""
[[0.5 1.  1.5 1.  1.  1.5]
 [2.  2.5 3.  2.  3.  3. ]
 [3.5 4.  4.5 3.5 4.  5. ]]"""

# 8. 矩阵转置
H = H.transpose()
print(H)
