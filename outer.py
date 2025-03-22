import numpy as np

# 外积的定义
# a.shape = (m,)
# b.shape = (n,)
# res.shape = (m, n)
# res[i, j] = a[i] * b[j]

# 定义两个一维数组
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 计算外积
outer_product = np.outer(a, b)

print("数组 a:", a)
print("数组 b:", b)
print("外积结果:")
print(outer_product)
