import numpy as np

# 生成一个 (10000, 176) 形状的随机 ndarray
random_array = np.random.rand(10000, 176)

# 生成一个长度为 44 的随机整数列表，范围在 0 到 175 之间
selected_strategies = np.random.choice(176, 44, replace=False).tolist()

res_m = []
for i in range(10000):
    res = [random_array[i][j] for j in selected_strategies]
    res_m.append(res)
result = np.array(res_m)

print("随机生成的 ndarray 形状:", random_array.shape)
print("随机生成的长度为 44 的列表:", selected_strategies)