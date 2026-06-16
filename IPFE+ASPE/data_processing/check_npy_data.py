import numpy as np

# 加载文件
data = np.load(r'encrypted_centroids.npy')

# 查看基本信息
print("数据类型:", data.dtype)
print("数据形状:", data.shape)

# 查看具体内容（前 5 行）
print("前 5 行数据:\n", data[:5])

# 如果数据太大，只想看统计信息
print("最大值:", np.max(data))
print("最小值:", np.min(data))
print("平均值:", np.mean(data))