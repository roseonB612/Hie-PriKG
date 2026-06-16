import numpy as np
import os

# 1. 加载数据
data_path = "data/fb15k_entity_embeddings.npy"
labels_path = "data/lsh_bucket_labels.npy"

data = np.load(data_path)
labels = np.load(labels_path)

print(f"--- 正在为 LSH 分桶计算质心 ---")
print(f"实体总数: {len(data)}, 维度: {data.shape[1]}")

# 2. 计算每个桶的质心
unique_buckets = np.unique(labels)
num_buckets = len(unique_buckets)
dim = data.shape[1]

# 创建一个字典来存储质心，或者直接用矩阵（如果桶 ID 是连续的）
# 推荐使用字典，因为 LSH 产生的桶 ID 往往是不连续的
lsh_centroids_dict = {}

for bucket_id in unique_buckets:
    # 提取属于当前桶的所有向量
    bucket_vectors = data[labels == bucket_id]
    # 计算均值作为质心
    centroid = np.mean(bucket_vectors, axis=0)
    lsh_centroids_dict[bucket_id] = centroid

# 为了方便后续 IPFE 计算，我们将质心转化为固定顺序的数组和对应的映射表
sorted_bucket_ids = sorted(unique_buckets)
lsh_centroids_matrix = np.array([lsh_centroids_dict[bid] for bid in sorted_bucket_ids])

# 3. 保存结果
save_dir = "e:/实验数据/"
np.save(os.path.join(save_dir, 'lsh_centroids.npy'), lsh_centroids_matrix)
np.save(os.path.join(save_dir, 'lsh_bucket_mapping.npy'), np.array(sorted_bucket_ids))

print(f"\n[成功] LSH 质心计算完成！")
print(f"有效桶数量: {num_buckets}")
print(f"质心矩阵形状: {lsh_centroids_matrix.shape}")
print(f"- 已保存 lsh_centroids.npy (用于 IPFE 第一层匹配)")
print(f"- 已保存 lsh_bucket_mapping.npy (用于映射质心索引到实际桶ID)")