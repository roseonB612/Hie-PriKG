import numpy as np
import os
import time

# 配置
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
data = np.load(os.path.join(ROOT_DIR, "fb15k_entity_embeddings.npy"))
DIM = data.shape[1]
NUM_BITS = 7    # 每张表的位数
L_TABLES = 5    # 表的数量（尝试用 5 张表来抢救）

print(f"--- 正在构建多表 LSH (表数: {L_TABLES}, 每表位数: {NUM_BITS}) ---")

# 1. 构建多组随机投影面
np.random.seed(42)
multi_planes = [np.random.randn(DIM, NUM_BITS).astype(np.float32) for _ in range(L_TABLES)]

# 2. 对原始数据进行多表哈希
# multi_labels 形状为 (L_TABLES, 14951)
multi_labels = []
powers_of_two = 1 << np.arange(NUM_BITS)[::-1]

for i in range(L_TABLES):
    projections = np.dot(data, multi_planes[i])
    hashes = (projections > 0).astype(int)
    bucket_ids = np.dot(hashes, powers_of_two)
    multi_labels.append(bucket_ids)

multi_labels = np.array(multi_labels)

# 3. 模拟测试
np.random.seed(123)
query_indices = np.random.choice(len(data), 100, replace=False)
queries = data[query_indices]

print("--- 正在进行多表检索测试 ---")
hits = 0
for i, query in enumerate(queries):
    real_index = query_indices[i]
    
    # 在 L 张表里分别去找
    for table_idx in range(L_TABLES):
        # 1. 计算 Query 在这张表里的桶 ID
        proj = np.dot(query, multi_planes[table_idx])
        query_bucket_id = np.dot((proj > 0).astype(int), powers_of_two)
        
        # 2. 检查这张表里，Query 的桶 ID 是否和真实数据的桶 ID 一致
        if query_bucket_id == multi_labels[table_idx, real_index]:
            hits += 1
            break # 只要有一张表命中了，就跳出，看下一个 Query

accuracy = (hits / len(queries)) * 100
print(f"\n[结果] 多表 LSH (L={L_TABLES}) 命中率: {accuracy:.2f}%")
print(f"对比之前单表命中率: 16.00%")