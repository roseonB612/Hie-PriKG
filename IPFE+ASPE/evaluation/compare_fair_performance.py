import numpy as np
import os
import time

# ================= 1. 环境与数据加载 =================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
data = np.load(os.path.join(ROOT_DIR, "fb15k_entity_embeddings.npy"))
km_centroids = np.load(os.path.join(ROOT_DIR, "centroids.npy"))
km_labels = np.load(os.path.join(ROOT_DIR, "bucket_labels.npy"))

DIM = data.shape[1]
NUM_QUERIES = 100
np.random.seed(123)
query_indices = np.random.choice(len(data), NUM_QUERIES, replace=False)
queries = data[query_indices]

# ================= 2. 构建多表 LSH (L=5) =================
L_TABLES = 5
NUM_BITS = 7
np.random.seed(42)
multi_planes = [np.random.randn(DIM, NUM_BITS).astype(np.float32) for _ in range(L_TABLES)]
powers_of_two = 1 << np.arange(NUM_BITS)[::-1]

# 预计算所有数据的多表标签
multi_lsh_labels = []
for i in range(L_TABLES):
    proj = np.dot(data, multi_planes[i])
    bucket_ids = np.dot((proj > 0).astype(int), powers_of_two)
    multi_lsh_labels.append(bucket_ids)
multi_lsh_labels = np.array(multi_lsh_labels)

# ================= 3. 公平对比测试 =================

def test_kmeans_topk(k=5):
    hits = 0
    start = time.time()
    for i, query in enumerate(queries):
        # 计算与 100 个质心的内积，选前 k 个
        sims = np.dot(km_centroids, query)
        top_k_buckets = np.argsort(sims)[-k:]
        real_label = km_labels[query_indices[i]]
        if real_label in top_k_buckets:
            hits += 1
    return (hits / NUM_QUERIES) * 100, (time.time() - start) / NUM_QUERIES * 1000

def test_lsh_multitable(l=5):
    hits = 0
    start = time.time()
    for i, query in enumerate(queries):
        real_idx = query_indices[i]
        # 在 L 张表里查，只要有一张表命中 Query 所在的桶
        for t in range(l):
            proj = np.dot(query, multi_planes[t])
            q_bucket = np.dot((proj > 0).astype(int), powers_of_two)
            if q_bucket == multi_lsh_labels[t, real_idx]:
                hits += 1
                break
    return (hits / NUM_QUERIES) * 100, (time.time() - start) / NUM_QUERIES * 1000

# 执行对比
km_acc, km_time = test_kmeans_topk(k=5)
lsh_acc, lsh_time = test_lsh_multitable(l=5)

# ================= 4. 输出公平报告 =================
print("\n" + "="*50)
print(f"{'算法策略':<20} | {'命中率 (%)':<12} | {'检索耗时 (ms)':<12}")
print("-" * 50)
print(f"{'K-Means (Top-5)':<20} | {km_acc:<12.2f} | {km_time:<12.4f}")
print(f"{'LSH (5-Tables)':<20} | {lsh_acc:<12.2f} | {lsh_time:<12.4f}")
print("="*50)
print("公平性说明：两者第一层检索均涉及 5 次以上的向量计算。")