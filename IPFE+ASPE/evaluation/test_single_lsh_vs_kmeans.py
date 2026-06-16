import numpy as np
import time
import os

# ================= 1. 环境准备 =================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# 直接拼接路径并【直接加载】到最终变量中
data = np.load(os.path.join(ROOT_DIR, "fb15k_entity_embeddings.npy"))

# 加载 K-Means 结果
km_centroids = np.load(os.path.join(ROOT_DIR, "centroids.npy"))
km_labels = np.load(os.path.join(ROOT_DIR, "bucket_labels.npy"))

# 加载 LSH 结果
lsh_centroids = np.load(os.path.join(ROOT_DIR, "lsh_centroids.npy"))
lsh_labels = np.load(os.path.join(ROOT_DIR, "lsh_bucket_labels.npy"))

# 随机抽取 100 个样本作为查询 Query
np.random.seed(123)
query_indices = np.random.choice(len(data), 100, replace=False)
queries = data[query_indices]

# 随机抽取 100 个样本作为查询 Query
np.random.seed(123)
query_indices = np.random.choice(len(data), 100, replace=False)
queries = data[query_indices]

def evaluate_retrieval(queries, query_indices, centroids, data_labels, method_name):
    print(f"--- 正在测试 {method_name} 检索性能 ---")
    start_time = time.time()
    hits = 0
    
    for i, query in enumerate(queries):
        # 第一层检索：寻找最近的质心 (使用余弦相似度/内积)
        # 注意：对于 IPFE 场景，通常使用内积
        similarities = np.dot(centroids, query)
        best_bucket_idx = np.argmax(similarities)
        
        # 验证：真实的标签是否对应这个桶？
        # 注意：LSH 的 centroid 索引是按 sorted_bucket_ids 排列的
        # 这里简化处理：直接对比该桶包含的索引是否包含原查询索引
        real_label = data_labels[query_indices[i]]
        
        # 获取该桶对应的所有数据索引 (这里为了方便直接对比标签)
        # 如果是 K-Means，best_bucket_idx 就是标签
        # 如果是 LSH，需要通过 mapping 找回，这里假设你已按顺序对齐
        if best_bucket_idx == real_label: 
            hits += 1
            
    avg_time = (time.time() - start_time) / len(queries) * 1000 # 毫秒
    accuracy = (hits / len(queries)) * 100
    return accuracy, avg_time

# ================= 2. 执行对比 =================
# 注意：LSH 这里的准确率定义为“Query 是否落入其质心代表的桶”
km_acc, km_time = evaluate_retrieval(queries, query_indices, km_centroids, km_labels, "K-Means")
lsh_acc, lsh_time = evaluate_retrieval(queries, query_indices, lsh_centroids, lsh_labels, "LSH")

# ================= 3. 输出报告 =================
print("\n" + "="*30)
print(f"{'指标':<15} | {'K-Means':<12} | {'LSH':<12}")
print("-" * 45)
print(f"{'第一层命中率 (%)':<15} | {km_acc:<12.2f} | {lsh_acc:<12.2f}")
print(f"{'单次检索耗时 (ms)':<15} | {km_time:<12.4f} | {lsh_time:<12.4f}")
print("="*30)
print("备注：命中率代表 Query 能够通过质心正确定位到其所在分桶的概率。")