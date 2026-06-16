import numpy as np
from sklearn.cluster import KMeans
import time

def run_clustering():
    # 1. 加载 SBERT 向量
    print("--- 正在加载 SBERT 向量数据 ---")
    data = np.load('fb15k_entity_embeddings.npy')
    print(f"向量数据加载成功，形状: {data.shape}")

    # 2. 配置分桶参数
    # n_clusters=100 表示分 100 个桶。这是你论文中可以调整的超参数
    n_clusters = 100
    print(f"正在进行 K-Means 分桶，目标桶数: {n_clusters}...")

    start_time = time.time()
    # n_init=10 表示运行10次选最好的结果，random_state 确保实验可复现
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(data)
    end_time = time.time()

    # 3. 提取核心结果
    centroids = kmeans.cluster_centers_  # 形状 (100, 384)
    labels = kmeans.labels_              # 形状 (14951,)

    print(f"分桶完成！耗时: {end_time - start_time:.2f} 秒")

    # 4. 统计分桶均衡性 (这对论文实验分析非常有用)
    unique, counts = np.unique(labels, return_counts=True)
    print("\n--- 分桶质量统计 ---")
    print(f"平均每个桶包含实体数: {np.mean(counts):.1f}")
    print(f"最大桶包含实体数: {np.max(counts)}")
    print(f"最小桶包含实体数: {np.min(counts)}")
    print(f"分桶标准差: {np.std(counts):.2f} (越小代表分布越均匀)")

    # 5. 保存结果供后续加密阶段使用
    np.save('centroids.npy', centroids)
    np.save('bucket_labels.npy', labels)
    print("\n[成功] 结果已保存：")
    print("- centroids.npy (重心向量，用于第一层 IPFE)")
    print("- bucket_labels.npy (分配标签，用于第二层定位)")

if __name__ == "__main__":
    run_clustering()