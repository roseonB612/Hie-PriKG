import numpy as np
import os
from collections import Counter
import time

# ================= 1. 加载数据 =================
data_path = "fb15k_entity_embeddings.npy" # 请确保这是你 SBERT 向量的真实路径

if not os.path.exists(data_path):
    print(f"找不到文件: {data_path}，正在生成模拟数据进行演示...")
    data = np.random.randn(14951, 384).astype(np.float32)
else:
    data = np.load(data_path)
    print(f"--- 向量数据加载成功，形状: {data.shape} ---")

# ================= 2. LSH 分桶配置 =================
# 目标桶数约 100 个，由于 LSH 是 2^n 进制，取 num_bits = 7 (即 128 个桶)
NUM_BITS = 7 
DIM = data.shape[1]

print(f"正在进行 LSH 分桶，位数: {NUM_BITS} (理论最大桶数: {2**NUM_BITS})...")
start_time = time.time()

# --- 核心算法 ---
# 生成随机投影平面 (固定种子以保证实验可重复性)
np.random.seed(42) 
planes = np.random.randn(DIM, NUM_BITS).astype(np.float32)

# 计算投影并生成哈希码
# dot product: (14951, 384) * (384, 7) -> (14951, 7)
projections = np.dot(data, planes)
hashes = (projections > 0).astype(int)

# 将二进制位转换为十进制桶 ID
powers_of_two = 1 << np.arange(NUM_BITS)[::-1]
bucket_labels = np.dot(hashes, powers_of_two)

duration = time.time() - start_time
print(f"LSH 分桶完成！耗时: {duration:.4f} 秒")

# ================= 3. 统计结果 =================
counts = list(Counter(bucket_labels).values())
num_active_buckets = len(counts)
avg_size = np.mean(counts)
max_size = np.max(counts)
min_size = np.min(counts)
std_dev = np.std(counts)

print("\n--- LSH 分桶质量统计 ---")
print(f"实际占用桶数: {num_active_buckets}")
print(f"平均每个桶包含实体数: {avg_size:.1f}")
print(f"最大桶包含实体数: {max_size}")
print(f"最小桶包含实体数: {min_size}")
print(f"分桶标准差: {std_dev:.2f} (通常比 K-Means 大)")

# ================= 4. 保存结果 =================
save_dir = "e:/实验数据/"
np.save(os.path.join(save_dir, 'lsh_bucket_labels.npy'), bucket_labels)
np.save(os.path.join(save_dir, 'lsh_planes.npy'), planes)

print(f"\n[成功] 结果已保存至 {save_dir}：")
print("- lsh_bucket_labels.npy (分配标签，用于第二层定位)")
print("- lsh_planes.npy (随机投影矩阵，用于第一层检索)")