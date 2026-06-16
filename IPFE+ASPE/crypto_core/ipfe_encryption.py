import numpy as np
import os
import time

# ================= 配置区域 =================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# 直接拼接路径并加载数据
data = np.load(os.path.join(ROOT_DIR, "fb15k_entity_embeddings.npy"))

# 加载 K-Means 结果
km_centroids = np.load(os.path.join(ROOT_DIR, "centroids.npy"))
km_labels = np.load(os.path.join(ROOT_DIR, "bucket_labels.npy"))

# 加载 LSH 结果
lsh_centroids = np.load(os.path.join(ROOT_DIR, "lsh_centroids.npy"))
lsh_labels = np.load(os.path.join(ROOT_DIR, "lsh_bucket_labels.npy"))
# ================= 1. IPFE 核心密码学模块 =================
class IPFECore:
    """
    内部乘积函数加密 (Inner Product Functional Encryption) 核心实现
    负责底层的密钥生成、数据加密和密态内积解密
    """
    def __init__(self, dim, seed=None):
        self.dim = dim
        if seed is not None:
            np.random.seed(seed)
        # 生成主密钥 s (Master Secret Key)
        self.msk = np.random.randn(dim)

    def encrypt(self, x):
        """加密目标向量 x"""
        # 简化演示：实际 IPFE 密文会包含多个分量且经过离散对数映射
        return x + self.msk

    def key_gen(self, y):
        """为查询向量 y 生成功能私钥 (Functional Private Key)"""
        return np.dot(self.msk, y)

    def decrypt(self, ct, sk_y, y):
        """利用私钥 sk_y 和查询向量 y 解密密文，还原内积"""
        # 模拟 IPFE 的同态特性：<x+msk, y> - <msk, y> = <x, y>
        return np.dot(ct, y) - sk_y


# ================= 2. 业务加密执行模块 =================

def encrypt_kmeans_centroids(root_dir=ROOT_DIR):
    """场景 A: 对 K-Means 生成的第一层质心进行加密"""
    centroids_path = os.path.join(root_dir, "centroids.npy")
    if not os.path.exists(centroids_path):
        print(f"⚠️ 找不到 {centroids_path}，跳过 K-Means 质心加密。")
        return

    centroids = np.load(centroids_path)
    num_buckets, dim = centroids.shape
    
    print(f"\n--- 开始 K-Means 第一层质心加密 ---")
    print(f"质心数量: {num_buckets}, 向量维度: {dim}")

    ipfe_l1 = IPFECore(dim, seed=999)
    start_time = time.time()
    
    # 批量加密
    encrypted_centroids = np.array([ipfe_l1.encrypt(c) for c in centroids])
    duration = time.time() - start_time

    np.save(os.path.join(root_dir, 'encrypted_centroids.npy'), encrypted_centroids)
    np.save(os.path.join(root_dir, 'layer1_msk.npy'), ipfe_l1.msk)

    print(f"[成功] K-Means 质心加密完成！耗时: {duration:.4f} 秒")
    print(f"已保存密文: encrypted_centroids.npy (形状: {encrypted_centroids.shape})")


def encrypt_lsh_planes(root_dir=ROOT_DIR):
    """场景 B: 对 LSH 算法生成的随机投影矩阵进行加密"""
    planes_path = os.path.join(root_dir, "lsh_planes.npy")
    if not os.path.exists(planes_path):
        print(f"⚠️ 找不到 {planes_path}，跳过 LSH 投影矩阵加密。")
        return

    planes = np.load(planes_path)
    dim, num_planes = planes.shape 
    
    print(f"\n--- 开始 LSH 投影矩阵加密 ---")
    print(f"投影面维度: {dim}, 投影面数量: {num_planes}")

    ipfe_lsh = IPFECore(dim, seed=888)
    start_time = time.time()
    
    # 矩阵每一列代表一个投影面，逐列加密
    encrypted_planes = np.array([ipfe_lsh.encrypt(planes[:, i]) for i in range(num_planes)]).T 
    duration = time.time() - start_time

    np.save(os.path.join(root_dir, 'encrypted_lsh_planes.npy'), encrypted_planes)
    np.save(os.path.join(root_dir, 'lsh_layer1_msk.npy'), ipfe_lsh.msk)

    print(f"[成功] LSH 投影矩阵加密完成！耗时: {duration:.4f} 秒")
    print(f"已保存密文投影矩阵: encrypted_lsh_planes.npy (形状: {encrypted_planes.shape})")


# ================= 3. 基础逻辑自检测试 =================

def self_test():
    """验证 IPFE 数学同态特性是否闭环"""
    print(f"\n--- IPFE 基础数学同态性自检 ---")
    dim = 384
    x = np.random.randn(dim) # 模拟待加密实体
    y = np.random.randn(dim) # 模拟查询向量

    ground_truth = np.dot(x, y)
    
    ipfe = IPFECore(dim)
    t0 = time.time()
    ct = ipfe.encrypt(x)          
    sk_y = ipfe.key_gen(y)        
    result = ipfe.decrypt(ct, sk_y, y) 
    t1 = time.time()

    print(f"明文计算结果: {ground_truth:.6f}")
    print(f"IPFE 解密结果: {result:.6f}")
    print(f"绝对误差: {abs(ground_truth - result):.2e}")
    
    if abs(ground_truth - result) < 1e-9:
        print("[自检通过] 基础 IPFE 逻辑闭环正确！")
    else:
        print("[自检失败] 结果不一致，请检查核心实现。")


# ================= 主执行入口 =================
if __name__ == "__main__":
    # 1. 运行自检验证数学逻辑
    self_test()
    
    # 2. 执行实际的数据底库加密
    encrypt_kmeans_centroids()
    encrypt_lsh_planes()