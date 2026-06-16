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
DIM = 384
EXT_DIM = DIM + 2 # 扩展维度：+1(防缩放) +1(常数项)

# ================= 1. ASPE 核心密码学模块 =================

def generate_invertible_matrix(n):
    """生成 n x n 的随机可逆矩阵"""
    while True:
        matrix = np.random.randn(n, n)
        # 提示：由于 386x386 的矩阵行列式极易溢出，这里也可以用 np.linalg.cond 判断条件数
        if np.abs(np.linalg.det(matrix)) > 1e-3: 
            return matrix

def encrypt_entity_aspe(x, S, M1, M2):
    """加密底库实体向量 (对应 7.1)"""
    x_ext = np.concatenate([x, [1, 0]]) # 补充 [1, 0] 用于计算平移
    x_a, x_b = np.zeros_like(x_ext), np.zeros_like(x_ext)
    
    for i in range(len(S)):
        if S[i] == 1:
            x_a[i] = x_b[i] = x_ext[i]
        else:
            v = np.random.randn()
            x_a[i] = x_ext[i] - v
            x_b[i] = v
            
    return np.dot(M1.T, x_a), np.dot(M2.T, x_b)

def generate_query_token_aspe(q, S, M1_inv, M2_inv, fixed_r=None):
    """生成查询令牌 (整合了 7.2 和 7.3 的逻辑)"""
    r = fixed_r if fixed_r is not None else np.random.uniform(0.5, 1.5)
    q_ext = r * np.concatenate([q, [0, 1]])
    q_a, q_b = np.zeros_like(q_ext), np.zeros_like(q_ext)
    
    for i in range(len(S)):
        if S[i] == 0:
            q_a[i] = q_b[i] = q_ext[i]
        else:
            v = np.random.randn()
            q_a[i] = q_ext[i] - v
            q_b[i] = v
            
    return np.dot(M1_inv, q_a), np.dot(M2_inv, q_b), r


# ================= 2. 业务执行模块 =================

def setup_aspe_keys():
    """【对应 7.0】: 生成并保存 ASPE 密钥体系"""
    print(f"\n--- 开始生成 ASPE 密钥 (维度: {EXT_DIM}) ---")
    np.random.seed(42) 
    
    S = np.random.randint(0, 2, EXT_DIM)
    M1 = generate_invertible_matrix(EXT_DIM)
    M2 = generate_invertible_matrix(EXT_DIM)
    M1_inv = np.linalg.inv(M1)
    M2_inv = np.linalg.inv(M2)

    np.save(os.path.join(ROOT_DIR, "aspe_S.npy"), S)
    np.save(os.path.join(ROOT_DIR, "aspe_M1.npy"), M1)
    np.save(os.path.join(ROOT_DIR, "aspe_M2.npy"), M2)
    np.save(os.path.join(ROOT_DIR, "aspe_M1_inv.npy"), M1_inv)
    np.save(os.path.join(ROOT_DIR, "aspe_M2_inv.npy"), M2_inv)
    
    print(f"[成功] ASPE 密钥生成完成并已保存至 {ROOT_DIR}")

def encrypt_full_database():
    """【对应 7.1】: 对整个向量底库进行 ASPE 加密"""
    data = np.load(os.path.join(ROOT_DIR, "fb15k_entity_embeddings.npy"))
    S = np.load(os.path.join(ROOT_DIR, "aspe_S.npy"))
    M1 = np.load(os.path.join(ROOT_DIR, "aspe_M1.npy"))
    M2 = np.load(os.path.join(ROOT_DIR, "aspe_M2.npy"))

    print(f"\n--- 正在执行 ASPE 数据加密 ({len(data)}条) ---")
    enc_library_a, enc_library_b = [], []

    for vec in data:
        ca, cb = encrypt_entity_aspe(vec, S, M1, M2)
        enc_library_a.append(ca)
        enc_library_b.append(cb)

    np.save(os.path.join(ROOT_DIR, "enc_aspe_lib_a.npy"), np.array(enc_library_a))
    np.save(os.path.join(ROOT_DIR, "enc_aspe_lib_b.npy"), np.array(enc_library_b))
    print(f"[成功] ASPE 密文底库构建完成！")

def test_hybrid_retrieval(test_idx=88):
    """【对应 7.2】: K-Means(IPFE) + ASPE 全流程检索测试"""
    data = np.load(os.path.join(ROOT_DIR, "fb15k_entity_embeddings.npy"))
    real_labels = np.load(os.path.join(ROOT_DIR, "bucket_labels.npy"))
    enc_centroids = np.load(os.path.join(ROOT_DIR, "encrypted_centroids.npy"))
    msk_l1 = np.load(os.path.join(ROOT_DIR, "layer1_msk.npy"))
    
    enc_lib_a = np.load(os.path.join(ROOT_DIR, "enc_aspe_lib_a.npy"))
    enc_lib_b = np.load(os.path.join(ROOT_DIR, "enc_aspe_lib_b.npy"))
    S = np.load(os.path.join(ROOT_DIR, "aspe_S.npy"))
    M1_inv = np.load(os.path.join(ROOT_DIR, "aspe_M1_inv.npy"))
    M2_inv = np.load(os.path.join(ROOT_DIR, "aspe_M2_inv.npy"))

    query_vec = data[test_idx]
    print(f"\n--- 开始 ASPE + IPFE 混合检索测试 (测试目标 ID: {test_idx}) ---")
    start_time = time.time()

    # 1. IPFE 定位桶
    sk_q = np.dot(msk_l1, query_vec)
    scores = np.dot(enc_centroids, query_vec) - sk_q
    target_bucket = np.argmax(scores)

    # 2. 生成 ASPE 令牌
    qa, qb, _ = generate_query_token_aspe(query_vec, S, M1_inv, M2_inv)

    # 3. 桶内精筛
    in_bucket_indices = np.where(real_labels == target_bucket)[0]
    best_score = -1e9
    pred_id = -1

    for idx in in_bucket_indices:
        score = np.dot(enc_lib_a[idx], qa) + np.dot(enc_lib_b[idx], qb)
        if score > best_score:
            best_score = score
            pred_id = idx

    duration = (time.time() - start_time) * 1000
    print(f"[检索报告]")
    print(f"定位桶: {target_bucket} (真实: {real_labels[test_idx]})")
    print(f"匹配 ID: {pred_id} (目标: {test_idx})")
    print(f"状态: {'【完美匹配】' if pred_id == test_idx else '【误差】'}")
    print(f"耗时: {duration:.2f} ms")

def verify_math_link(sample_indices=[10, 500, 14000]):
    """【对应 7.3】: 深入验证 IPFE 与 ASPE 的数学计算链路"""
    data = np.load(os.path.join(ROOT_DIR, "fb15k_entity_embeddings.npy"))
    centroids = np.load(os.path.join(ROOT_DIR, "centroids.npy"))
    msk_l1 = np.load(os.path.join(ROOT_DIR, "layer1_msk.npy"))
    enc_centroids = np.load(os.path.join(ROOT_DIR, "encrypted_centroids.npy"))
    
    S = np.load(os.path.join(ROOT_DIR, "aspe_S.npy"))
    M1_inv = np.load(os.path.join(ROOT_DIR, "aspe_M1_inv.npy"))
    M2_inv = np.load(os.path.join(ROOT_DIR, "aspe_M2_inv.npy"))

    for test_idx in sample_indices:
        print(f"\n{'='*30} 目标实体 ID: {test_idx} {'='*30}")
        query_vec = data[test_idx]
        true_bucket = np.load(os.path.join(ROOT_DIR, "bucket_labels.npy"))[test_idx]
        
        print(f"[1. 查询明文] 前5维分量: {query_vec[:5]}")
        print(f"[2. 目标质心] ID: {true_bucket}, 质心前5维: {centroids[true_bucket][:5]}")

        # IPFE 验证
        sk_q = np.dot(msk_l1, query_vec)
        ipfe_score = np.dot(enc_centroids[true_bucket], query_vec) - sk_q
        actual_inner_product = np.dot(centroids[true_bucket], query_vec)
        
        print(f"[3. IPFE 过程]")
        print(f"   - 质心密文(前3维): {enc_centroids[true_bucket][:3]}")
        print(f"   - 功能私钥 sk_q (标量): {sk_q:.4f}")
        print(f"   - 密态计算结果: {ipfe_score:.6f}")
        print(f"   - 误差: {abs(ipfe_score - actual_inner_product):.2e}")

        # ASPE 验证 (固定 r = 1.2 方便观察)
        qa, qb, r = generate_query_token_aspe(query_vec, S, M1_inv, M2_inv, fixed_r=1.2)
        print(f"[4. ASPE 过程]")
        print(f"   - 查询令牌 token_a (前3维): {qa[:3]}")
        
        ca = np.load(os.path.join(ROOT_DIR, "enc_aspe_lib_a.npy"))[test_idx]
        cb = np.load(os.path.join(ROOT_DIR, "enc_aspe_lib_b.npy"))[test_idx]
        
        aspe_score = np.dot(ca, qa) + np.dot(cb, qb)
        print(f"   - 密文相似度得分: {aspe_score:.6f}")
        print(f"   - 消除缩放 r 后的得分: {aspe_score/r:.6f}")
        print(f"   - 原始向量自相关内积: {np.dot(query_vec, query_vec):.6f}")

        success = "YES" if abs(aspe_score/r - np.dot(query_vec, query_vec)) < 1e-5 else "NO"
        print(f"[5. 结论] IPFE与ASPE计算链路闭环: {success}")


# ================= 3. 主执行入口 =================
if __name__ == "__main__":
    # 你可以通过注释/取消注释来选择要运行的模块
    
    # setup_aspe_keys()              # 生成密钥 (仅需运行一次)
    # encrypt_full_database()        # 加密底库 (仅需运行一次)
    
    test_hybrid_retrieval(test_idx=88)   # 测试混合检索
    # verify_math_link([10, 500, 14000]) # 验证数学链路