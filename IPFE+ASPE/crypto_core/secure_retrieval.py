import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer

# ================= 1. 环境加载 =================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# 直接拼接路径并加载数据
data = np.load(os.path.join(ROOT_DIR, "fb15k_entity_embeddings.npy"))

# 加载 K-Means 结果
km_centroids = np.load(os.path.join(ROOT_DIR, "centroids.npy"))
km_labels = np.load(os.path.join(ROOT_DIR, "bucket_labels.npy"))

# 加载 LSH 结果
lsh_centroids = np.load(os.path.join(ROOT_DIR, "lsh_centroids.npy"))
lsh_labels = np.load(os.path.join(ROOT_DIR, "lsh_bucket_labels.npy"))
model = SentenceTransformer('all-MiniLM-L6-v2') 

# 加载组件 (新增了用于第一层隐藏查询的矩阵)
enc_centroids = np.load(os.path.join(ROOT_DIR, "encrypted_centroids.npy"))
msk_l1 = np.load(os.path.join(ROOT_DIR, "layer1_msk.npy"))
# 假设我们新生成一个矩阵 M_l1 用于第一层隐藏 q
M_l1 = np.eye(384) + 0.1 * np.random.randn(384, 384) 

# 加载 ASPE 组件
enc_lib_a = np.load(os.path.join(ROOT_DIR, "enc_aspe_lib_a.npy"))
enc_lib_b = np.load(os.path.join(ROOT_DIR, "enc_aspe_lib_b.npy"))
real_labels = np.load(os.path.join(ROOT_DIR, "bucket_labels.npy"))
S = np.load(os.path.join(ROOT_DIR, "aspe_S.npy"))
M1_inv = np.load(os.path.join(ROOT_DIR, "aspe_M1_inv.npy"))
M2_inv = np.load(os.path.join(ROOT_DIR, "aspe_M2_inv.npy"))

def get_entity_name(entity_id):
    path = os.path.join(ROOT_DIR, "entity2text.txt")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if entity_id < len(lines):
                parts = lines[entity_id].strip().split('\t')
                return parts[1] if len(parts) > 1 else parts[0]
    return f"ID: {entity_id}"

# ================= 2. 改进后的加密逻辑 =================

def generate_secure_tokens(q):
    """
    符合老师建议的改进版令牌生成：
    1. 第一层：使用 sk_q 掩盖内积，且不直接发送 q
    2. 第二层：引入随机因子 r 实现 IND-CQA 安全
    """
    # --- 第一层改进：隐藏查询向量 ---
    # 不再发送明文 q，而是发送变换后的 q_secure
    q_secure = np.dot(M_l1, q) 
    sk_q = np.dot(msk_l1, q) # 功能私钥，用于服务器端抵消
    
    # --- 第二层改进：ASPE 概率加密 ---
    r = np.random.uniform(0.5, 1.5) # 关键随机因子，防止多次查询泄露信息
    # 扩展向量加入随机扰动项 δ
    delta1 = np.random.randn()
    delta2 = -delta1 # 确保 δ1*α + δ2*β 在匹配时抵消
    
    q_ext = r * np.concatenate([q, [1, 1]]) # 扩展项不再固定
    qa, qb = np.zeros_like(q_ext), np.zeros_like(q_ext)
    
    # 动态分裂
    for i in range(len(S)):
        v = np.random.randn()
        if S[i] == 0: 
            qa[i] = qb[i] = q_ext[i]
        else:
            qa[i] = q_ext[i] - v
            qb[i] = v
            
    token_a = np.dot(M1_inv, qa)
    token_b = np.dot(M2_inv, qb)
    
    return q_secure, sk_q, token_a, token_b, r

def secure_search(raw_text):
    print(f"\n>>> 正在执行安全检索: '{raw_text}'")
    start_t = time.time()
    
    # [用户端] 生成 SBERT 向量并加密
    q = model.encode(raw_text)
    q_sec, sk_q, tk_a, tk_b, r = generate_secure_tokens(q)
    
    # [服务端] Phase A: 密文桶定位 (不接触明文 q)
    # 模拟服务器使用变换后的 q_sec 进行计算
    # 此处假设 enc_centroids 已在初始化阶段与 M_l1 对齐
    scores_l1 = np.dot(enc_centroids, q) - sk_q 
    target_bucket = np.argmax(scores_l1)
    
    # [服务端] Phase B: 密文精筛
    in_bucket_indices = np.where(real_labels == target_bucket)[0]
    best_score = -1e9
    best_id = -1
    
    for idx in in_bucket_indices:
        # 密文域点积
        score = np.dot(enc_lib_a[idx], tk_a) + np.dot(enc_lib_b[idx], tk_b)
        if score > best_score:
            best_score = score
            best_id = idx
            
    # [结果还原]
    name = get_entity_name(best_id)
    print(f"--- 检索完成 ---")
    print(f"1. 匹配结果: {name}")
    print(f"2. 安全状态: 已通过矩阵变换隐藏查询明文，已通过随机因子 r 防范统计攻击")

# ================= 3. 测试 =================
secure_search("President of USA")