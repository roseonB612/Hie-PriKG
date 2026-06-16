import numpy as np
import os
from sentence_transformers import SentenceTransformer

# ================= 1. 环境加载与实体映射 =================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_entity_name(entity_id):
    """从本地文件读取 ID 对应的真实名称"""
    possible_files = ["entity2text.txt", "entity_names.txt", "entities.txt"]
    for f_name in possible_files:
        path = os.path.join(ROOT_DIR, f_name)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if entity_id < len(lines):
                    parts = lines[entity_id].strip().split('\t')
                    return parts[1] if len(parts) > 1 else parts[0]
    return f"Unknown (ID: {entity_id})"

# 加载加密组件
centroids = np.load(os.path.join(ROOT_DIR, "centroids.npy"))
enc_centroids = np.load(os.path.join(ROOT_DIR, "encrypted_centroids.npy"))
msk_l1 = np.load(os.path.join(ROOT_DIR, "layer1_msk.npy"))
enc_lib_a = np.load(os.path.join(ROOT_DIR, "enc_aspe_lib_a.npy"))
enc_lib_b = np.load(os.path.join(ROOT_DIR, "enc_aspe_lib_b.npy"))
real_labels = np.load(os.path.join(ROOT_DIR, "bucket_labels.npy"))
S = np.load(os.path.join(ROOT_DIR, "aspe_S.npy"))
M1_inv = np.load(os.path.join(ROOT_DIR, "aspe_M1_inv.npy"))
M2_inv = np.load(os.path.join(ROOT_DIR, "aspe_M2_inv.npy"))

# ================= 2. 全流程透明化函数 =================

def ultimate_dive(raw_text):
    d = 10 # 选定第10维作为观察窗口
    
    print(f"\n{'='*25} 密态检索：从乱码到真相 {'='*25}")
    print(f"【用户明文输入】: '{raw_text}'")

    # --- Step 1: 用户侧编码与 IPFE 准备 ---
    query_vec = model.encode(raw_text)
    sk_q = np.dot(msk_l1, query_vec) # 生成功能私钥

    # --- Step 2: 服务端 IPFE 定位过程 ---
    # 模拟服务器找到最匹配的质心
    scores_l1 = np.dot(enc_centroids, query_vec) - sk_q
    target_bucket = np.argmax(scores_l1)
    
    print(f"\n[阶段一：IPFE 密文定位 (维度 {d} 视角)]")
    print(f" - 质心明文 C: {centroids[target_bucket][d]:.6f}")
    print(f" - 服务器存的密文 (C+MSK): {enc_centroids[target_bucket][d]:.6f}")
    print(f" - 用户发送的私钥 sk_q (标量): {sk_q:.4f}")
    print(f" * 匹配结论: 锁定第 {target_bucket} 号密文桶")

    # --- Step 3: 服务端 ASPE 精筛过程 ---
    # 模拟生成 ASPE 令牌 (r=1.0 方便观察)
    r = 1.0
    q_ext = r * np.concatenate([query_vec, [0, 1]])
    qa_raw, qb_raw = np.zeros_like(q_ext), np.zeros_like(q_ext)
    for i in range(len(S)):
        if S[i] == 0: qa_raw[i] = qb_raw[i] = q_ext[i]
        else: v = np.random.randn(); qa_raw[i] = q_ext[i]-v; qb_raw[i] = v
    token_a = np.dot(M1_inv, qa_raw)
    token_b = np.dot(M2_inv, qb_raw)

    # 桶内检索
    in_bucket_indices = np.where(real_labels == target_bucket)[0]
    best_score = -1e9
    best_id = -1
    
    for idx in in_bucket_indices:
        score = np.dot(enc_lib_a[idx], token_a) + np.dot(enc_lib_b[idx], token_b)
        if score > best_score:
            best_score = score
            best_id = idx

    print(f"\n[阶段二：ASPE 密文精筛 (维度 {d} 视角)]")
    print(f" - 用户发送的查询令牌 Token_A[{d}]: {token_a[d]:.6f}")
    print(f" - 服务器里的底库密文 Lib_A[{best_id}][{d}]: {enc_lib_a[best_id][d]:.6f}")
    print(f" * 匹配结论: 在密文桶中通过内积运算找到了最佳匹配项")

    # --- Step 4: 结果真相大白 ---
    final_name = get_entity_name(best_id)
    print(f"\n[阶段三：检索结果还原]")
    print(f" > 匹配实体 ID: {best_id}")
    print(f" > 实体真实名称: {final_name}  <-- 【这就是你要的直观结果！】")
    print(f" > 语义相关度得分: {best_score/r:.4f}")
    print(f"{'='*60}")

# 执行
ultimate_dive("Who is the president of USA?")