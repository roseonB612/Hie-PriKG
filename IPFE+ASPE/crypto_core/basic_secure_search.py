import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer

# ================= 1. 强力加载实体映射 =================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# 直接拼接路径并加载数据
data = np.load(os.path.join(ROOT_DIR, "fb15k_entity_embeddings.npy"))

# 加载 K-Means 结果
km_centroids = np.load(os.path.join(ROOT_DIR, "centroids.npy"))
km_labels = np.load(os.path.join(ROOT_DIR, "bucket_labels.npy"))

# 加载 LSH 结果
lsh_centroids = np.load(os.path.join(ROOT_DIR, "lsh_centroids.npy"))
lsh_labels = np.load(os.path.join(ROOT_DIR, "lsh_bucket_labels.npy"))

def load_entity_names_robust():
    # 尝试多种可能的文件名
    possible_files = ["entity2text.txt", "entity_names.txt", "entities.txt"]
    id_to_name = {}
    
    found_file = None
    for f_name in possible_files:
        path = os.path.join(ROOT_DIR, f_name)
        if os.path.exists(path):
            found_file = path
            break
    
    if found_file:
        print(f"--- 正在从 {os.path.basename(found_file)} 加载映射 ---")
        try:
            with open(found_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    parts = line.strip().split('\t')
                    # 适配两种格式：1. 只有名称  2. ID \t 名称
                    name = parts[1] if len(parts) > 1 else parts[0]
                    # 将行号或索引作为 key
                    id_to_name[i] = name
            print(f"--- 成功加载 {len(id_to_name)} 条实体名 ---")
        except Exception as e:
            print(f"!!! 加载失败: {e} !!!")
    else:
        print(f"!!! 警告：在 {ROOT_DIR} 下未找到任何实体名称文件 !!!")
        print("请检查文件夹中是否存在包含实体名的 .txt 文件")
        
    return id_to_name

# 加载实体名
entity_dict = load_entity_names_robust()

# ================= 2. 加载其余组件 (保持不变) =================
model = SentenceTransformer('all-MiniLM-L6-v2') 
enc_centroids = np.load(os.path.join(ROOT_DIR, "encrypted_centroids.npy"))
msk_l1 = np.load(os.path.join(ROOT_DIR, "layer1_msk.npy"))
enc_lib_a = np.load(os.path.join(ROOT_DIR, "enc_aspe_lib_a.npy"))
enc_lib_b = np.load(os.path.join(ROOT_DIR, "enc_aspe_lib_b.npy"))
real_labels = np.load(os.path.join(ROOT_DIR, "bucket_labels.npy"))
S = np.load(os.path.join(ROOT_DIR, "aspe_S.npy"))
M1_inv = np.load(os.path.join(ROOT_DIR, "aspe_M1_inv.npy"))
M2_inv = np.load(os.path.join(ROOT_DIR, "aspe_M2_inv.npy"))

def generate_aspe_token(q, S, M1_inv, M2_inv):
    r = np.random.uniform(0.8, 1.2) 
    q_ext = r * np.concatenate([q, [0, 1]])
    qa, qb = np.zeros_like(q_ext), np.zeros_like(q_ext)
    for i in range(len(S)):
        if S[i] == 0: qa[i] = qb[i] = q_ext[i]
        else:
            v = np.random.randn(); qa[i] = q_ext[i] - v; qb[i] = v
    return np.dot(M1_inv, qa), np.dot(M2_inv, qb), r

def real_world_search(raw_text):
    print(f"\n" + "-"*40)
    query_vec = model.encode(raw_text)
    
    # 模拟用户令牌生成
    sk_q = np.dot(msk_l1, query_vec)
    qa, qb, r = generate_aspe_token(query_vec, S, M1_inv, M2_inv)
    
    # 服务端计算
    scores = np.dot(enc_centroids, query_vec) - sk_q
    target_bucket = np.argmax(scores)
    
    in_bucket_indices = np.where(real_labels == target_bucket)[0]
    best_score = -1e9
    best_id = -1
    
    for idx in in_bucket_indices:
        score = np.dot(enc_lib_a[idx], qa) + np.dot(enc_lib_b[idx], qb)
        if score > best_score:
            best_score = score
            best_id = idx
            
    # 获取名称
    name = entity_dict.get(best_id, f"Unknown (ID: {best_id})")
    print(f"查询: {raw_text}")
    print(f"结果: {name} (相关度: {best_score/r:.4f})")

# 执行
queries = ["President of USA", "Capital of France", "Author of Harry Potter"]
for q in queries:
    real_world_search(q)