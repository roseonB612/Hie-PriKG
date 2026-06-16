import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def process_fb15k_to_embeddings(data_path, model_name='all-MiniLM-L6-v2'):
    """
    读取 FB15k-237 数据，将其转换为 SBERT 向量并保存。
    """
    print(f"--- 开始处理数据集: {data_path} ---")

    # 1. 加载 SBERT 模型
    # 第一次运行会自动下载
    print(f"正在加载 SBERT 模型: {model_name}...")
    model = SentenceTransformer(model_name)

    # 2. 读取实体映射 (这是最关键的一步)
    # 假设你的文件叫 entity2text.txt，格式为: ID \t 描述
    # 如果你只有 entities.dict，你需要找一个带文字描述的版本
    mapping_file = os.path.join(data_path, 'entity2text.txt') 
    
    if not os.path.exists(mapping_file):
        print(f"警告: 未找到 {mapping_file}。")
        print("正在尝试读取 entities.dict 作为备选（注意：如果 dict 里只有 ID，向量语义会很弱）...")
        mapping_file = os.path.join(data_path, 'entities.dict')
        # 读取 entities.dict (通常格式是: index \t ID)
        df = pd.read_csv(mapping_file, sep='\t', header=None, names=['index', 'text'])
    else:
        # 读取 entity2text.txt
        df = pd.read_csv(mapping_file, sep='\t', header=None, names=['id', 'text'])

    entity_texts = df['text'].astype(str).tolist()
    print(f"成功读取 {len(entity_texts)} 个实体文本。")

    # 3. 提取向量 (Encoding)
    print("正在使用 SBERT 提取向量（如果实体较多，这可能需要几分钟）...")
    embeddings = model.encode(entity_texts, show_progress_bar=True, batch_size=32)

    # 4. 保存结果
    output_file = 'fb15k_entity_embeddings.npy'
    np.save(output_file, embeddings)
    
    # 同时保存一下文本索引，方便后面对应
    df.to_csv('entity_metadata.csv', index=False)

    print(f"--- 处理完成！ ---")
    print(f"向量矩阵形状: {embeddings.shape}")
    print(f"结果已保存至: {output_file}")
    return embeddings

# --- 运行脚本 ---
# 请确保路径指向你解压后的 FB15k-237 文件夹
DATASET_PATH = './FB15k-237-main' 

if __name__ == "__main__":
    if os.path.exists(DATASET_PATH):
        embeddings = process_fb15k_to_embeddings(DATASET_PATH)
    else:
        print(f"错误：找不到路径 {DATASET_PATH}，请检查文件夹名称。")