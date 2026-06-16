import requests
import os

# --- 配置区域 ---
# 请确保这里是你存放 FB15k-237 文件的那个文件夹路径
DATA_DIR = r"E:\实验数据\FB15k-237-main"

# 目标文件名
TARGET_FILE = os.path.join(DATA_DIR, "entity2text.txt")

# 备选下载源列表 (按推荐程度排序)
# 来源1: SimKGC (顶会论文，描述质量很高)
# 来源2: PKU (北大开源数据集仓库)
# 来源3: KG-BERT (经典的描述文件)
URLS = [
    "https://raw.githubusercontent.com/intfloat/SimKGC/main/data/FB15k237/entity2text.txt",
    "https://raw.githubusercontent.com/xujun-pku/KG-datasets/master/FB15k-237/entity2text.txt",
    "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237/entity2text.txt"
]

def download_mapping():
    if not os.path.exists(DATA_DIR):
        print(f"错误：文件夹 {DATA_DIR} 不存在！请先创建或修改路径。")
        return

    print(f"准备下载实体描述文件到: {TARGET_FILE}")
    
    for i, url in enumerate(URLS):
        print(f"--- 尝试第 {i+1} 个源: {url} ---")
        try:
            # 增加 verify=False 防止某些 SSL 证书报错，timeout 防止卡死
            response = requests.get(url, timeout=20, verify=False)
            
            if response.status_code == 200:
                # 检查一下内容是不是空的
                if len(response.content) > 1000:
                    with open(TARGET_FILE, "wb") as f:
                        f.write(response.content)
                    print(f"✅ 成功！文件已保存。")
                    print(f"文件大小: {len(response.content)/1024:.2f} KB")
                    print("现在你可以重新运行 1.py 了！")
                    return
                else:
                    print("⚠️ 下载虽然成功，但文件太小，可能不对。尝试下一个...")
            else:
                print(f"❌ 下载失败，状态码: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 发生网络错误: {e}")
            
    print("\n😭 所有源都尝试失败了。")
    print("建议方案：")
    print("1. 请检查你的网络是否能访问 GitHub (可能需要科学上网)。")
    print("2. 尝试下面的'方案二'手动复制。")

if __name__ == "__main__":
    # 忽略 SSL 警告
    import urllib3
    urllib3.disable_warnings()
    download_mapping()