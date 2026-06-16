# Hie-PriKG
Hie-PriKG 是一个面向大规模知识图谱的**分层隐私保护索引与检索系统**。本项目通过结合 **内部乘积函数加密 (IPFE)** 和 **非对称标量积保留加密 (ASPE)**，在保护用户查询隐私和服务器底库数据安全的前提下，实现了对图谱实体（基于 FB15k-237 数据集）的高效、精准语义检索。## 📊 架构设计*(请将架构图导出并命名为 `architecture.png` 放置在项目的 `assets/` 目录下)*
![System Architecture](assets/architecture.png)

系统的整体流程采用了直观的四宫格 (Four-square Quadrant) 布局进行模块化设计，将复杂的安全检索解耦为以下四个阶段：数据预处理、分层索引构建、核心密码学密钥/密文生成，以及双层密态检索。## ✨ 核心特性- **双层加密检索架构**：第一层使用 IPFE 掩蔽聚类质心（或 LSH 投影面）实现桶定位，第二层使用 ASPE 保护桶内实体向量进行精准打分，实现了“先粗筛、后精筛”的 O(√N) 级检索效率。- **严谨的密码学安全**：内置了经过严格校准的 ASPE 矩阵乘法计算逻辑，确保在密文域中精确无误地恢复内积。通过引入随机缩放因子 $r$ 与隐藏查询向量的掩码矩阵 $M_{l1}$，有效抵御统计分析攻击，满足 IND-CQA 安全标准。
- **卓越的召回性能**：在 SBERT 提取的 384 维稠密语义空间中，通过优化的多表局部敏感哈希 (Multi-table LSH) 或 K-Means 策略，第一层桶定位可实现高达 100% 的命中率。
- **工程化与自动化测试**：提供了一键式的数学同态性与密文标量积还原自检工具，保障底层密码学链路的绝对闭环。

## 📁 目录结构

```text
Hie-PriKG/
├── data_processing/        # 数据预处理模块 (包含数据集下载与 SBERT 实体向量化)
├── indexing/               # 分层索引构建模块 (K-Means 聚类与 LSH 分桶划分)
├── crypto_core/            # 核心加密模块 (IPFE 与 ASPE 的密钥生成与数据加密)
├── evaluation/             # 测试评估模块 (多算法公平对比与性能指标测试)
├── tests/                  # 自动化测试用例 (底库加密逻辑与数学链路验证)
├── data/                   # 本地数据集与密文库暂存区 (受 .gitignore 保护)
└── README.md
⚙️ 快速开始 (Quick Start)
1. 环境配置
请确保你的系统已安装 Python 3.8+，并安装以下核心依赖：

Bash

pip install numpy pandas sentence-transformers scikit-learn pytest requests
2. 数据准备与预处理
在项目根目录下，依次运行以下脚本下载数据并生成向量：

Bash

python data_processing/download_fb15k.py
python data_processing/entity_encoder.py
3. 构建安全索引并加密底库

Bash

# 生成分层索引 (以 K-Means 为例)
python indexing/kmeans_clustering.py# 执行 IPFE 第一层加密与 ASPE 第二层加密
python crypto_core/ipfe_encryption.py
python crypto_core/aspe_encryption.py
4. 运行密态检索
体验从输入明文问题到密文解析的全过程：

Bash

python crypto_core/secure_retrieval.py
5. 运行自动化测试
验证系统密码学链路的数学正确性：

Bash

pytest tests/
🚀 下一步计划 (Roadmap)
本仓库目前完整实现了基于 Python 的 IPFE + ASPE 密码学检索链路的雏形。虽然当前的方案在语义匹配上表现优异，但仍存在双向隐藏（双向隐私保护）的理论局限性。
下一阶段计划：完全函数隐藏内部乘积加密 (Fully Function-Hiding IPE, FH-IPE)
为了实现更高级别的安全性，项目将进行底层的技术栈重构：

C++ 核心重构：Python 在底层复杂密码学配对运算（Pairing-based Cryptography）上存在严重的性能瓶颈，且缺乏对真正的 FH-IPE 的原生高效支持。下一版本将使用 C++ 编写核心加密库，以获得原生的硬件级加速。
Docker 容器化部署：考虑到 C++ 密码学依赖库（如 PBC, GMP）的编译环境配置较为繁琐，完整的 FH-IPE 检索节点将被封装在 Docker 容器中运行，实现开箱即用。