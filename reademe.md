# Hie-PriKG: Hierarchical Privacy-preserving Knowledge Graph Retrieval

Hie-PriKG 是一个面向大规模知识图谱（Knowledge Graph, KG）的**分层隐私保护索引与检索框架**。该系统通过解耦检索流程，旨在解决高安全需求下的计算延迟瓶颈。

## 🔬 项目架构

本仓库包含两个核心实现版本，完整展示了从理论验证到高性能工程实现的演进路径：

* **`/IPFE+ASPE` (Python 原型)**: 
  基于 Python 的算法原型验证模块。利用 `IPFE`（内部乘积函数加密）实现聚类中心的粗筛，并结合 `ASPE`（非对称标量积保留加密）完成桶内实体的精准排序。该模块适合快速理解算法逻辑和进行数学链路闭环测试。

* **`/FH-IPE-CPP` (C++ 高性能引擎)**: 
  生产级高性能检索引擎。为了突破 Python 在底层双线性配对运算（Pairing-based Cryptography）上的性能瓶颈，我们使用 C++ 重构了核心算子。通过集成 `mcl` 和 `Eigen3` 库，实现了真正的 **FH-IPE（完全函数隐藏内部乘积加密）**，并支持硬件级并行加速。

## 🚀 快速导航

| 模块 | 核心语言 | 主要用途 | 状态 |
| :--- | :--- | :--- | :--- |
| **Python 原型** | Python | 算法理论验证、数学链路校验 | 稳定 |
| **C++ 引擎** | C++17 | 生产级检索、高性能 Benchmark | 稳定 |

## 🛠️ 如何编译与运行

请根据您的需求进入相应目录，参考各文件夹下的详细说明：

1. **若要进行算法验证**：进入 `IPFE+ASPE/`，查看其 `README.md`，运行 `pytest` 测试核心数学逻辑。
2. **若要运行高性能检索**：进入 `FH-IPE-CPP/`，利用提供的 `Dockerfile` 构建高性能执行环境，并参考其说明进行编译测试。

## 📄 引用说明
本项目相关论文正在发表中。如果您在研究中使用此代码，请按照以下规范引用：

# Hie-PriKG: Hierarchical Privacy-preserving Knowledge Graph Retrieval

Hie-PriKG 是一个面向大规模知识图谱（Knowledge Graph, KG）的**分层隐私保护索引与检索框架**。该系统通过解耦检索流程，旨在解决高安全需求下的计算延迟瓶颈。

## 🔬 项目架构

本仓库包含两个核心实现版本，完整展示了从理论验证到高性能工程实现的演进路径：

* **`/IPFE+ASPE` (Python 原型)**: 
  基于 Python 的算法原型验证模块。利用 `IPFE`（内部乘积函数加密）实现聚类中心的粗筛，并结合 `ASPE`（非对称标量积保留加密）完成桶内实体的精准排序。该模块适合快速理解算法逻辑和进行数学链路闭环测试。

* **`/FH-IPE-CPP` (C++ 高性能引擎)**: 
  生产级高性能检索引擎。为了突破 Python 在底层双线性配对运算（Pairing-based Cryptography）上的性能瓶颈，我们使用 C++ 重构了核心算子。通过集成 `mcl` 和 `Eigen3` 库，实现了真正的 **FH-IPE（完全函数隐藏内部乘积加密）**，并支持硬件级并行加速。

## 🚀 快速导航

| 模块 | 核心语言 | 主要用途 | 状态 |
| :--- | :--- | :--- | :--- |
| **Python 原型** | Python | 算法理论验证、数学链路校验 | 稳定 |
| **C++ 引擎** | C++17 | 生产级检索、高性能 Benchmark | 稳定 |

## 🛠️ 如何编译与运行

请根据您的需求进入相应目录，参考各文件夹下的详细说明：

1. **若要进行算法验证**：进入 `IPFE+ASPE/`，查看其 `README.md`，运行 `pytest` 测试核心数学逻辑。
2. **若要运行高性能检索**：进入 `FH-IPE-CPP/`，利用提供的 `Dockerfile` 构建高性能执行环境，并参考其说明进行编译测试。

## 📄 引用说明
本项目相关论文正在发表中。如果您在研究中使用此代码，请按照以下规范引用：

```bibtex
@misc{hieprikg2026,
  title={Hie-PriKG: Hierarchical Privacy-preserving Knowledge Graph Indexing and Retrieval System},
  author={[陕西师范大学-杨可颖], [陕西师范大学-王涛], [陕西师范大学-冉洁茹], et al.},
  year={2026},
  publisher={GitHub}
}