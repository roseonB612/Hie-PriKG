#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <numeric>
#include <algorithm>

using namespace Eigen;
using namespace std;

// 复用刚才的数据加载器
MatrixXd loadBinaryToEigen(const string& filename, int rows, int cols) {
    MatrixXd mat(rows, cols);
    ifstream file(filename, ios::binary);
    if (!file.is_open()) { cerr << "文件打开失败!" << endl; exit(1); }
    Matrix<double, Dynamic, Dynamic, RowMajor> tempMap(rows, cols);
    file.read(reinterpret_cast<char*>(tempMap.data()), rows * cols * sizeof(double));
    mat = tempMap;
    file.close();
    return mat;
}

int main() {
    int N = 14541; // FB15K237 实体数
    int d = 384;   // 原始维度
    int k = 10;    // Top-K

    cout << "1. 正在加载数据..." << endl;
    MatrixXd dataset = loadBinaryToEigen("/app/data/FB15K237_embeddings.bin", N, d);

    cout << "2. KeyGen: 生成 (d+2)x(d+2) 可逆密钥矩阵 M..." << endl;
    // 使用随机浮点数生成主密钥矩阵
    MatrixXd M = MatrixXd::Random(d + 2, d + 2);
    MatrixXd M_inv = M.inverse();

    cout << "3. EncData: 构建数据库增广矩阵并加密..." << endl;
    auto start_enc = chrono::high_resolution_clock::now();
    
    // 初始化 (N) x (d+2) 的增广矩阵
    MatrixXd P_aug(N, d + 2);
    P_aug.leftCols(d) = dataset;
    
    // 计算每个实体的 L2 范数平方
    VectorXd p_sq_norms = dataset.rowwise().squaredNorm();
    P_aug.col(d) = -0.5 * p_sq_norms; // 扩展维度 1
    P_aug.col(d + 1) = VectorXd::Ones(N); // 扩展维度 2

    // 极其核心：批量加密（注意方向，等价于 (M^T * p)^T ）
    MatrixXd I_enc = P_aug * M; 
    
    auto end_enc = chrono::high_resolution_clock::now();
    cout << "   -> 数据库离线加密耗时: " 
         << chrono::duration<double, milli>(end_enc - start_enc).count() << " ms" << endl;

    cout << "\n4. EncQuery: 生成查询陷门..." << endl;
    // 为了测试准确性，我们直接拿第 0 行作为查询目标
    VectorXd q = dataset.row(0).transpose(); 
    double r = 2.5; // ASPE 的动态随机混淆标量 (必须 > 0)
    
    VectorXd q_aug(d + 2);
    q_aug.head(d) = r * q;
    q_aug(d) = r;
    q_aug(d + 1) = -0.5 * r * q.squaredNorm();

    // 生成查询陷门
    VectorXd Q_trapdoor = M_inv * q_aug;

    cout << "5. Search: 执行密文全量检索 (Phase II 核心算子)..." << endl;
    auto start_search = chrono::high_resolution_clock::now();
    
    // 高能预警：一步完成所有 N 个实体的密文内积计算！
    VectorXd scores = I_enc * Q_trapdoor;

    // 获取 Top-K 的索引
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                 [&](int i, int j) { return scores(i) > scores(j); }); // 降序排序

    auto end_search = chrono::high_resolution_clock::now();
    double search_time = chrono::duration<double, milli>(end_search - start_search).count();
    
    cout << "   -> 全量检索 (Search) 耗时: " << search_time << " ms" << endl;

    cout << "\n检索结果 Top-5 ID: ";
    for(int i = 0; i < 5; ++i) cout << indices[i] << " ";
    cout << "\n(预期 Top-1 必须是 0，验证召回率逻辑)" << endl;

    return 0;
}
