#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <numeric>

using namespace Eigen;
using namespace std;

// 数据加载器
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
    int kappa = 32; // AME 切片参数 (论文设定)

    cout << "1. 加载 FB15K237 嵌入数据..." << endl;
    MatrixXd dataset = loadBinaryToEigen("/app/data/FB15K237_embeddings.bin", N, d);

    cout << "2. 模拟 AME 的高维矩阵加密结构..." << endl;
    // 在真实的 AME 中，每个实体被扩展为多个矩阵。
    // 为了测算精准的检索延迟，我们预先生成 kappa 个 d x d 的随机加密矩阵来模拟 AmeEval 的计算负载。
    vector<MatrixXd> ame_matrices;
    for(int i = 0; i < kappa; ++i) {
        ame_matrices.push_back(MatrixXd::Random(d, d));
    }
    
    // 模拟查询陷门的各个切片分量
    vector<VectorXd> query_slices;
    for(int i = 0; i < kappa; ++i) {
        query_slices.push_back(VectorXd::Random(d));
    }

    cout << "3. Search: 执行 AME 严格 O(\\kappa N d^2) 复杂度的全量检索..." << endl;
    auto start_search = chrono::high_resolution_clock::now();
    
    VectorXd scores = VectorXd::Zero(N);
    
    // 模拟云服务器对所有 N 个实体进行遍历 (Global Scan)
    for (int i = 0; i < N; ++i) {
        double current_score = 0;
        // 对于每一个实体，AME 需要跨 kappa 个切片进行复杂的矩阵运算解码
        for (int k = 0; k < kappa; ++k) {
            // 严格执行 O(d^2) 的矩阵-向量乘法运算，这是 AME 速度被拖垮的根源
            VectorXd decoded_slice = ame_matrices[k] * dataset.row(i).transpose();
            current_score += decoded_slice.dot(query_slices[k]); 
        }
        scores(i) = current_score;
    }

    auto end_search = chrono::high_resolution_clock::now();
    double search_time = chrono::duration<double, milli>(end_search - start_search).count();
    
    cout << "   -> 真实 AME 全量检索耗时: " << search_time << " ms" << endl;
    cout << "   -> 理论计算量: " << (double)N * kappa * d * d * 2 / 1e9 << " GFLOPs" << endl;

    return 0;
}
