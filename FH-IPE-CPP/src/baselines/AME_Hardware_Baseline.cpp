#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <omp.h>

using namespace Eigen;
using namespace std;

// 数据加载器
MatrixXd loadBinaryToEigen(const string& filename, int rows, int cols) {
    MatrixXd mat(rows, cols);
    ifstream file(filename, ios::binary);
    if (!file.is_open()) exit(1);
    Matrix<double, Dynamic, Dynamic, RowMajor> tempMap(rows, cols);
    file.read(reinterpret_cast<char*>(tempMap.data()), rows * cols * sizeof(double));
    mat = tempMap;
    file.close();
    return mat;
}

int main() {
    int N = 14541; // FB15K237 实体数
    int d = 384;   // 原始维度
    int kappa = 32; // AME 切片膨胀参数

    cout << "1. 加载 FB15K237 数据..." << endl;
    MatrixXd dataset = loadBinaryToEigen("/app/data/FB15K237_embeddings.bin", N, d);

    // 预生成 AME 检索阶段的矩阵，模拟真实的内存占用和计算流
    vector<MatrixXd> ame_matrices(kappa);
    vector<VectorXd> query_slices(kappa);
    for(int k = 0; k < kappa; ++k) {
        ame_matrices[k] = MatrixXd::Random(d, d);
        query_slices[k] = VectorXd::Random(d);
    }

    cout << "2. 执行 AME (O(\\kappa N d^2)) 硬件级并发检索..." << endl;
    auto start = chrono::high_resolution_clock::now();
    
    VectorXd final_scores = VectorXd::Zero(N);

    // 【关键优化1】关闭 Eigen 内部多线程，防止线程踩踏！
    setNbThreads(1);

    // 【关键优化2】使用 OpenMP 静态调度，将 14541 次检索均分给 32 线程
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double current_score = 0;
        // 提取当前实体向量
        VectorXd row_vec = dataset.row(i).transpose();
        
        // 严格执行 O(d^2) 的底层数学逻辑
        for (int k = 0; k < kappa; ++k) {
            current_score += (ame_matrices[k] * row_vec).dot(query_slices[k]);
        }
        final_scores(i) = current_score;
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "   -> 真实 AME 全量检索耗时: " 
         << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

    return 0;
}
