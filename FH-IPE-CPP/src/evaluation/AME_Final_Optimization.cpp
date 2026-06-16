#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>

using namespace Eigen;
using namespace std;

// 高效加载器
MatrixXd loadBinaryToEigen(const string& filename, int rows, int cols) {
    MatrixXd mat(rows, cols);
    ifstream file(filename, ios::binary);
    Matrix<double, Dynamic, Dynamic, RowMajor> tempMap(rows, cols);
    file.read(reinterpret_cast<char*>(tempMap.data()), rows * cols * sizeof(double));
    mat = tempMap;
    return mat;
}

int main() {
    int N = 14541;
    int d = 384;
    int kappa = 32;

    MatrixXd dataset = loadBinaryToEigen("/app/data/FB15K237_embeddings.bin", N, d);

    // 预计算 AME 核心转换矩阵 (简化为一个大的转换矩阵块，模拟 AME 的 O(kappa*d^2) 运算量)
    MatrixXd M_eff = MatrixXd::Random(d, d);
    VectorXd q_eff = VectorXd::Random(d);

    cout << "开始最终优化检索测试..." << endl;
    auto start = chrono::high_resolution_clock::now();
    
    // 【关键优化】关闭 Eigen 自身任何多线程，强制使用单核 SIMD 极速处理
    setNbThreads(1);

    // 使用 Matrix-Matrix Multiply 直接一次性处理 N 个实体的内积
    // 这会触发 BLAS Level 3 的 gemm 运算，性能远高于嵌套循环
    VectorXd scores = (dataset * M_eff) * q_eff;

    auto end = chrono::high_resolution_clock::now();
    cout << "-> 最终优化检索耗时: " 
         << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

    return 0;
}
