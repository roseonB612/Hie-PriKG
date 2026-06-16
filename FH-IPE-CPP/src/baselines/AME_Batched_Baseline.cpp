#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <omp.h>

using namespace Eigen;
using namespace std;

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
    int N = 14541;
    int d = 384;
    int kappa = 32;

    cout << "1. 加载数据..." << endl;
    MatrixXd dataset = loadBinaryToEigen("/app/data/FB15K237_embeddings.bin", N, d);

    vector<MatrixXd> ame_matrices(kappa);
    vector<VectorXd> query_slices(kappa);
    for(int i = 0; i < kappa; ++i) {
        ame_matrices[i] = MatrixXd::Random(d, d);
        query_slices[i] = VectorXd::Random(d);
    }

    cout << "2. 执行 AME 并行批处理检索 (BLAS Level 3 + 32-Thread OpenMP)..." << endl;
    auto start = chrono::high_resolution_clock::now();
    
    VectorXd final_scores = VectorXd::Zero(N);

    // 跨 32 线程的矩阵分块批处理
    #pragma omp parallel for
    for (int k = 0; k < kappa; ++k) {
        // (N x d) * (d x d) -> N x d 批量计算，极大提升 CPU L1/L2 缓存命中率
        VectorXd slice_scores = (dataset * ame_matrices[k]) * query_slices[k];
        
        #pragma omp critical
        final_scores += slice_scores;
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "   -> AME 批处理检索耗时: " 
         << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

    return 0;
}
