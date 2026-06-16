#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// 使用显式的 RowMajor 布局，彻底消除对齐错误
typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMat;

double run_aspe(int N, int d, const string& filename) {
    ifstream file(filename, ios::binary);
    RowMat data(N, d);
    file.read(reinterpret_cast<char*>(data.data()), N * d * sizeof(double));
    
    // ASPE 核心：(N x d) * (d x d)
    MatrixXd M = MatrixXd::Random(d, d);
    VectorXd q = VectorXd::Random(d);
    
    auto start = chrono::high_resolution_clock::now();
    VectorXd scores = (data * M) * q;
    auto end = chrono::high_resolution_clock::now();
    
    return chrono::duration<double, milli>(end - start).count();
}

int main() {
    cout << "--- ASPE 基准耗时 (单核 SIMD 优化) ---" << endl;
    cout << "FB15K237: " << run_aspe(14541, 384, "data/FB15K237_embeddings.bin") << " ms" << endl;
    cout << "WN18RR:   " << run_aspe(40943, 384, "data/WN18RR_embeddings.bin")   << " ms" << endl;
    cout << "PrimeKG:  " << run_aspe(100000, 384, "data/PrimeKG_embeddings.bin")  << " ms" << endl;
    return 0;
}
