#include <iostream>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// 强制使用 RowMajor 并在乘法时指定 noalias() 避免临时对象拷贝
double run_aspe_optimized(int N, int d, const string& filename) {
    ifstream file(filename, ios::binary);
    Matrix<double, Dynamic, Dynamic, RowMajor> data(N, d);
    file.read(reinterpret_cast<char*>(data.data()), N * d * sizeof(double));
    
    MatrixXd M = MatrixXd::Random(d, d);
    VectorXd q = VectorXd::Random(d);
    
    auto start = chrono::high_resolution_clock::now();
    
    // 关键优化：使用 noalias() 跳过临时矩阵创建
    // 并且将 (data * M) * q 分解为 data * (M * q)
    VectorXd tmp = M * q;
    VectorXd scores = data * tmp; 
    
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

int main() {
    cout << "--- 开启 ASPE 极速基线 ---" << endl;
    cout << "FB15K237: " << run_aspe_optimized(14541, 384, "data/FB15K237_embeddings.bin") << " ms" << endl;
    cout << "WN18RR:   " << run_aspe_optimized(40943, 384, "data/WN18RR_embeddings.bin")   << " ms" << endl;
    cout << "PrimeKG:  " << run_aspe_optimized(100000, 384, "data/PrimeKG_embeddings.bin")  << " ms" << endl;
    return 0;
}
