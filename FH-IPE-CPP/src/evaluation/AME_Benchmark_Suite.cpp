#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMat;

// 通用的基准测试函数
double run_benchmark(const string& filename, int N, int d) {
    ifstream file(filename, ios::binary);
    RowMat data(N, d);
    file.read(reinterpret_cast<char*>(data.data()), N * d * sizeof(double));
    
    MatrixXd M = MatrixXd::Random(d, d);
    VectorXd q = VectorXd::Random(d);

    auto start = chrono::high_resolution_clock::now();
    // GEMM 基准测试 (AME 模拟)
    VectorXd scores = (data * M) * q;
    auto end = chrono::high_resolution_clock::now();

    return chrono::duration<double, milli>(end - start).count();
}

int main() {
    // 数据集配置表 (你可以根据实际文件调整 N 和 d)
    struct Dataset { string name; string file; int N; int d; };
    vector<Dataset> dbs = {
        {"FB15K237", "data/FB15K237_embeddings.bin", 14541, 384},
        {"WN18RR",   "data/WN18RR_embeddings.bin",   40943, 384},
        {"PrimeKG",  "data/PrimeKG_embeddings.bin",  100000, 384} // 假设 PrimeKG 大约 10 万节点
    };

    cout << "--- 开启全量基线大统考 ---" << endl;
    for(auto& db : dbs) {
        double time = run_benchmark(db.file, db.N, db.d);
        cout << "数据集: " << db.name << " | 基线耗时: " << time << " ms" << endl;
    }
    return 0;
}
