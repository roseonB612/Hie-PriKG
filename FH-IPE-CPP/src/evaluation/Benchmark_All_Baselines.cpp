#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMat;

// ASPE 核心逻辑: O(N * d^2)
double run_aspe(const RowMat& data, int d) {
    MatrixXd M = MatrixXd::Random(d + 2, d + 2); // ASPE 增广维度 d+2
    VectorXd q = VectorXd::Random(d + 2);
    
    auto start = chrono::high_resolution_clock::now();
    // ASPE 核心计算 (直接映射)
    VectorXd scores = (data * M) * q; 
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

// AME 核心逻辑: O(kappa * N * d^2)
double run_ame(const RowMat& data, int d, int kappa) {
    MatrixXd M = MatrixXd::Random(d, d);
    VectorXd q = VectorXd::Random(d);
    
    auto start = chrono::high_resolution_clock::now();
    for(int i=0; i<kappa; ++i) {
        VectorXd scores = (data * M) * q;
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

int main() {
    struct Dataset { string name; string file; int N; int d; };
    vector<Dataset> dbs = {
        {"FB15K237", "data/FB15K237_embeddings.bin", 14541, 384},
        {"WN18RR",   "data/WN18RR_embeddings.bin",   40943, 384},
        {"PrimeKG",  "data/PrimeKG_embeddings.bin",  100000, 384}
    };

    cout << "--- 基准测试对比: ASPE vs AME ---" << endl;
    for(auto& db : dbs) {
        ifstream file(db.file, ios::binary);
        RowMat data(db.N, db.d);
        file.read(reinterpret_cast<char*>(data.data()), db.N * db.d * sizeof(double));
        
        double t_aspe = run_aspe(data, db.d);
        double t_ame = run_ame(data, db.d, 32);
        
        cout << "数据集: " << db.name << endl;
        cout << "  -> ASPE (Baseline 1) 耗时: " << t_aspe << " ms" << endl;
        cout << "  -> AME  (Baseline 2) 耗时: " << t_ame << " ms" << endl;
    }
    return 0;
}
