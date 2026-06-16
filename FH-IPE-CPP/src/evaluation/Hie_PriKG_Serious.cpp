#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <mcl/bn.hpp>
#include <omp.h>

using namespace std;
using namespace mcl::bn;
using namespace Eigen;

// 真实模拟：每个中心点需要 d 次 Pairing (对应 384 维向量的加密内积)
double run_phase1_real(int num_clusters, int d) {
    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for(int i = 0; i < num_clusters; ++i) {
        for(int j = 0; j < d; ++j) { // 核心：真实的 FH-IPE 维度开销
            G1 g1; G2 g2; GT gt;
            pairing(gt, g1, g2);
        }
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

// 真实模拟：对选定的簇进行完整矩阵检索 (例如 3 个簇 x 700 个实体)
double run_phase2_real(int num_items, int d) {
    MatrixXd data = MatrixXd::Random(num_items, d);
    MatrixXd M = MatrixXd::Random(d, d);
    VectorXd q = VectorXd::Random(d);
    
    auto start = chrono::high_resolution_clock::now();
    VectorXd scores = data * (M * q); 
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

int main() {
    initPairing();
    int d = 384;
    // 使用你真实的图谱规模参数
    struct KG { string name; int clusters; int items_in_search; };
    vector<KG> tasks = {
        {"FB15K237", 300, 2100}, // 假设过滤后需要检索 3 个簇
        {"WN18RR",   500, 2500},
        {"PrimeKG",  1200, 3000} 
    };

    cout << "=== Hie-PriKG 真实性能仿真 (还原 Pairing 维度压力) ===" << endl;
    for(auto& t : tasks) {
        double t1 = run_phase1_real(t.clusters, d);
        double t2 = run_phase2_real(t.items_in_search, d);
        cout << t.name << " | Phase I: " << t1 << "ms | Phase II: " << t2 << "ms | Total: " << (t1+t2) << "ms" << endl;
    }
    return 0;
}
