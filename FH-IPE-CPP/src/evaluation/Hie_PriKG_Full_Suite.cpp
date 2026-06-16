#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <mcl/bn.hpp>
#include <omp.h>

using namespace std;
using namespace mcl::bn;
using namespace Eigen;

// 真实测算 Pairing 时间 (之前的逻辑)
double run_phase1_ipe(int num_clusters) {
    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for(int i = 0; i < num_clusters; ++i) {
        G1 g1; G2 g2; GT gt;
        pairing(gt, g1, g2); 
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

// 真实测算 ASPE 时间 (优化过的 GEMM)
double run_phase2_aspe(int num_items, int d) {
    MatrixXd data = MatrixXd::Random(num_items, d);
    MatrixXd M = MatrixXd::Random(d, d);
    VectorXd q = VectorXd::Random(d);
    
    auto start = chrono::high_resolution_clock::now();
    VectorXd scores = data * (M * q); // 优化后的 O(Nd)
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

int main() {
    initPairing();
    struct KG { string name; int clusters; int size_per_cluster; };
    vector<KG> tasks = {
        {"FB15K237", 20, 700},
        {"WN18RR",   50, 800},
        {"PrimeKG",  150, 1000} // 根据 PrimeKG 大规模特性设置
    };

    cout << "=== Hie-PriKG 跨数据集压力测试 ===" << endl;
    for(auto& t : tasks) {
        cout << "\n--- 测试图谱: " << t.name << " ---" << endl;
        double t1 = run_phase1_ipe(t.clusters);
        double t2 = run_phase2_aspe(t.size_per_cluster * 3, 384); // 假设筛选出 3 个簇
        cout << "总延迟: " << (t1 + t2) << " ms (Phase I: " << t1 << "ms, Phase II: " << t2 << "ms)" << endl;
    }
    return 0;
}
