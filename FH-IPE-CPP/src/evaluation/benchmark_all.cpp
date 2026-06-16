#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>
#include <mcl/bn.hpp>
#include <omp.h>

using namespace std;
using namespace mcl::bn;
using namespace Eigen;

int get_cluster_count(const string& filename) {
    ifstream file(filename);
    string line;
    int count = 0;
    while (getline(file, line)) count++;
    return count;
}

double run_phase1_real(int num_clusters, int d) {
    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < num_clusters; ++i) {
        for (int j = 0; j < d; ++j) {
            G1 g1; G2 g2; GT gt;
            pairing(gt, g1, g2); 
        }
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

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
    
    struct KG { string name; string cluster_file; int items; };
    vector<KG> tasks = {
        {"FB15K237", "data/FB15K237_clusters.txt", 2100},
        {"WN18RR",   "data/WN18RR_clusters.txt",   2500},
        {"PrimeKG",  "data/PrimeKG_clusters.txt",  3000}
    };

    cout << "=== Hie-PriKG 性能基准测试 ===" << endl;
    for(const auto& t : tasks) {
        int real_c = get_cluster_count(t.cluster_file);
        cout << "\n[测试目标] " << t.name << " | 聚类数: " << real_c << endl;
        
        double t1 = run_phase1_real(real_c, d);
        double t2 = run_phase2_real(t.items, d);
        
        cout << "  Phase I (Pairing): " << t1 << " ms" << endl;
        cout << "  Phase II (ASPE):   " << t2 << " ms" << endl;
        cout << "  总延迟 (Total):     " << (t1 + t2) << " ms" << endl;
    }
    return 0;
}
