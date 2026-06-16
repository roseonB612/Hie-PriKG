#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <mcl/bn.hpp>

using namespace std;
using namespace mcl::bn;
using namespace Eigen;

double phase1(int C) {
    auto s = chrono::high_resolution_clock::now();
    for (int i = 0; i < C; ++i) {
        G1 g1; G2 g2; GT gt;
        pairing(gt, g1, g2);  // 每个簇心2次pairing，这里先测1次
    }
    auto e = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(e - s).count();
}

double phase2(int M, int Nc, int d) {
    MatrixXd data = MatrixXd::Random(M * Nc, d);
    VectorXd q = VectorXd::Random(d);
    auto s = chrono::high_resolution_clock::now();
    VectorXd scores = data * q;
    auto e = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(e - s).count();
}

int main() {
    initPairing();

    struct KG { string name; int C; int N; int M; };
    vector<KG> datasets = {
        {"FB15K237", 36,  14541, 4},
        {"WN18RR",   100, 40943, 4},
        {"PrimeKG",  300, 129375, 32}
    };

    cout << "=== 正确参数下的Hie-PriKG延迟 ===" << endl;
    for (auto& kg : datasets) {
        int Nc = kg.N / kg.C;
        double t1 = phase1(kg.C) * 2;  // 每个簇心2次pairing
        double t2 = phase2(kg.M, Nc, 384);
        cout << kg.name << ": Phase I=" << t1
             << "ms, Phase II=" << t2
             << "ms, Total=" << (t1+t2) << "ms" << endl;
        cout << "  (C=" << kg.C << ", M=" << kg.M
             << ", Nc=" << Nc << ")" << endl;
    }
    return 0;
}
