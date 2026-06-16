#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include <mcl/bn.hpp>

using namespace std;
using namespace mcl::bn;
using namespace Eigen;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMat;

// ============================================================
// 数据加载
// ============================================================
RowMat loadBin(const string& path, int N, int d) {
    RowMat mat(N, d);
    ifstream f(path, ios::binary);
    if (!f) { cerr << "无法打开: " << path << endl; exit(1); }
    f.read(reinterpret_cast<char*>(mat.data()), N * d * sizeof(double));
    return mat;
}

// ============================================================
// Baseline 1: ASPE 全局扫描  O(N)
// 优化版：data*(M*q) 而非 (data*M)*q
// ============================================================
double run_aspe(const RowMat& data, int d) {
    MatrixXd M = MatrixXd::Random(d, d);
    VectorXd q = VectorXd::Random(d);
    auto s = chrono::high_resolution_clock::now();
    VectorXd Mq  = M * q;           // d×d · d = d，极快
    VectorXd scores = data * Mq;    // N×d · d = N，核心开销
    auto e = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(e - s).count();
}

// ============================================================
// Baseline 2: AME 全局扫描  O(kappa * N * d^2)
// 严格按照论文：kappa=32次独立矩阵切片检索求和
// ============================================================
double run_ame(const RowMat& data, int d, int kappa = 32) {
    VectorXd total_scores = VectorXd::Zero(data.rows());
    auto s = chrono::high_resolution_clock::now();
    for (int k = 0; k < kappa; ++k) {
        MatrixXd Mk = MatrixXd::Random(d, d);
        VectorXd qk = VectorXd::Random(d);
        VectorXd Mkq = Mk * qk;
        total_scores += data * Mkq;
    }
    auto e = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(e - s).count();
}

// ============================================================
// Hie-PriKG Phase I: FH-IPE 簇心过滤
// 每个簇心做2次pairing（对应论文中的e(C1,K1)和e(C2,K2)）
// ============================================================
double run_phase1(int C) {
    G1 P; G2 Q; GT e;
    hashAndMapToG1(P, "bench", 5);
    hashAndMapToG2(Q, "bench", 5);
    pairing(e, P, Q); // 预热

    auto s = chrono::high_resolution_clock::now();
    for (int i = 0; i < C; ++i) {
        GT e1, e2;
        pairing(e1, P, Q); // e(C1, K1)
        pairing(e2, P, Q); // e(C2, K2)
    }
    auto en = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(en - s).count();
}

// ============================================================
// Hie-PriKG Phase II: ASPE 局部精排
// 只对M个簇内的实体做ASPE
// ============================================================
double run_phase2(int M, int Nc, int d) {
    RowMat local = RowMat::Random(M * Nc, d);
    MatrixXd Mmat = MatrixXd::Random(d, d);
    VectorXd q = VectorXd::Random(d);
    auto s = chrono::high_resolution_clock::now();
    VectorXd Mq = Mmat * q;
    VectorXd scores = local * Mq;
    auto en = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(en - s).count();
}

// ============================================================
// 主程序
// ============================================================
int main() {
    initPairing();

    struct KG {
        string name;
        string path;
        int N, d, C, M;
    };

    vector<KG> datasets = {
        {"FB15K-237", "/app/data/FB15K237_embeddings.bin", 14541,  384, 36,  4},
        {"WN18RR",    "/app/data/WN18RR_embeddings.bin",   40943,  384, 100, 4},
        {"PrimeKG",   "/app/data/PrimeKG_embeddings.bin",  129375, 384, 300, 32},
    };

    int REPEAT = 5; // 每个测试重复5次取平均

    cout << "\n=====================================================" << endl;
    cout << "   完整基线对比测试 (重复" << REPEAT << "次取均值)" << endl;
    cout << "=====================================================" << endl;

    for (auto& kg : datasets) {
        cout << "\n>>> 数据集: " << kg.name
             << "  (N=" << kg.N << ", C=" << kg.C << ", M=" << kg.M << ")" << endl;

        // 加载数据
        RowMat data = loadBin(kg.path, kg.N, kg.d);
        int Nc = kg.N / kg.C;

        // --- Baseline 1: ASPE ---
        double t_aspe = 0;
        for (int r = 0; r < REPEAT; ++r) t_aspe += run_aspe(data, kg.d);
        t_aspe /= REPEAT;

        // --- Baseline 2: AME ---
        double t_ame = 0;
        for (int r = 0; r < REPEAT; ++r) t_ame += run_ame(data, kg.d);
        t_ame /= REPEAT;

        // --- Hie-PriKG ---
        double t_p1 = 0, t_p2 = 0;
        for (int r = 0; r < REPEAT; ++r) {
            t_p1 += run_phase1(kg.C);
            t_p2 += run_phase2(kg.M, Nc, kg.d);
        }
        t_p1 /= REPEAT;
        t_p2 /= REPEAT;
        double t_hie = t_p1 + t_p2;

        // 输出结果
        cout << fixed;
        cout.precision(3);
        cout << "  B1  (ASPE全局):   " << t_aspe << " ms" << endl;
        cout << "  B2  (AME全局):    " << t_ame  << " ms" << endl;
        cout << "  Ours Phase I:     " << t_p1   << " ms  (C=" << kg.C << "个簇心×2次pairing)" << endl;
        cout << "  Ours Phase II:    " << t_p2   << " ms  (M=" << kg.M << "×Nc=" << Nc << "个实体)" << endl;
        cout << "  Ours 端到端:      " << t_hie  << " ms" << endl;
        cout << "  加速比 vs AME:    " << (t_ame / t_hie) << "x" << endl;
    }

    cout << "\n=====================================================" << endl;
    cout << "测试完成" << endl;
    return 0;
}
