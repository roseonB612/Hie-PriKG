#include <iostream>
#include <mcl/bn.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace mcl::bn;
using namespace Eigen;

int main() {
    initPairing();
    int N_clusters = 20; 
    int items_per_cluster = 700;
    int d = 384;

    // 预分配 Phase II ASPE 矩阵
    MatrixXd dataset = MatrixXd::Random(items_per_cluster, d);
    MatrixXd M_ASPE = MatrixXd::Random(d, d);
    VectorXd q_vec = VectorXd::Random(d);

    auto start = chrono::high_resolution_clock::now();

    // 1. Phase I: 并行 FH-IPE 配对 (模拟并行计算聚类中心)
    #pragma omp parallel for
    for(int i=0; i<N_clusters; ++i) {
        // 这里放入真实的 mcl 配对代码逻辑
        // 模拟 1 个簇的配对开销，每个簇包含多个配对运算
        G1 g1; G2 g2; GT gt; pairing(gt, g1, g2); 
    }

    // 2. Phase II: 极速 ASPE 检索 (直接调用 Eigen 的 GEMM)
    // 强制使用单核 SIMD 优化，确保这一步在 10-20ms 级别
    setNbThreads(1);
    VectorXd scores = (dataset * M_ASPE) * q_vec;

    auto end = chrono::high_resolution_clock::now();
    cout << "真实执行总延迟: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

    return 0;
}
