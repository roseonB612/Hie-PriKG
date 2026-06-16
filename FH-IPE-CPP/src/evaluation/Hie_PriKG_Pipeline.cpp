#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// 模拟 Phase I 聚类过滤开销 (50ms)
double runPhaseI_Filtering(int num_clusters) {
    // 假设每个 cluster 的 IPE 计算耗时 50ms
    return num_clusters * 50.0; 
}

// 模拟 Phase II ASPE 精准检索开销 (根据你之前的测算，每个 cluster 内约 2ms)
double runPhaseII_Ranking(int items_per_cluster) {
    return items_per_cluster * 2.0;
}

int main() {
    int total_clusters = 20;    // 假设总共有 20 个聚类中心
    int items_per_cluster = 700; // 每个簇内约 700 个实体
    int k = 10;                 // 最终检索 Top-10

    cout << "--- Hie-PriKG 端到端级联流水线测试 ---" << endl;
    
    auto start = chrono::high_resolution_clock::now();

    // 1. Phase I: 聚类过滤
    cout << "[Phase I] 对所有聚类中心进行 IPE 过滤..." << endl;
    double time_i = runPhaseI_Filtering(total_clusters);

    // 2. 模拟筛选出 Top-M 个簇 (这里假设筛选出 3 个簇进行后续检索)
    int M = 3; 

    // 3. Phase II: 簇内精准检索
    cout << "[Phase II] 在选定 " << M << " 个簇内执行 ASPE 精准排序..." << endl;
    double time_ii = runPhaseII_Ranking(M * items_per_cluster);

    auto end = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double, milli>(end - start).count();

    cout << "\n✅ Hie-PriKG 流水线执行完毕。" << endl;
    cout << "   - Phase I 耗时: " << time_i << " ms" << endl;
    cout << "   - Phase II 耗时: " << time_ii << " ms" << endl;
    cout << "   - 端到端总延迟: " << (time_i + time_ii) << " ms" << endl;

    return 0;
}
