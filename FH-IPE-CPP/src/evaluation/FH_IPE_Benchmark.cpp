#include <iostream>
#include <mcl/bn.hpp>
#include <chrono>

using namespace std;
using namespace mcl::bn;

int main() {
    initPairing();
    int d = 384;
    int dim = 2 * d + 4;
    
    // 预分配向量 (模拟论文中的加密向量)
    vector<Fr> mu(dim), q(dim);
    for(int i=0; i<dim; ++i) { mu[i].setByCSPRNG(); q[i].setByCSPRNG(); }

    cout << "开始 FH-IPE 全链路耗时测试 (dim=772)..." << endl;
    auto start = chrono::high_resolution_clock::now();

    // 1. 模拟 Encrypt: 映射到 G1
    vector<G1> ct(dim);
    G1 g1; g1.clear();
    for(int i=0; i<dim; ++i) G1::mul(ct[i], g1, mu[i]);

    // 2. 模拟 KeyGen: 映射到 G2
    vector<G2> tk(dim);
    G2 g2; g2.clear();
    for(int i=0; i<dim; ++i) G2::mul(tk[i], g2, q[i]);

    // 3. Eval: 双线性配对计算核心 (Pairing)
    // 这是最消耗 CPU 的部分: Sum(e(ct_i, tk_i))
    GT result; result.clear();
    GT temp;
    for(int i=0; i<dim; ++i) {
        pairing(temp, ct[i], tk[i]);
        result *= temp;
    }

    auto end = chrono::high_resolution_clock::now();
    double total_ms = chrono::duration<double, milli>(end - start).count();

    cout << "✅ FH-IPE 配对计算完成。" << endl;
    cout << "   -> FH-IPE 总耗时: " << total_ms << " ms" << endl;
    cout << "   -> 单维配对平均耗时: " << total_ms / dim << " ms" << endl;

    return 0;
}
