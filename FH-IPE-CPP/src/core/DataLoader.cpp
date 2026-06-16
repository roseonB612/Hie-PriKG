#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>

using namespace Eigen;
using namespace std;

// 读取纯二进制文件并映射到 Eigen 矩阵
MatrixXd loadBinaryToEigen(const string& filename, int rows, int cols) {
    MatrixXd mat(rows, cols);
    ifstream file(filename, ios::binary);
    
    if (!file.is_open()) {
        cerr << "❌ 错误: 无法打开文件 " << filename << endl;
        exit(1);
    }

    // 将二进制流直接读入 Eigen 矩阵的内存块
    // 注意：Eigen 默认是列优先 (Column-Major)，而 NumPy 保存的默认是行优先 (Row-Major)
    // 所以我们需要一个临时的 RowMajor 矩阵来接收数据，然后再转给默认的 MatrixXd
    Matrix<double, Dynamic, Dynamic, RowMajor> tempMap(rows, cols);
    if(file.read(reinterpret_cast<char*>(tempMap.data()), rows * cols * sizeof(double))) {
        mat = tempMap;
    } else {
        cerr << "❌ 错误: 读取文件数据失败或数据量不匹配" << endl;
        exit(1);
    }
    
    file.close();
    return mat;
}

int main() {
    // 针对 FB15K237 数据集的参数
    string binFile = "/app/data/FB15K237_embeddings.bin";
    int numEntities = 14541;
    int dimension = 384;

    cout << "开始加载 FB15K237 嵌入数据..." << endl;
    
    auto start = chrono::high_resolution_clock::now();
    MatrixXd dataset = loadBinaryToEigen(binFile, numEntities, dimension);
    auto end = chrono::high_resolution_clock::now();
    
    double duration = chrono::duration<double, std::milli>(end - start).count();

    cout << "✅ 加载成功！" << endl;
    cout << "矩阵尺寸: " << dataset.rows() << " 行 x " << dataset.cols() << " 列" << endl;
    cout << "加载耗时: " << duration << " 毫秒" << endl;
    cout << "预览第一行前 5 个数值:\n" << dataset.row(0).head(5) << endl;

    return 0;
}
