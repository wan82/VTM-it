#ifndef W_CU_PROCESS_H
#define W_CU_PROCESS_H

#include <map>
#include "EncLib.h"
#include <utility> // for std::pair


const size_t POOL_SIZE = 2;

// Sobel 卷积核 // Sobel convolution kernels
const int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int Gy[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};

const std::map<int, int> POC_Offsets = {
    {0, -32},
    {16, -16},
    {8,  -8},
    {4,  -4},
    {2,  -2},
    {1,  -1},
    {3,  -1},
    {6,  -2},
    {5,  -1},
    {7,  -1},
    {12, -4},
    {10, -2},
    {9,  -1},
    {11, -1},
    {14, -2},
    {13, -1},
    {15, -1},
    {24, -8},
    {20, -4},
    {18, -2},
    {17, -1},
    {19, -1},
    {22, -2},
    {21, -1},
    {23, -1},
    {28, -4},
    {26, -2},
    {25, -1},
    {27, -1},
    {30, -2},
    {29, -1},
    {31, -1}
};

template<typename Pel>

// 将二维向量转换为一维向量 // Flatten a 2D vector to a 1D vector
std::vector<double> flattenToDouble(const std::vector<std::vector<Pel>>& matrix);

// 将10bit转换为8bit的函数 Function to convert 10bit to 8bit
std::vector<std::vector<uint8_t>> convertTo8bit(const std::vector<std::vector<Pel>>& input);

//打印cu中的最大值 // Print the maximum value in a CU matrix
void printMaxCuValue(const std::vector<std::vector<Pel>>& matrix);

// 将 PelBuf 转换为二维向量 // Convert PelBuf to a 2D vector
std::vector<std::vector<Pel>> convertPelBufToIntMatrix(const PelBuf& cuMatrix);

std::vector<std::vector<Pel>> convertPelBufToIntMatrix(const PelBuf& cuMatrix, int cuSize, int frameWidth, int frameHeight, int32_t posX, int32_t posY);

// 最大池化函数 // Max pooling function
std::vector<std::vector<Pel>> maxPooling(const std::vector<std::vector<Pel>>& input);

// Sobel 滤波函数 // Sobel filter function
std::vector<std::vector<Pel>> sobelFilter(const std::vector<std::vector<Pel>>& input);

// 打印二维向量 // Print a 2D vector
void printVector2D(const std::vector<std::vector<Pel>>& vec);

// 计算两个矩阵的差 // Subtract two matrices
std::vector<std::vector<Pel>> subtractMatrices(const std::vector<std::vector<Pel>>& matrix1, 
    const std::vector<std::vector<Pel>>& matrix2);

// 计算均值和标准差 // Calculate mean and standard deviation
std::pair<double, double> calculate_mean_std(const std::vector<std::vector<Pel>>& matrix);

// 单独计算均值 // Calculate mean separately
double calculate_mean(const std::vector<std::vector<Pel>>& matrix);

// 单独计算std // Calculate standard deviation separately
double calculate_std(const std::vector<std::vector<Pel>>& matrix);

// 计算 skewness // Calculate skewness
double calculate_skewness(const std::vector<std::vector<Pel>>& matrix);

// 计算 kurtosis // Calculate kurtosis
double calculate_kurtosis(const std::vector<std::vector<Pel>>& matrix);

// 根据res帧的权重来缩放res值 // Scale resolution value based on resolution frame weight
double scale_res_value(double value, int time_res_L0);

// 声明 calculate_mean 模板函数 // Declare calculate_mean template function
template<typename T>
double calculate_mean(const std::vector<T>& vec);

// 计算 calculate_levene  // Calculate Levene's test statistic
double calculate_levene(const std::vector<std::vector<Pel>>& matrix);

// 计算 Kruskal-Wallis 统计量  // Calculate Kruskal-Wallis statistic
double calculate_kruskal_w(const std::vector<std::vector<Pel>>& matrix);

// 计算 levene 和 Kruskal_Waills  // Calculate both Levene's and Kruskal-Wallis statistics
std::pair<double, double> calculate_levene_kruskal(const std::vector<std::vector<Pel>>& matrix);

// 计算 qp_value // Calculate qp_value based on POC and QP
int get_qp_value(int poc, int qp);

// 在原有数值的基础上乘 qp_weight 来缩放数值 // Scale the value based on qp_weight
double multi_qp_weight(double data, double qp_value); //TODO::wan

template<typename... Args>
std::vector<double> multi_qpweight_pack(double qp_value, Args... args) {
    const double INV_51 = 0.01960784; // 1/51
    const double beta = 0.5;
    double qp_weight = std::exp(-beta * qp_value * INV_51);

    std::vector<double> vec = { (args * qp_weight)... }; // 每个值都乘上 qp_weight// Each value is multiplied by qp_weight
    vec.push_back(qp_value); // 把原始 qp_value 也加进去 // Add the original qp_value to the vector

    return vec;
}
template<typename... Args>
std::vector<double> multi_qpweight_pack_nonqp(double qp_value, Args... args) {
    const double INV_51 = 0.01960784; // 1/51
    const double beta = 0.5;
    double qp_weight = std::exp(-beta * qp_value * INV_51);

    std::vector<double> vec = { (args * qp_weight)... }; // 每个值都乘上 qp_weight // Each value is multiplied by qp_weight
    return vec;
}

#endif 

