#include "w_cuProcess.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <string>

std::vector<double> flattenToDouble(const std::vector<std::vector<Pel>>& matrix) {
    std::vector<double> flat;
    size_t totalSize = 0;
    for (const auto& row : matrix) {
        totalSize += row.size();
    }

    flat.reserve(totalSize); // 提前分配内存，避免重复 realloc

    for (const auto& row : matrix) {
        std::transform(row.begin(), row.end(), std::back_inserter(flat),
                       [](const Pel& val) { return static_cast<double>(val); });
    }

    return flat;
}

// 将10bit转换为8bit的函数 Function to convert 10bit to 8bit
std::vector<std::vector<uint8_t>> convertTo8bit(const std::vector<std::vector<Pel>>& input) {
    size_t height_size = input.size();
    size_t width_size  = input[0].size();

    int height = static_cast<int>(height_size);
    int width  = static_cast<int>(width_size);


    std::vector<std::vector<uint8_t>> output(height, std::vector<uint8_t>(width));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // 将10bit的值缩放到8bit范围
            output[y][x] = static_cast<uint8_t>((input[y][x] * 255 + 511) / 1023);
            // +511 是为了四舍五入而不是直接截断
        }
    }

    return output;
}


void printMaxCuValue(const std::vector<std::vector<Pel>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) {
        std::cout << "Matrix is empty" << std::endl;
        return;
    }

    Pel maxVal = matrix[0][0]; // 初始化为第一个元素
    for (const auto& row : matrix) {
        auto rowMax = *std::max_element(row.begin(), row.end());
        maxVal = std::max(maxVal, rowMax);
    }

    std::cout << "Max value in matrix: " << maxVal << std::endl;
}

// 将 PelBuf 转换为 int 矩阵的函数 Function to convert a PelBuf to an int matrix
std::vector<std::vector<Pel>> convertPelBufToIntMatrix(const PelBuf& cuMatrix) {
    int width = cuMatrix.width;
    int height = cuMatrix.height;

    if (width <= 0 || height <= 0) {
        return std::vector<std::vector<Pel>>(); // 修复(2.2)：尺寸无效返回空矩阵，不抛异常
    }
    std::vector<std::vector<Pel>> intMatrix(height, std::vector<Pel>(width));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            intMatrix[y][x] = cuMatrix.at(x, y);
        }
    }

    return intMatrix;
}

// 将 PelBuf 转换为 int 矩阵的函数 //在边缘填充后要消耗大量的时间 Function to convert PelBuf to int matrix //It takes a lot of time after filling the edges
std::vector<std::vector<Pel>> convertPelBufToIntMatrix(const PelBuf& cuMatrix, int cuSize, int frameWidth, int frameHeight, int32_t posX, int32_t posY) {
    int width = cuMatrix.width;
    int height = cuMatrix.height;
    int cuX = (int)posX; // CU 在帧中的 X 位置
    int cuY = (int)posY; // CU 在帧中的 Y 位置

    // 检查无效尺寸
    if (width <= 0 || height <= 0) {
        return std::vector<std::vector<Pel>>(cuSize, std::vector<Pel>(cuSize, 0)); // 修复(2.2)：返回零矩阵，不抛异常
    }

    // 检查 CU 是否超出帧边界，裁剪到帧内 Check if the CU exceeds the frame boundary and crop it to within the frame
    width = std::min(width, frameWidth - cuX);
    height = std::min(height, frameHeight - cuY);
    if (width <= 0 || height <= 0) {
        return std::vector<std::vector<Pel>>(cuSize, std::vector<Pel>(cuSize, 0)); // 修复(2.2)：超界返回零矩阵，不抛异常
    }

    // 检查是否符合 cuSize 且方形
    bool needsPadding = (width != cuSize || height != cuSize || width != height);
    int outputSize = cuSize; // 输出矩阵尺寸固定为 cuSize × cuSize

    // 分配输出矩阵
    std::vector<std::vector<Pel>> intMatrix(outputSize, std::vector<Pel>(outputSize, 0));

    // 填充有效像素 Fill valid pixels
    for (int y = 0; y < outputSize; y++) {
        for (int x = 0; x < outputSize; x++) {
            if (y < height && x < width && cuY + y < frameHeight && cuX + x < frameWidth) {
                intMatrix[y][x] = cuMatrix.at(x, y); // 读取有效像素
            } else {
                intMatrix[y][x] = 0; // 填充 0（越界或不符合 cuSize）
            }
        }
    }
    // // 日志警告（调试用）
    // if (needsPadding) {
    //     std::cerr << "Warning: Padded CU to " << outputSize << "x" << outputSize
    //               << " from " << width << "x" << height << " at (" << cuX << "," << cuY << ")" << std::endl;
    // }
    return intMatrix;
}

// 最大池化函数 Max pooling function
std::vector<std::vector<Pel>> maxPooling(const std::vector<std::vector<Pel>>& input) {
    if (input.empty() || input[0].empty()) {
        return {};
    }

    const size_t inputHeight = input.size();
    const size_t inputWidth = input[0].size();
    const size_t outputHeight = (inputHeight + POOL_SIZE - 1) / POOL_SIZE;
    const size_t outputWidth = (inputWidth + POOL_SIZE - 1) / POOL_SIZE;

    std::vector<std::vector<Pel>> output(outputHeight, std::vector<Pel>(outputWidth));

    for (size_t i = 0; i < inputHeight; i += POOL_SIZE) {
        size_t outRow = i / POOL_SIZE;
        for (size_t j = 0; j < inputWidth; j += POOL_SIZE) {
            size_t outCol = j / POOL_SIZE;
            Pel maxVal = input[i][j];

            size_t poolHeight = std::min(POOL_SIZE, inputHeight - i);
            for (size_t m = 0; m < poolHeight; ++m) {
                auto start = input[i + m].begin() + j;
                auto end = start + std::min(POOL_SIZE, inputWidth - j);
                maxVal = std::max(maxVal, *std::max_element(start, end));
            }
            output[outRow][outCol] = maxVal;
        }
    }

    return output;
}

// Sobel 滤波函数
std::vector<std::vector<Pel>> sobelFilter(const std::vector<std::vector<Pel>>& input) {
    if (input.empty() || input[0].empty()) {
        return {};
    }

    const size_t height = input.size();
    const size_t width = input[0].size();

    std::vector<std::vector<Pel>> output(height, std::vector<Pel>(width, 0));
    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t x = 1; x < width - 1; ++x) {
            Pel sumX = 0;
            Pel sumY = 0;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    sumX += input[y + ky][x + kx] * Gx[ky + 1][kx + 1];
                    sumY += input[y + ky][x + kx] * Gy[ky + 1][kx + 1];
                }
            }

            output[y][x] = std::sqrt(sumX * sumX + sumY * sumY);
        }
    }

    return output;
}

void printVector2D(const std::vector<std::vector<Pel>>& vec) {
    // 检查是否为空
    if (vec.empty()) {
        std::cout << "Vector is empty!" << std::endl;
        return;
    }

    // 遍历每一行
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "Row " << i << ": ";
        
        // 遍历每一列
        if (vec[i].empty()) {
            std::cout << "empty";
        } else {
            for (size_t j = 0; j < vec[i].size(); ++j) {
                std::cout << vec[i][j];
                // 如果不是最后一个元素，添加分隔符
                if (j < vec[i].size() - 1) {
                    std::cout << ", ";
                }
            }
        }
        std::cout << std::endl;
    }
}

// 矩阵减法函数 Matrix Subtraction Function
std::vector<std::vector<Pel>> subtractMatrices(const std::vector<std::vector<Pel>>& matrix1, 
    const std::vector<std::vector<Pel>>& matrix2) {
    // 检查输入矩阵是否为空
    if (matrix1.empty() || matrix2.empty()) {
    std::cerr << "Error: One or both input matrices are empty!" << std::endl;
    return std::vector<std::vector<Pel>>();
    }

    // 获取矩阵尺寸
    size_t rows1 = matrix1.size();
    size_t cols1 = matrix1[0].size();
    size_t rows2 = matrix2.size();
    size_t cols2 = matrix2[0].size();

    // 检查矩阵尺寸是否匹配
    if (rows1 != rows2 || cols1 != cols2) {
    std::cerr << "Error: Matrix dimensions do not match! " 
    << "matrix1: " << rows1 << "x" << cols1 
    << ", matrix2: " << rows2 << "x" << cols2 << std::endl;
    return std::vector<std::vector<Pel>>();
    }

    // 初始化结果矩阵
    std::vector<std::vector<Pel>> result(rows1, std::vector<Pel>(cols1));

    // 逐元素相减
    for (size_t i = 0; i < rows1; i++) {
    for (size_t j = 0; j < cols1; j++) {
    result[i][j] = matrix1[i][j] - matrix2[i][j];
    }
    }

    return result;
}

// 计算均值和标准差的函数 Function to calculate mean and standard deviation
std::pair<double, double> calculate_mean_std(const std::vector<std::vector<Pel>>& matrix) {
    if (matrix.empty()) {
        return {0.0, 0.0}; // 返回默认值
    }

    double sum = 0.0;
    double sumSq = 0.0;
    int count = 0;

    for (const auto& row : matrix) {
        if (row.empty()) {
            continue; // 跳过空行
        }
        for (const auto& val : row) {
            sum += val;
            sumSq += val * val;
            count++;
        }
    }

    if (count == 0) {
        return {0.0, 0.0}; // 没有有效元素，返回默认值
    }

    double mean = sum / count;
    double variance = (sumSq / count) - (mean * mean);
    double stddev = std::sqrt(variance);

    return {mean, stddev};
}

// 单独计算mean Calculate mean separately
double calculate_mean(const std::vector<std::vector<Pel>>& matrix) {
    if (matrix.empty()) {
        return 0.0; // 空矩阵返回 0.0
    }

    double sum = 0.0;
    size_t count = 0;

    for (const auto& row : matrix) {
        if (row.empty()) {
            continue; // 跳过空行
        }
        sum += std::accumulate(row.begin(), row.end(), 0.0);
        count += row.size();
    }

    if (count == 0) {
        return 0.0; // 无有效元素返回 0.0
    }

    return sum / count;
}

// 单独计算std
double calculate_std(const std::vector<std::vector<Pel>>& matrix) {
    if (matrix.empty()) {
        return 0.0; // 空矩阵返回 0.0
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    size_t count = 0;

    for (const auto& row : matrix) {
        if (row.empty()) {
            continue; // 跳过空行
        }
        for (const auto& val : row) {
            sum += val;
            sum_sq += static_cast<double>(val) * val;
            count++;
        }
    }

    if (count == 0) {
        return 0.0; // 无有效元素返回 0.0
    }

    double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    return std::sqrt(variance < 0.0 ? 0.0 : variance); // 防止浮点误差导致负方差
}

//计算skewness
double calculate_skewness(const std::vector<std::vector<Pel>>& matrix) {
    auto values = flattenToDouble(matrix);
    size_t N = values.size();
    if (N < 2) return 0.0; // 修复(2.2)：数据不足返回0，不抛异常

    double mean = std::accumulate(values.begin(), values.end(), 0.0) / N;

    double m2 = 0.0, m3 = 0.0;
    for (auto x : values) {
        double d = x - mean;
        m2 += d * d;
        m3 += d * d * d;
    }

    m2 /= N;
    m3 /= N;

    // if (m2 == 0.0) throw std::runtime_error("Variance is zero.");
    if (m2 == 0.0) return 0.0; // 如果方差为零，返回零

    return m3 / std::pow(std::sqrt(m2), 3);
}

//计算kurtosis
double calculate_kurtosis(const std::vector<std::vector<Pel>>& matrix) {
    auto values = flattenToDouble(matrix);
    size_t N = values.size();
    if (N < 2) return 0.0; // 修复(2.2)：数据不足返回0，不抛异常

    double mean = std::accumulate(values.begin(), values.end(), 0.0) / N;

    double m2 = 0.0, m4 = 0.0;
    for (auto x : values) {
        double d = x - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }

    m2 /= N;
    m4 /= N;

    // if (m2 == 0.0) throw std::runtime_error("Variance is zero.");
    if (m2 == 0.0) return 0.0; // 如果方差为零，返回零

    return m4 / (m2 * m2);
}



// 计算res缩放 Calculate res scaling
double scale_res_value(double value, int time_res_L0){
    //对time_res_L0取绝对值
    int abs_time_res_L0 = std::abs(time_res_L0);
    double alpha = 0.1;

    double scaled_weight = std::exp(-alpha * abs_time_res_L0);

    double scale_value = value * scaled_weight;
    return scale_value;
}

// 计算一维向量的均值 Calculates the mean of a one-dimensional vector
template<typename T>
double calculate_mean(const std::vector<T>& vec) {
    if (vec.empty()) return 0.0;
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

// 计算 Levene 统计量
double calculate_levene(const std::vector<std::vector<Pel>>& matrix) {
    // 检查是否为方阵
    size_t size = matrix.size();
    if (size == 0 || matrix[0].size() != size) {
        return 0.0; // 修复(2.2)：非方阵返回0，不抛异常
    }

    // 子 CU 大小为边长的一半
    size_t sub_cu_size = size / 2;
    if (sub_cu_size == 0) {
        return 0.0; // 修改：矩阵太小返回 0.0
    }

    // 分割成 4 个子 CU 并展平
    std::vector<std::vector<double>> sub_cus(4);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            std::vector<double>& sub_cu = sub_cus[i * 2 + j];
            sub_cu.reserve(sub_cu_size * sub_cu_size); // 预分配空间
            for (size_t y = i * sub_cu_size; y < (i + 1) * sub_cu_size; ++y) {
                for (size_t x = j * sub_cu_size; x < (j + 1) * sub_cu_size; ++x) {
                    sub_cu.push_back(static_cast<double>(matrix[y][x]));
                }
            }
        }
    }

    // 检查是否有空子集或常量子集
    for (const auto& sub_cu : sub_cus) {
        if (sub_cu.empty()) {
            return 0.0; // 修改：空子集返回 0.0
        }
        double first_val = sub_cu[0];
        bool all_same = true;
        for (double val : sub_cu) {
            if (val != first_val) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            return 0.0; // 修改：常量子集返回 0.0
        }
    }

    // 计算每个子 CU 的均值和绝对偏差
    std::vector<double> group_means(4);
    std::vector<std::vector<double>> abs_deviations(4);
    for (size_t i = 0; i < 4; ++i) {
        group_means[i] = calculate_mean(sub_cus[i]);
        abs_deviations[i].reserve(sub_cus[i].size());
        for (double val : sub_cus[i]) {
            abs_deviations[i].push_back(std::abs(val - group_means[i]));
        }
    }

    // 计算所有绝对偏差的总体均值
    std::vector<double> all_deviations;
    for (const auto& devs : abs_deviations) {
        all_deviations.insert(all_deviations.end(), devs.begin(), devs.end());
    }
    double overall_mean = calculate_mean(all_deviations);

    // 计算 Levene 统计量
    size_t k = 4; // 组数
    size_t N = all_deviations.size(); // 总样本数
    double numerator = 0.0;
    double denominator = 0.0;

    for (size_t i = 0; i < k; ++i) {
        double ni = abs_deviations[i].size();
        double zi_bar = calculate_mean(abs_deviations[i]);
        numerator += ni * std::pow(zi_bar - overall_mean, 2);

        for (double zij : abs_deviations[i]) {
            denominator += std::pow(zij - zi_bar, 2);
        }
    }

    if (denominator == 0.0) {
        return 0.0; // 修改：分母为零返回 0.0
    }

    double levene_stat = (N - k) * numerator / ((k - 1) * denominator);
    return levene_stat;
}

// 计算 Kruskal-Wallis 统计量
double calculate_kruskal_w(const std::vector<std::vector<Pel>>& matrix) {
    // 检查是否为方阵
    size_t size = matrix.size();
    if (size == 0 || matrix[0].size() != size) {
        return 0.0; // 修复(2.2)：非方阵返回0，不抛异常
    }

    // 验证边长为偶数
    if (size % 2 != 0) {
        return 0.0; // 修复(2.2)：边长非偶返回0，不抛异常
    }

    // 子 CU 大小为边长的一半
    size_t sub_cu_size = size / 2;
    if (sub_cu_size == 0) {
        return 0.0; // 修改：矩阵太小返回 0.0
    }

    // 分割成 4 个子 CU 并展平
    std::vector<std::vector<double>> sub_cus(4);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            std::vector<double>& sub_cu = sub_cus[i * 2 + j];
            sub_cu.reserve(sub_cu_size * sub_cu_size); // 预分配空间
            for (size_t y = i * sub_cu_size; y < (i + 1) * sub_cu_size; ++y) {
                for (size_t x = j * sub_cu_size; x < (j + 1) * sub_cu_size; ++x) {
                    if (std::isnan(matrix[y][x]) || std::isinf(matrix[y][x])) {
                        return 0.0; // 修改：无效值返回 0.0
                    }
                    sub_cu.push_back(static_cast<double>(matrix[y][x]));
                }
            }
        }
    }

    // 检查是否有空子集
    for (const auto& sub_cu : sub_cus) {
        if (sub_cu.empty()) {
            return 0.0; // 修改：空子集返回 0.0
        }
    }

    // 收集所有样本并记录组别
    std::vector<std::pair<double, size_t>> all_samples;
    for (size_t i = 0; i < sub_cus.size(); ++i) {
        all_samples.reserve(all_samples.size() + sub_cus[i].size());
        for (double val : sub_cus[i]) {
            all_samples.emplace_back(val, i); // 记录值和组别
        }
    }

    // 检查是否有样本
    if (all_samples.empty()) {
        return 0.0; // 修改：无样本返回 0.0
    }

    // 按值排序以计算秩
    std::sort(all_samples.begin(), all_samples.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // 计算秩（简单平均秩，无并列处理）
    std::vector<double> ranks(all_samples.size());
    for (size_t i = 0; i < all_samples.size(); ++i) {
        ranks[i] = static_cast<double>(i + 1); // 秩从1开始
    }

    // 计算每组的秩和
    std::vector<double> rank_sums(4, 0.0);
    std::vector<size_t> group_sizes(4, 0);
    for (size_t i = 0; i < all_samples.size(); ++i) {
        size_t group_idx = all_samples[i].second;
        rank_sums[group_idx] += ranks[i];
        group_sizes[group_idx]++;
    }

    // 检查每组是否有样本
    for (size_t ni : group_sizes) {
        if (ni == 0) {
            return 0.0; // 修改：空组返回 0.0
        }
    }

    // 计算 Kruskal-Wallis 统计量 H
    size_t N = all_samples.size(); // 总样本数
    size_t k = 4; // 组数
    double sum_term = 0.0;
    for (size_t i = 0; i < k; ++i) {
        sum_term += rank_sums[i] * rank_sums[i] / group_sizes[i];
    }

    double H = (12.0 / (N * (N + 1))) * sum_term - 3.0 * (N + 1);

    // 确保 H 非负（理论上应为非负，但浮点误差可能导致小负值）
    return H < 0.0 ? 0.0 : H;
}

// 计算 Levene 和 Kruskal-Wallis 统计量
std::pair<double, double> calculate_levene_kruskal(const std::vector<std::vector<Pel>>& matrix) {
    // 检查矩阵是否为空或非方阵
    if (matrix.empty() || matrix[0].empty()) {
        std::cerr << "Warning: Empty matrix detected" << std::endl;
        return { 0.0, 0.0 }; // 修改：空矩阵返回 {0.0, 0.0}
    }
    size_t size = matrix.size();
    if (size != matrix[0].size()) {
        return { 0.0, 0.0 }; // 修复(2.2)：非方阵返回{0,0}，不抛异常
    }
    if (size < 2) {
        std::cerr << "Warning: Matrix too small" << std::endl;
        return { 0.0, 0.0 }; // 修改：矩阵太小返回 {0.0, 0.0}
    }

    // 子 CU 大小
    size_t sub_cu_size = size / 2;
    if (sub_cu_size == 0) {
        return { 0.0, 0.0 }; // 修改：子CU大小为0返回 {0.0, 0.0}
    }

    // 分割为 4 个子 CU
    std::vector<std::vector<Pel>> sub_cus(4);
    for (size_t idx = 0, y = 0; y < size; y += sub_cu_size) {
        for (size_t x = 0; x < size; x += sub_cu_size, ++idx) {
            sub_cus[idx].reserve(sub_cu_size * sub_cu_size);
            for (size_t i = y; i < y + sub_cu_size && i < size; ++i) {
                for (size_t j = x; j < x + sub_cu_size && j < size; ++j) {
                    sub_cus[idx].push_back(matrix[i][j]);
                }
            }
        }
    }

    // 检查空子集或常量子集
    for (const auto& cu : sub_cus) {
        if (cu.empty()) {
            std::cerr << "Warning: Empty sub-CUs detected" << std::endl;
            return { 0.0, 0.0 }; // 修改：空子集返回 {0.0, 0.0}
        }
        if (cu.size() > 1 && std::all_of(cu.begin() + 1, cu.end(), 
            [&cu](Pel val) { return val == cu[0]; })) {
            return { 0.0, 0.0 }; // 修改：常量子集返回 {0.0, 0.0}
        }
    }

    // 计算 Levene 统计量
    double levene = [&]() {
        std::vector<double> group_means(4);
        std::vector<std::vector<double>> abs_devs(4);
        size_t total_n = 0;

        // 计算均值和绝对偏差
        for (size_t i = 0; i < 4; ++i) {
            const auto& cu = sub_cus[i];
            double sum = std::accumulate(cu.begin(), cu.end(), 0.0);
            group_means[i] = sum / cu.size();
            abs_devs[i].reserve(cu.size());
            for (const auto& val : cu) {
                abs_devs[i].push_back(std::abs(val - group_means[i]));
            }
            total_n += cu.size();
        }

        // 总体均值
        double overall_mean = 0.0;
        for (const auto& devs : abs_devs) {
            overall_mean += std::accumulate(devs.begin(), devs.end(), 0.0);
        }
        overall_mean /= total_n;

        // 计算平方和
        double ss_between = 0.0, ss_total = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            double group_mean = std::accumulate(abs_devs[i].begin(), abs_devs[i].end(), 0.0) / abs_devs[i].size();
            ss_between += abs_devs[i].size() * (group_mean - overall_mean) * (group_mean - overall_mean);
            for (const auto& dev : abs_devs[i]) {
                ss_total += (dev - overall_mean) * (dev - overall_mean);
            }
        }

        // Levene 统计量
        double df_between = 3.0; // k - 1
        double df_within = total_n - 4.0;
        double ms_between = ss_between / df_between;
        double ms_within = (ss_total - ss_between) / df_within;
        return ms_between / (ms_within + 1e-10); // 避免除零
    }();

    // 计算 Kruskal-Wallis 统计量
    double kruskal = [&]() {
        // 合并所有值并记录组索引
        std::vector<std::pair<double, int>> all_values;
        for (int i = 0; i < 4; ++i) {
            all_values.reserve(all_values.size() + sub_cus[i].size());
            for (const auto& val : sub_cus[i]) {
                all_values.emplace_back(val, i);
            }
        }

        // 排序并分配秩次
        std::sort(all_values.begin(), all_values.end());
        std::vector<double> ranks(all_values.size());
        for (size_t i = 0; i < all_values.size();) {
            size_t start = i;
            double sum_ranks = i + 1;
            while (i + 1 < all_values.size() && all_values[i].first == all_values[i + 1].first) {
                ++i;
                sum_ranks += i + 1;
            }
            double avg_rank = sum_ranks / (i - start + 1);
            for (size_t j = start; j <= i; ++j) {
                ranks[j] = avg_rank;
            }
            ++i;
        }

        // 计算秩和
        std::vector<double> rank_sums(4, 0.0);
        std::vector<size_t> group_sizes(4, 0);
        for (size_t i = 0; i < all_values.size(); ++i) {
            int group = all_values[i].second;
            rank_sums[group] += ranks[i];
            group_sizes[group]++;
        }

        // Kruskal-Wallis 统计量
        double n = static_cast<double>(all_values.size());
        double h = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            if (group_sizes[i] > 0) {
                h += rank_sums[i] * rank_sums[i] / group_sizes[i];
            }
        }
        h = 12.0 / (n * (n + 1)) * h - 3.0 * (n + 1);
        return h;
    }();

    return { levene, kruskal };
}

int get_qp_value(int poc_id, int qp) {
    int gop = poc_id % 32;

    static const std::unordered_map<int, int> gop2tid_map = {
        {0, 0}, {16, 1}, {8, 2}, {4, 3}, {2, 4}, {1, 5}, {3, 5},
        {6, 4}, {5, 5}, {7, 5}, {12, 3}, {10, 4}, {9, 5}, {11, 5},
        {14, 4}, {13, 5}, {15, 5}, {24, 2}, {20, 3}, {18, 4}, {17, 5},
        {19, 5}, {22, 4}, {21, 5}, {23, 5}, {28, 3}, {26, 4}, {25, 5},
        {27, 5}, {30, 4}, {29, 5}, {31, 5}
    };

    auto tid_it = gop2tid_map.find(gop);
    if (tid_it == gop2tid_map.end()) {
        return qp; // 修复(2.2)：未知gop退回原始QP，不抛异常（支持非GOP32结构）
    }
    int tid = tid_it->second;

    static const std::unordered_map<int, int> qp22tidmap = {{0, -3}, {1, 0}, {2, 1}, {3, 4}, {4, 6}, {5, 7}};
    static const std::unordered_map<int, int> qp27tidmap = {{0, -3}, {1, 1}, {2, 2}, {3, 5}, {4, 7}, {5, 9}};
    static const std::unordered_map<int, int> qp32tidmap = {{0, -3}, {1, 2}, {2, 3}, {3, 6}, {4, 8}, {5, 9}};
    static const std::unordered_map<int, int> qp37tidmap = {{0, -3}, {1, 3}, {2, 4}, {3, 6}, {4, 8}, {5, 9}};

    const std::unordered_map<int, int>* qp_map = nullptr;
    if (qp == 22) {
        qp_map = &qp22tidmap;
    } else if (qp == 27) {
        qp_map = &qp27tidmap;
    } else if (qp == 32) {
        qp_map = &qp32tidmap;
    } else if (qp == 37) {
        qp_map = &qp37tidmap;
    } else {
        return qp; // 修复(2.2)：非CTC QP退回原始QP，不抛异常（避免QP≠22/27/32/37时崩溃）
    }

    auto it = qp_map->find(tid);
    if (it == qp_map->end()) {
        return qp; // 修复(2.2)：未知tid退回原始QP，不抛异常
    }

    return qp + it->second;
}

// 计算多QP加权 Calculating multiple QP weights
double multi_qp_weight(double data, double qp_value) {
    const double INV_51 = 0.01960784f; // 1/51
    const double beta = 0.5f;
    double qp_weight = std::exp(-beta * qp_value * INV_51);
    return qp_weight * data;
}


