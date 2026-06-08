#pragma once
#include <vector>
#include <map>

class NaiveBayes {
public:
    // 特征数据结构 // Feature data structure
    struct Feature {
        std::vector<double> bins;         // 区间边界 // Bin boundaries
        std::vector<double> counts_0;     // Class 0 的计数 // Counts for Class 0
        std::vector<double> counts_1;     // Class 1 的计数 // Counts for Class 1
        double total_0;                   // Class 0 总计数 // Total counts for Class 0
        double total_1;                   // Class 1 总计数 // Total counts for Class 1
    };

    // 构造函数 // Constructor
    NaiveBayes(const std::vector<Feature>& features, 
               const std::vector<double>& prior_log, 
               size_t expected_features);

    // 预测函数 // Predict function
    bool predict(const std::vector<double>& X) const;

private:
    std::vector<Feature> features_;           // 特征数据 // Feature data
    std::vector<double> prior_log_;          // 先验概率的对数 // Logarithm of prior probabilities
    size_t expected_features_;               // 期望的特征数量 // Expected number of features

    // 找到值所属的区间索引 // Find the index of the bin that contains the value
    size_t find_bin(double value, const std::vector<double>& bins) const;
};


// 128x128 图像的朴素贝叶斯预测函数 // Naive Bayes prediction function for 128x128 images
bool naive_bayes_predict_128(const std::vector<double>& X);

// 64x64 图像的朴素贝叶斯预测函数 // Naive Bayes prediction function for 64x64 images
bool naive_bayes_predict_64(const std::vector<double>& X);

// 32x32 图像的朴素贝叶斯预测函数 // Naive Bayes prediction function for 32x32 images
bool naive_bayes_predict_32(const std::vector<double>& X);