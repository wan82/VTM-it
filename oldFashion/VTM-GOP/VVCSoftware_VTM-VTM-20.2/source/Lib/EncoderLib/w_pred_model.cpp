#include "w_pred_model.h"
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cstdint>

// NaiveBayes 类实现
NaiveBayes::NaiveBayes(const std::vector<Feature>& features, 
                       const std::vector<double>& prior_log, 
                       size_t expected_features)
    : features_(features), 
      prior_log_(prior_log), 
      expected_features_(expected_features) {
    if (prior_log_.size() != 2) {
        throw std::invalid_argument("Prior log probabilities must have 2 classes.");
    }
    if (features_.size() != expected_features_) {
        throw std::invalid_argument("Feature count does not match expected features.");
    }
}

size_t NaiveBayes::find_bin(double value, const std::vector<double>& bins) const {
    for (size_t i = 0; i < bins.size() - 1; ++i) {
        if (value >= bins[i] && value < bins[i + 1]) {
            return i;
        }
    }
    return bins.size() - 2; // 最后一个区间
}

bool NaiveBayes::predict(const std::vector<double>& X) const {
    if (X.size() != expected_features_) {
        throw std::invalid_argument("Input feature count does not match expected count.");
    }

    // 初始化后验概率
    std::vector<double> log_posteriors = prior_log_;

    // 遍历每个特征
    for (size_t i = 0; i < features_.size(); ++i) {
        double value = X[i];
        size_t bin_idx = find_bin(value, features_[i].bins);

        // 计算条件概率并累加对数
        for (int cls = 0; cls < 2; ++cls) {
            double count = (cls == 0) ? features_[i].counts_0[bin_idx] : features_[i].counts_1[bin_idx];
            double total = (cls == 0) ? features_[i].total_0 : features_[i].total_1;
            double prob = (count > 0) ? count / total : 1e-10; // 平滑处理
            log_posteriors[cls] += std::log(prob);
        }
    }

    // 返回概率最大的类别
    return log_posteriors[0] > log_posteriors[1] ? 0 : 1;
}


//  NaiveBayes 模型实现
// Naive Bayes prediction 128
bool naive_bayes_predict_128(const std::vector<double>& X) {
    static const std::vector<NaiveBayes::Feature> features = {
        // mean
        {
            { -std::numeric_limits<double>::infinity(), 50.668051, 55.700443, 243.44468, 397.316125, 
              453.312835, 556.740315, 596.662336, std::numeric_limits<double>::infinity() },
            { 541.0, 75.0, 790.0, 317.0, 272.0, 592.0, 33.0, 9.0 },
            { 18.0, 35.0, 1040.0, 813.0, 391.0, 586.0, 162.0, 342.0 },
            2629.0, 3387.0
        },
        // std
        {
            { -std::numeric_limits<double>::infinity(), 3.456036, 4.96729, 6.806647, 28.192388, 
              52.119281, 107.361255, std::numeric_limits<double>::infinity() },
            { 562.0, 188.0, 222.0, 725.0, 399.0, 449.0, 83.0 },
            { 8.0, 27.0, 89.0, 713.0, 697.0, 1275.0, 577.0 },
            2628.0, 3386.0
        },
        // sobel_mean
        {
            { -std::numeric_limits<double>::infinity(), 1.885134, 6.604774, 11.239765, 
              std::numeric_limits<double>::infinity() },
            { 449.0, 601.0, 231.0, 1344.0 },
            { 4.0, 122.0, 137.0, 3120.0 },
            2625.0, 3383.0
        },
        // res_std
        {
            { -std::numeric_limits<double>::infinity(), 5117.698023, 8750.135539, 17006.872391, 
              21736.970595, std::numeric_limits<double>::infinity() },
            { 191.0, 229.0, 584.0, 1405.0, 217.0 },
            { 250.0, 121.0, 1021.0, 1444.0, 548.0 },
            2626.0, 3384.0
        },
        // res_kurt
        {
            { -std::numeric_limits<double>::infinity(), -1.325434, -1.241247, -1.007847, -0.710405, 
              0.139987, 4.787635, 134.430757, std::numeric_limits<double>::infinity() },
            { 128.0, 258.0, 924.0, 356.0, 300.0, 269.0, 385.0, 9.0 },
            { 360.0, 390.0, 875.0, 556.0, 758.0, 313.0, 114.0, 21.0 },
            2629.0, 3387.0
        },
        // res_levene
        {
            { -std::numeric_limits<double>::infinity(), 0.850274, 4.296071, 11.799576, 96.415758, 
              std::numeric_limits<double>::infinity() },
            { 509.0, 733.0, 488.0, 825.0, 71.0 },
            { 203.0, 578.0, 578.0, 1676.0, 349.0 },
            2626.0, 3384.0
        },
        // res_krusw
        {
            { -std::numeric_limits<double>::infinity(), 2.470229, 21.530346, 151.632663, 
              std::numeric_limits<double>::infinity() },
            { 414.0, 983.0, 969.0, 259.0 },
            { 151.0, 822.0, 1506.0, 904.0 },
            2625.0, 3383.0
        },
        // qp_value
        {
            { -std::numeric_limits<double>::infinity(), 29, 34, 39, std::numeric_limits<double>::infinity() },
            { 479.0, 601.0, 773.0, 772.0 },
            { 1173.0, 916.0, 784.0, 510.0 },
            2625.0, 3383.0
        }
    };
    static const std::vector<double> prior_log = { std::log(0.44), std::log(0.56) }; // 先验概率 0, 1
    static const NaiveBayes classifier(features, prior_log, 8);
    return classifier.predict(X);
}

bool naive_bayes_predict_64(const std::vector<double>& X) {
    static const std::vector<NaiveBayes::Feature> features = {
        // sobel_std
        {
            { -std::numeric_limits<double>::infinity(), 6.600612, 9.229846, 12.224447, 30.692907, 47.08904, 
              std::numeric_limits<double>::infinity() },
            { 716, 358, 380, 1716, 1114, 489 },
            { 35, 82, 176, 1887, 2563, 1729 },
            4773, 6472
        },
        // res_std
        {
            { -std::numeric_limits<double>::infinity(), 3.879778, 11668.433387, std::numeric_limits<double>::infinity() },
            { 16, 947, 3807 },
            { 1, 1810, 4658 },
            4770, 6469
        },
        // res_kurt
        {
            { -std::numeric_limits<double>::infinity(), -1.019276, std::numeric_limits<double>::infinity() },
            { 2099, 2667 },
            { 2320, 4148 },
            4766, 6468
        },
        // res_levene
        {
            { -std::numeric_limits<double>::infinity(), 2.355374, std::numeric_limits<double>::infinity() },
            { 1483, 3283 },
            { 1382, 5086 },
            4766, 6468
        },
        // qp_value
        {
            { -std::numeric_limits<double>::infinity(), 25.01615, 26.981252, 27.003503, 27.998072, 28.000418, 
              29.999218, 30.000519, 31.999753, 32.006013, 32.997446, 38.00721, 39.99416, 40.010348, 
              41.998222, std::numeric_limits<double>::infinity() },
            { 359, 1, 450, 1, 746, 1, 146, 1, 402, 1, 1790, 1, 126, 1, 756 },
            { 483, 162, 500, 393, 1219, 362, 116, 353, 258, 242, 1644, 116, 45, 116, 472 },
            4782, 6481
        }
    };
    static const std::vector<double> prior_log = { std::log(0.42), std::log(0.58) };
    static const NaiveBayes classifier(features, prior_log, 5);
    return classifier.predict(X);
}

bool naive_bayes_predict_32(const std::vector<double>& X) {

    static const std::vector<NaiveBayes::Feature> features = {
        // skew
        {
            {-std::numeric_limits<double>::infinity(), -0.729267, std::numeric_limits<double>::infinity()},
            {739, 4797},
            {845, 4079},
            5536, 4924
        },
        // krusw_test
        {
            {-std::numeric_limits<double>::infinity(), 17.067541, 95.165705, 144.361476, std::numeric_limits<double>::infinity()},
            {990, 2997, 1414, 137},
            {524, 3180, 1174, 48},
            5538, 4926
        },
        // sobel_mean
        {
            {-std::numeric_limits<double>::infinity(), 14.48289, 20.661253, 47.236755, 62.018202, std::numeric_limits<double>::infinity()},
            {699, 605, 1996, 887, 1357},
            {96, 162, 1338, 902, 2429},
            5544, 4927
        },
        // sobel_levene
        {
            {-std::numeric_limits<double>::infinity(), 2.073656, std::numeric_limits<double>::infinity()},
            {1109, 4427},
            {649, 4275},
            5536, 4924
        },
        // res_mean
        {
            {-std::numeric_limits<double>::infinity(), 16.367786, std::numeric_limits<double>::infinity()},
            {32, 5509},
            {4, 4920},
            5541, 4924
        },
        // res_krusw
        {
            {-std::numeric_limits<double>::infinity(), 1.145043, 52.921545, std::numeric_limits<double>::infinity()},
            {527, 4454, 558},
            {308, 4267, 350},
            5539, 4925
        },
        // qp_value
        {
            {-std::numeric_limits<double>::infinity(), 27.000001, 27.995411, 28.000418, 29.993545, 30.000519, 
             31.999172, 32.012374, 32.995336, 40.029616, 41.994446, std::numeric_limits<double>::infinity()},
            {1139, 1, 1270, 1, 165, 1, 408, 1, 1997, 1, 566},
            {941, 305, 1306, 294, 87, 315, 137, 162, 1161, 62, 163},
            5550, 4933
        }
    };

    // 先验概率
    static const std::vector<double> prior_log = { std::log(0.53), std::log(0.47) };
    static const NaiveBayes classifier(features, prior_log, 7);
    return classifier.predict(X);
}

