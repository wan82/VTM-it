#include "SpatialFeatureLib/CtuFeatures.h"
#include "SpatialFeatureLib/Moments.h"
#include "SpatialFeatureLib/Filters.h"
#include "SpatialFeatureLib/VarianceTests.h"
#include <sstream>
#include <iomanip>

namespace sf {

std::string CtuFeatures::csvHeader() {
    return "mean,std,skew,kurt,"
           "Sobel_mean,Sobel_std,Sobel_skew,Sobel_kurt,"
           "Lap_mean,Lap_std,Lap_skew,Lap_kurt,"
           "LeveneTest,KrusWTest,SobelLevene,SobelKrusW,LapLevene,LapKrusW";
}

std::string CtuFeatures::toCsvFields() const {
    std::ostringstream os;
    os << std::setprecision(15);
    os << mean << ',' << std << ',' << skew << ',' << kurt << ','
       << sobelMean << ',' << sobelStd << ',' << sobelSkew << ',' << sobelKurt << ','
       << lapMean << ',' << lapStd << ',' << lapSkew << ',' << lapKurt << ','
       << levene << ',' << krusW << ',' << sobelLevene << ',' << sobelKrusW << ','
       << lapLevene << ',' << lapKrusW;
    return os.str();
}

CtuFeatures computeCtuFeatures(const Mat& block) {
    CtuFeatures f;

    // 1) 原始亮度块上的统计矩
    f.mean = mean(block);
    f.std  = stddev(block);
    f.skew = skewness(block);
    f.kurt = kurtosis(block);

    // 2) Sobel 边缘图上的统计矩
    Mat sob = sobel(block);
    f.sobelMean = mean(sob);
    f.sobelStd  = stddev(sob);
    f.sobelSkew = skewness(sob);
    f.sobelKurt = kurtosis(sob);

    // 3) Laplacian 图上的统计矩
    Mat lap = laplacian(block);
    f.lapMean = mean(lap);
    f.lapStd  = stddev(lap);
    f.lapSkew = skewness(lap);
    f.lapKurt = kurtosis(lap);

    // 4) 方差齐性 / 分布差异统计量（原图 / Sobel / Laplacian）
    f.levene      = levene(block);
    f.krusW       = kruskalWallis(block);
    f.sobelLevene = levene(sob);
    f.sobelKrusW  = kruskalWallis(sob);
    f.lapLevene   = levene(lap);
    f.lapKrusW    = kruskalWallis(lap);

    return f;
}

} // namespace sf
