//
// Created by Alex on 19.09.2025.
//

#ifndef METRICS_HPP
#define METRICS_HPP

#include <vector>
#include <cmath>
#include "../../types/eigen_types.hpp"

struct ClassificationMetrics {
    f32 accuracy = 0.0f;
    u32 correctPredictions = 0;
    u32 totalSamples = 0;
    // в будущем здесь можно добавить матрицу ошибок, точность, полноту и F1-меру
};

struct RegressionMetrics {
    f32 meanAbsoluteError = 0.0f;
    f32 rootMeanSquaredError = 0.0f;
    f32 meanAbsolutePercentageError = 0.0f;
};

class MetricsService {
public:
    static ClassificationMetrics calculateClassificationMetrics(
        const std::vector<Output>& predictions,
        const std::vector<Output>& groundTruth)
    {
        if (predictions.size() != groundTruth.size() || predictions.empty()) {
            return {}; // возвращаем метрики по умолчанию, если данные некорректны
        }

        ClassificationMetrics metrics;
        metrics.totalSamples = predictions.size();

        for (u32 i = 0; i < metrics.totalSamples; ++i) {
            Eigen::Index predictedIndex, expectedIndex;
            predictions[i].maxCoeff(&predictedIndex);
            groundTruth[i].maxCoeff(&expectedIndex);
            if (predictedIndex == expectedIndex) {
                metrics.correctPredictions++;
            }
        }

        if (metrics.totalSamples > 0) {
            metrics.accuracy = static_cast<f32>(metrics.correctPredictions) / metrics.totalSamples * 100.0f;
        }
        return metrics;
    }

    static RegressionMetrics calculateRegressionMetrics(
        const std::vector<Output>& predictions,
        const std::vector<Output>& groundTruth)
    {
        if (predictions.size() != groundTruth.size() || predictions.empty()) {
            return {};
        }

        RegressionMetrics metrics;
        const u32 totalSamples = predictions.size();
        f32 totalAbsoluteError = 0.0f;
        f32 totalSquaredError = 0.0f;
        f32 totalPercentageError = 0.0f;
        u32 mape_count = 0;

        for (u32 i = 0; i < totalSamples; ++i) {
            const f32 predictedValue = predictions[i](0);
            const f32 expectedValue = groundTruth[i](0);

            const f32 error = predictedValue - expectedValue;
            totalAbsoluteError += std::abs(error);
            totalSquaredError += error * error;

            if (std::abs(expectedValue) > 1e-9) {
                totalPercentageError += (std::abs(error) / std::abs(expectedValue)) * 100.0f;
                mape_count++;
            }
        }

        if (totalSamples > 0) {
            metrics.meanAbsoluteError = totalAbsoluteError / totalSamples;
            metrics.rootMeanSquaredError = std::sqrt(totalSquaredError / totalSamples);
        }
        if (mape_count > 0) {
            metrics.meanAbsolutePercentageError = totalPercentageError / mape_count;
        }
        return metrics;
    }
};

#endif //METRICS_HPP
