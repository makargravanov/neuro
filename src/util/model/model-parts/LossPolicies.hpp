//
// Created by Alex on 19.09.2025.
//

#ifndef LOSSPOLICIES_HPP
#define LOSSPOLICIES_HPP
#include <variant>
#include "../../types/eigen_types.hpp" // Убедитесь, что путь корректен

// Enum для удобного выбора функции потерь на верхнем уровне (в классе Model)
enum class LossType {
    MEAN_SQUARED_ERROR,
    CATEGORICAL_CROSS_ENTROPY
};

struct MeanSquaredErrorPolicy {
    static f32 calculate(const Output& actual, const Output& expected) {
        // Возвращаем среднюю квадратичную ошибку по батчу
        return (expected - actual).squaredNorm() / actual.cols();
    }

    static Output derivative(const Output& actual, const Output& expected) {
        // Производная средней MSE по выходу сети (dE/da)
        return (actual - expected) / actual.cols();
    }
};

struct CategoricalCrossEntropyPolicy {
    static f32 calculate(const Output& actual, const Output& expected) {
        constexpr f32 epsilon = 1e-9f;
        Output clippedActual = actual.cwiseMax(epsilon).cwiseMin(1.0f - epsilon);
        // Возвращаем среднюю CCE по батчу
        return -(expected.array() * clippedActual.array().log()).sum() / actual.cols();
    }

    static Output derivative(const Output& actual, const Output& expected) {
        // Упрощенная производная для связки CCE + Softmax, усредненная по батчу
        return (actual - expected) / actual.cols();
    }
};

using AnyLossPolicy = std::variant<MeanSquaredErrorPolicy, CategoricalCrossEntropyPolicy>;

#endif //LOSSPOLICIES_HPP
