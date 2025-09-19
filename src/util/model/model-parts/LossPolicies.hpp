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

// Политика для ошибки среднего квадратичного (хорошо для регрессии)
struct MeanSquaredErrorPolicy {
    static f32 calculate(const Output& actual, const Output& expected) {
        return (expected - actual).squaredNorm();
    }

    static Output derivative(const Output& actual, const Output& expected) {
        // Производная MSE по выходу сети (dE/da)
        return actual - expected;
    }
};

// Политика для категориальной кросс-энтропии (хорошо для классификации)
struct CategoricalCrossEntropyPolicy {
    static f32 calculate(const Output& actual, const Output& expected) {
        // Добавляем эпсилон для численной стабильности, чтобы избежать log(0)
        constexpr f32 epsilon = 1e-9f;
        Output clipped_actual = actual.cwiseMax(epsilon).cwiseMin(1.0f - epsilon);
        return -(expected.array() * clipped_actual.array().log()).sum();
    }

    static Output derivative(const Output& actual, const Output& expected) {
        // Это важное математическое упрощение, которое работает, когда CCE
        // используется вместе с активацией Softmax или Sigmoid на выходном слое.
        // Производная всей связки элегантно сводится к (actual - expected).
        return actual - expected;
    }
};

// Используем std::variant для хранения любой из политик без динамического полиморфизма
using AnyLossPolicy = std::variant<MeanSquaredErrorPolicy, CategoricalCrossEntropyPolicy>;

#endif //LOSSPOLICIES_HPP
