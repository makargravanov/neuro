//
// Created by Alex on 19.09.2025.
//

#ifndef COMPUTEPOLICIES_H
#define COMPUTEPOLICIES_H


#include "../../types/eigen_types.hpp"
#include "ActivationPolicies.hpp"

/**
 * @struct CpuEigenPolicy
 * @brief Политика вычислений, использующая библиотеку Eigen для операций на CPU.
 *
 * Инкапсулирует все математические операции, необходимые для прямого и обратного
 * распространения сигнала в нейронной сети.
 */
struct CpuEigenPolicy {
    /**
     * @brief Выполняет основной шаг прямого распространения: z = (W * X).colwise() + b.
     */
    static Input forwardPass(const WeightMatrix& weights, const Input& input, const BiasVector& biases) {
        return (weights * input).colwise() + biases;
    }

    /**
     * @brief Применяет функцию активации к выходу линейного слоя.
     * @tparam ActivationPolicy Политика функции активации (например, SigmoidPolicy, SoftmaxPolicy).
     * @param z Входные данные для функции активации (результат forwardPass).
     * @return Результат после применения функции активации.
     */
    template<typename ActivationPolicy>
    static Output activate(const Input& z) {
        if constexpr (std::is_same_v<ActivationPolicy, SoftmaxPolicy>) {
            Eigen::RowVectorXf maxCoeffs = z.colwise().maxCoeff();
            Output expZ = (z.rowwise() - maxCoeffs).array().exp();
            return expZ.array().rowwise() / expZ.colwise().sum().array();
        } else {
            return z.unaryExpr(&ActivationPolicy::activate);
        }
    }

    /**
     * @brief Вычисляет производную функции активации.
     */
    template<typename ActivationPolicy>
    static Output activationDerivative(const Output& lastOutput) {
        return lastOutput.unaryExpr(&ActivationPolicy::derivative);
    }

    /**
     * @brief Вычисляет градиент для матрицы весов.
     */
    static WeightMatrix calculateWeightGradient(const Output& delta, const Input& prevLayerOutput) {
        return delta * prevLayerOutput.transpose();
    }

    /**
     * @brief Вычисляет градиент для вектора смещений.
     */
    static BiasVector calculateBiasGradient(const Output& delta) {
        return delta.rowwise().mean();
    }

    /**
     * @brief Обновляет веса слоя с использованием градиентного спуска.
     */
    static void updateWeights(WeightMatrix& weights, f32 learningRate, const WeightMatrix& weightGrad) {
        weights -= learningRate * weightGrad;
    }

    /**
     * @brief Обновляет смещения слоя с использованием градиентного спуска.
     */
    static void updateBiases(BiasVector& biases, f32 learningRate, const BiasVector& biasGrad) {
        biases -= learningRate * biasGrad;
    }

    /**
     * @brief Вычисляет ошибку (delta) для передачи на предыдущий слой.
     */
    static Output calculateNextDelta(const WeightMatrix& currentWeights, const Output& delta, const Output& prevActivationDerivative) {
        return (currentWeights.transpose() * delta).cwiseProduct(prevActivationDerivative);
    }

    /**
     * @brief Выполняет поэлементное умножение двух матриц.
     */
    static Output elementwiseProduct(const Output& a, const Output& b) {
        return a.cwiseProduct(b);
    }
};

/**
 * @struct GpuPolicy
 * @brief Заглушка для будущей реализации политики вычислений на GPU.
 *
 * Здесь могли бы быть реализованы те же статические методы, что и в CpuEigenPolicy,
 * но с использованием CUDA, OpenCL или других GPU-ускоренных библиотек.
 */
struct GpuPolicy {
    // TODO: Реализовать вычисления на GPU
};


#endif //COMPUTEPOLICIES_H
