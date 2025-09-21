//
// Created by Alex on 20.09.2025.
//

#ifndef POOLINGLAYER_HPP
#define POOLINGLAYER_HPP

#include "../../../types/eigen_types.hpp"
#include "../ComputePolicies.hpp"

// Конфигурация для слоя пулинга
struct PoolConfig {
    u32 poolSize; // Размер окна (квадратного)
    u32 stride;
};

template<typename ComputePolicy>
class PoolingLayer {
    PoolConfig _config;

    Tensor4f _lastInput;
    Tensor4f _lastOutput;
    // Маска для хранения индексов максимальных элементов для backpropagation
    Eigen::Tensor<Eigen::DenseIndex, 4, Eigen::RowMajor> _maxIndices;

public:
    explicit PoolingLayer(PoolConfig config) : _config(config) {}

    /**
     * @brief Прямое распространение через слой Max-Pooling.
     * @param input 4D-тензор (N, C, H_in, W_in).
     * @return 4D-тензор (N, C, H_out, W_out).
     */
    Tensor4f activate(const Tensor4f& input) {
        _lastInput = input;

        // Делегируем вычисление политике, которая вернет и результат, и маску
        auto [output, indices] = ComputePolicy::maxPooling(input, _config.poolSize, _config.stride);
        _lastOutput = output;
        _maxIndices = indices;

        return _lastOutput;
    }

    // --- Геттеры ---
    [[nodiscard]] const Tensor4f& getLastOutput() const { return _lastOutput; }
    [[nodiscard]] const Tensor4f& getLastInput() const { return _lastInput; }
    [[nodiscard]] const auto& getMaxIndices() const { return _maxIndices; }
    [[nodiscard]] const PoolConfig& getConfig() const { return _config; }

    [[nodiscard]] Tensor4f activationDerivative() const {
        return _lastOutput.constant(1.0f);
    }
};
#endif //POOLINGLAYER_HPP
