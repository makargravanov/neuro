//
// Created by Alex on 20.09.2025.
//

#ifndef CONVLAYER_HPP
#define CONVLAYER_HPP


#include "../../../types/eigen_types.hpp"
#include "../ActivationPolicies.hpp"
#include "../ComputePolicies.h"

struct ConvConfig {
    u32 inputChannels;
    u32 outputChannels;
    u32 kernelSize; // Предполагаем квадратные ядра
    u32 stride = 1;
    u32 padding = 0;
};

template<typename ActivationPolicy, typename ComputePolicy>
class ConvLayer {
    KernelTensor _kernels; // Веса (фильтры)
    BiasVector _biases;    // Смещения (одно на каждый выходной канал)

    Tensor4f _lastInput;
    Tensor4f _lastOutput;

    ConvConfig _config;

public:
    ConvLayer(u32 inputHeight, u32 inputWidth, ConvConfig config) : _config(config) {
        // Инициализация весов случайными значениями
        _kernels(std::array<Eigen::DenseIndex, 4>{
            static_cast<Eigen::DenseIndex>(_config.outputChannels),
            static_cast<Eigen::DenseIndex>(_config.inputChannels),
            static_cast<Eigen::DenseIndex>(_config.kernelSize),
            static_cast<Eigen::DenseIndex>(_config.kernelSize)
        });
        _kernels.setRandom();
        _kernels = _kernels * static_cast<f32>(
            std::sqrt(2.0 / (_config.inputChannels * _config.kernelSize * _config.kernelSize))
        ); // Инициализация He

        // Инициализация смещений
        _biases = BiasVector::Zero(_config.outputChannels);
    }

    /**
     * @brief Прямое распространение через сверточный слой.
     * @param input 4D-тензор (N, C_in, H_in, W_in).
     * @return 4D-тензор (N, C_out, H_out, W_out).
     */
    Tensor4f activate(const Tensor4f& input) {
        _lastInput = input;

        // 1. Выполнение свертки через ComputePolicy
        Tensor4f z = ComputePolicy::convolution(input, _kernels, _biases, _config.stride, _config.padding);

        // 2. Применение поэлементной функции активации
        _lastOutput = ComputePolicy::template activate<ActivationPolicy>(z);

        return _lastOutput;
    }

    // --- Геттеры ---
    KernelTensor& getKernels() { return _kernels; }
    BiasVector& getBiases() { return _biases; }
    [[nodiscard]] const KernelTensor& getKernels() const { return _kernels; }
    [[nodiscard]] const BiasVector& getBiases() const { return _biases; }

    [[nodiscard]] const Tensor4f& getLastOutput() const { return _lastOutput; }
    [[nodiscard]] const Tensor4f& getLastInput() const { return _lastInput; }
    [[nodiscard]] const ConvConfig& getConfig() const { return _config; }

    /**
     * @brief Вычисляет производную функции активации для последнего выхода.
     */
    [[nodiscard]] Tensor4f activationDerivative() const {
        return ComputePolicy::template activationDerivative<ActivationPolicy>(_lastOutput);
    }
};
#endif //CONVLAYER_HPP
