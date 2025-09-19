
#ifndef LAYER_HPP
#define LAYER_HPP

#include "../../../types/eigen_types.hpp"

template<typename ActivationPolicy, typename ComputePolicy>
class DenseLayer {
    WeightMatrix _weights;
    BiasVector _biases;

    DenseInput _lastInput;
    DenseOutput _lastOutput;

public:
    DenseLayer() = default;

    explicit DenseLayer(u32 numberOfNeurons, u32 lastNumberOfNeurons) {
        _weights = WeightMatrix::Random(numberOfNeurons, lastNumberOfNeurons);
        _biases = BiasVector::Random(numberOfNeurons);
    };

    /**
     * @brief Выполняет прямое распространение через полносвязный слой.
     * @param input Входные данные (выход предыдущего слоя).
     * @return Выходные данные этого слоя.
     */
    DenseOutput activate(const DenseInput& input) {
        _lastInput = input;

        // 1. Линейное преобразование с использованием политики вычислений
        DenseInput z = ComputePolicy::forwardPass(_weights, input, _biases);

        // 2. Применение функции активации через политику вычислений
        _lastOutput = ComputePolicy::template activate<ActivationPolicy>(z);

        return _lastOutput;
    }

    // --- Геттеры ---
    WeightMatrix& getWeights() { return _weights; }
    BiasVector& getBiases() { return _biases; }
    [[nodiscard]] const WeightMatrix& getWeights() const { return _weights; }
    [[nodiscard]] const BiasVector& getBiases() const { return _biases; }

    [[nodiscard]] const DenseOutput& getLastOutput() const { return _lastOutput; }
    [[nodiscard]] const DenseInput& getLastInput() const { return _lastInput; }

    /**
     * @brief Вычисляет производную функции активации для последнего выхода.
     * @return Матрица со значениями производных.
     */
    [[nodiscard]] DenseOutput activationDerivative() const {
        return ComputePolicy::template activationDerivative<ActivationPolicy>(_lastOutput);
    }
};


#endif
