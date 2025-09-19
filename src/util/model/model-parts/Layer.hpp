
#ifndef LAYER_HPP
#define LAYER_HPP

#include "../../types/eigen_types.hpp"

template<typename ActivationPolicy>
class Layer {
    WeightMatrix _weights;
    BiasVector _biases;

    Output _lastOutput;
    Input _lastInput;
public:
    Layer() = default;

    explicit Layer(const u32 numberOfNeurons, u32 lastNumberOfNeurons) {
        _weights = WeightMatrix::Random(numberOfNeurons, lastNumberOfNeurons);
        _biases = BiasVector::Random(numberOfNeurons);
    };

    Output activate(const Input& input) {
        _lastInput = input;
        Input z = _weights * input + _biases;

        // Используем if constexpr для специализации логики во время компиляции
        // Это не создает никаких накладных расходов в рантайме.
        if constexpr (std::is_same_v<ActivationPolicy, SoftmaxPolicy>) {
            // Реализация Softmax с защитой от численной нестабильности.
            // Вычитание максимального элемента из вектора z не меняет результат,
            // но предотвращает переполнение (overflow) при вычислении exp() для больших чисел.
            f32 maxCoeff = z.maxCoeff();
            Output expZ = (z.array() - maxCoeff).exp();
            _lastOutput = expZ / expZ.sum();
        } else {
            // Стандартная поэлементная активация для всех остальных политик
            _lastOutput = z.unaryExpr(&ActivationPolicy::activate);
        }

        return _lastOutput;
    }


    WeightMatrix& getWeights() { return _weights; }
    BiasVector& getBiases() { return _biases; }
    [[nodiscard]] const WeightMatrix& getWeights() const { return _weights; }
    [[nodiscard]] const BiasVector& getBiases() const { return _biases; }

    [[nodiscard]] const Output& getLastOutput() const { return _lastOutput; }
    [[nodiscard]] const Input& getLastInput() const { return _lastInput; }

    [[nodiscard]] Output activationDerivative() const {
        return _lastOutput.unaryExpr(&ActivationPolicy::derivative);
    }
};

#endif
