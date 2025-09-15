
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
        _lastOutput = z.unaryExpr(&ActivationPolicy::activate);

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
