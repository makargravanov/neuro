export module Neuron;

import std;
import types;

export using Input = std::vector<f32>;
export using Output = Input;
export using Weights = std::vector<f32>;

export class Neuron {
    Weights _weights{}; // по одному весу на каждый вход
    f32 _bias = 0;
    f32 _output = 0;
    f32 _delta = 0;

public:

    Neuron() = default;

    explicit Neuron(const Weights& weights, f32 bias)
        : _weights(weights), _bias(bias) {}


    f32 activation(const Input& input) { // Убираем const
        if (input.size() != _weights.size()) {
            throw std::invalid_argument("Input size must match weights size");
        }

        f32 sum = 0;
        for (u32 i = 0; i < input.size(); i++) {
            sum += input[i] * _weights[i];
        }
        sum += _bias;

        _output = 1.0 / (1.0 + std::exp(-sum));
        return _output;
    }

    f32 activationDerivative() const {
        return _output * (1.0f - _output);
    }

    const std::vector<f32>& getWeights() const { return _weights; }

    f32 getBias() const { return _bias; }

    void setWeights(const std::vector<f32>& w) {
        if (w.size() != _weights.size()) {
            throw std::invalid_argument("Size mismatch");
        }
        _weights = w;
    }

    void updateWeights(const Input& input, f32 learningRate) {
        for (u32 i = 0; i < _weights.size(); ++i) {
            _weights[i] += learningRate * _delta * input[i];
        }
        _bias += learningRate * _delta;
    }

    f32 getOutput() const { return _output; }
    f32 getDelta() const { return _delta; }
    void setDelta(f32 delta) { _delta = delta; }

    void setBias(f32 b) { _bias = b; }
};
