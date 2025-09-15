
#ifndef LAYER_HPP
#define LAYER_HPP

#include <random>
#include "Neuron.hpp"


template<typename ActivationPolicy>
class Layer {
    std::vector<Neuron<ActivationPolicy>> _neurons;
public:

    Layer() = default;

    explicit Layer(const u32 numberOfNeurons, u32 lastNumberOfNeurons) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dis(-0.5f, 0.5f);

        _neurons.resize(numberOfNeurons);
        for (u32 i = 0; i < numberOfNeurons; i++) {
            Weights initialWeights(lastNumberOfNeurons);
            for(f32& weight : initialWeights) {
                weight = dis(gen);
            }
            f32 initialBias = dis(gen);
            _neurons.at(i) = Neuron<ActivationPolicy>(initialWeights, initialBias);
        }
    };

    std::vector<Neuron<ActivationPolicy>>& getNeurons() { return _neurons; }
    const std::vector<Neuron<ActivationPolicy>>& getNeurons() const { return _neurons; }

    Output activate(const Input& input) {
        Output output;
        output.reserve(_neurons.size());
        for (auto& neuron : _neurons) {
            output.push_back(neuron.activation(input));
        }

        return output;
    }
};

#endif
