
#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <stdexcept>
#include <variant>

#include "ActivationPolicies.hpp"
#include "Layer.hpp"
#include "../util/eigen_types.hpp"
#include "../util/constants.hpp"

using AnyLayer = std::variant<
    Layer<SigmoidPolicy>,
    Layer<LinearPolicy>,
    Layer<ReLUPolicy>
>;

class Network {
    std::vector<AnyLayer> _layers{};
public:

    explicit Network(u32 inputSize, const std::vector<std::pair<u32, PolicyType>>& layersConfig) {
        if (layersConfig.empty()) {
            throw std::invalid_argument("Layers config must not be empty");
        }

        u32 lastLayerSize = inputSize;
        for (const auto& config : layersConfig) {
            u32 layerSize = config.first;
            const PolicyType& activation = config.second;

            if (activation == PolicyType::RELU) {
                _layers.emplace_back(Layer<ReLUPolicy>(layerSize, lastLayerSize));
            } else if (activation == PolicyType::SIGMOID) {
                _layers.emplace_back(Layer<SigmoidPolicy>(layerSize, lastLayerSize));
            } else if (activation == PolicyType::LINEAR) {
                _layers.emplace_back(Layer<LinearPolicy>(layerSize, lastLayerSize));
            } else {
                throw std::invalid_argument("Unknown activation function");
            }
            lastLayerSize = layerSize;
        }
    }

    Output run(const Input& input){
        if (_layers.empty()) throw std::invalid_argument("No layers provided");
        if (input.empty()) throw std::invalid_argument("No input provided");

        Input temp(input);
        for (auto& layer_variant : _layers) {
            temp = std::visit([&](auto& concrete_layer) {
                return concrete_layer.activate(temp);
            }, layer_variant);
        }
        return temp;
    };

    void train(const std::vector<Input>& trainingData, const std::vector<Output>& expectedOutputs, u32 epochs, f32 learningRate) {
        if (trainingData.size() != expectedOutputs.size()) {
            throw std::invalid_argument("Training data and expected outputs must have the same size.");
        }

        for (u32 epoch = 0; epoch < epochs; ++epoch) {
            f32 totalError = 0;
            for (u32 i = 0; i < trainingData.size(); ++i) {
                const auto& input = trainingData[i];
                const auto& expected = expectedOutputs[i];

                Output actual = run(input);

                for(u32 j = 0; j < actual.size(); ++j) {
                    totalError += (expected[j] - actual[j]) * (expected[j] - actual[j]);
                }

                backwardPass(expected);
                updateWeights(input, learningRate);
            }
            if (LOG_STATUS) {
                std::println(std::cout, "Epoch {}/{}, Error: {}", epoch + 1,epochs,totalError);
            }
        }
    }

private:
    void backwardPass(const Output& expectedOutput) {
        // --- Обработка выходного слоя ---
        auto& outputLayer_variant = _layers.back();
        std::visit([&](auto& outputLayer) {
            auto& outputNeurons = outputLayer.getNeurons();
            for (u32 i = 0; i < outputNeurons.size(); ++i) {
                auto& neuron = outputNeurons[i];
                f32 error = expectedOutput[i] - neuron.getOutput();
                neuron.setDelta(error * neuron.activationDerivative());
            }
        }, outputLayer_variant);

        // --- Обработка скрытых слоев ---
        for (i64 i = _layers.size() - 2; i >= 0; --i) {
            auto& hiddenLayer_variant = _layers[i];
            auto& nextLayer_variant = _layers[i + 1];

            std::visit([&](auto& hiddenLayer, const auto& nextLayer) {
                auto& hiddenNeurons = hiddenLayer.getNeurons();
                const auto& nextNeurons = nextLayer.getNeurons();

                for (u32 j = 0; j < hiddenNeurons.size(); ++j) {
                    auto& neuron = hiddenNeurons[j];
                    f32 error = 0.0f;
                    for (const auto& nextNeuron : nextNeurons) {
                        error += nextNeuron.getWeights()[j] * nextNeuron.getDelta();
                    }
                    neuron.setDelta(error * neuron.activationDerivative());
                }
            }, hiddenLayer_variant, nextLayer_variant);
        }
    }

    void updateWeights(const Input& initialInput, f32 learningRate) {
        Input currentInput = initialInput;

        for (auto& layer_variant : _layers) {
            currentInput = std::visit([&](auto& layer) {
                auto& neurons = layer.getNeurons();
                for (auto& neuron : neurons) {
                    neuron.updateWeights(currentInput, learningRate);
                }

                Output layerOutput;
                layerOutput.reserve(neurons.size());
                for(const auto& neuron : neurons) {
                    layerOutput.push_back(neuron.getOutput());
                }
                return layerOutput; // Возвращаем выход для следующего слоя
            }, layer_variant);
        }
    }
};

#endif
