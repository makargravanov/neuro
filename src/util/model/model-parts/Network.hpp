
#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <stdexcept>
#include <variant>

#include "ActivationPolicies.hpp"
#include "Layer.hpp"
#include "../../types/eigen_types.hpp"
#include "../../logging.hpp"

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

    Output run(const Input& input) {
        if (_layers.empty()) throw std::invalid_argument("No layers provided");
        Input temp = input;
        for (auto& layer_variant : _layers) {
            temp = std::visit([&](auto& concrete_layer) {
                return concrete_layer.activate(temp);
            }, layer_variant);
        }
        return temp;
    }

    void train(const std::vector<Input>& trainingData, const std::vector<Output>& expectedOutputs, u32 epochs, f32 learningRate) {
        if (trainingData.size() != expectedOutputs.size()) {
            throw std::invalid_argument("Training data and expected outputs must have the same size.");
        }

        for (u32 epoch = 0; epoch < epochs; ++epoch) {
            f32 totalError = 0;
            for (u32 i = 0; i < trainingData.size(); ++i) {
                const auto& input = trainingData[i];
                const auto& expected = expectedOutputs[i];

                // Forward Pass
                Output actual = run(input);
                totalError += (expected - actual).squaredNorm();

                // Backward Pass
                Output delta = (actual - expected).cwiseProduct(
                    std::visit([](auto& layer){ return layer.activationDerivative(); }, _layers.back())
                );

                // обновляем выходной слой
                std::visit([&](auto& layer){
                    auto& weights = layer.getWeights();
                    auto& biases = layer.getBiases();
                    const auto& prevLayerOutput = _layers.size() > 1 ?
                        std::visit([](auto& l){ return l.getLastOutput(); }, _layers[_layers.size()-2]) :
                        input;

                    weights -= learningRate * (delta * prevLayerOutput.transpose());
                    biases -= learningRate * delta;
                }, _layers.back());


                // распространение ошибки на скрытые слои
                for (i64 j = _layers.size() - 2; j >= 0; --j) {
                    delta = std::visit([&](const auto& nextLayer, auto& currentLayer) -> Output {
                        Output newDelta = (nextLayer.getWeights().transpose() * delta).cwiseProduct(currentLayer.activationDerivative());

                        // обновление весов текущего слоя
                        const auto& prevLayerOutput = (j > 0) ?
                            std::visit([](auto& l){ return l.getLastOutput(); }, _layers[j-1]) :
                            input;

                        currentLayer.getWeights() -= learningRate * (newDelta * prevLayerOutput.transpose());
                        currentLayer.getBiases() -= learningRate * newDelta;

                        return newDelta;
                    }, _layers[j+1], _layers[j]);
                }
            }
            Log::Logger().debug("Epoch {}/{}, Error: {}", epoch + 1, epochs, totalError);
        }
    }
};

#endif
