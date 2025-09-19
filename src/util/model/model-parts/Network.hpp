
#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <stdexcept>
#include <variant>

#include "ActivationPolicies.hpp"
#include "Layer.hpp"
#include "LossPolicies.hpp"
#include "../../types/eigen_types.hpp"
#include "../../logging.hpp"


using AnyLayer = std::variant<
    Layer<SigmoidPolicy>,
    Layer<LinearPolicy>,
    Layer<ReLUPolicy>,
    Layer<SoftmaxPolicy>
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
            } else if (activation == PolicyType::SOFTMAX) {
                _layers.emplace_back(Layer<SoftmaxPolicy>(layerSize, lastLayerSize));
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

    void train(const std::vector<Input>& trainingData, const std::vector<Output>& expectedOutputs, u32 epochs, f32 learningRate, const AnyLossPolicy& lossFunction) {
        if (trainingData.size() != expectedOutputs.size()) {
            throw std::invalid_argument("Training data and expected outputs must have the same size.");
        }

        for (u32 epoch = 0; epoch < epochs; ++epoch) {
            f32 totalError = 0;
            for (u32 i = 0; i < trainingData.size(); ++i) {
                const auto& input = trainingData[i];
                const auto& expected = expectedOutputs[i];

                Output actual = run(input);

                // вычисляем ошибку, используя переданную политику
                totalError += std::visit([&](const auto& policy) {
                    return policy.calculate(actual, expected);
                }, lossFunction);

                // backward Pass
                // вычисляем производную функции потерь
                Output loss_derivative = std::visit([&](const auto& policy) {
                    return policy.derivative(actual, expected);
                }, lossFunction);

                // вычисляем начальную дельту для выходного слоя
                Output delta = loss_derivative.cwiseProduct(
                    std::visit([](auto& layer){ return layer.activationDerivative(); }, _layers.back())
                );

                // обновляем веса выходного слоя (логика та же)
                std::visit([&](auto& layer){
                    auto& weights = layer.getWeights();
                    auto& biases = layer.getBiases();
                    const auto& prevLayerOutput = _layers.size() > 1 ?
                        std::visit([](auto& l){ return l.getLastOutput(); }, _layers[_layers.size()-2]) :
                        input;

                    weights -= learningRate * (delta * prevLayerOutput.transpose());
                    biases -= learningRate * delta;
                }, _layers.back());

                // распространяем ошибку на скрытые слои (логика та же)
                for (i64 j = _layers.size() - 2; j >= 0; --j) {
                    delta = std::visit([&](const auto& nextLayer, auto& currentLayer) -> Output {
                        Output newDelta = (nextLayer.getWeights().transpose() * delta).cwiseProduct(currentLayer.activationDerivative());

                        const auto& prevLayerOutput = (j > 0) ?
                            std::visit([](auto& l){ return l.getLastOutput(); }, _layers[j-1]) :
                            input;

                        currentLayer.getWeights() -= learningRate * (newDelta * prevLayerOutput.transpose());
                        currentLayer.getBiases() -= learningRate * newDelta;

                        return newDelta;
                    }, _layers[j+1], _layers[j]);
                }
            }
            if ((epoch + 1) % 10 == 0) {
                 Log::Logger().debug("Epoch {}/{}, Avg Error: {}", epoch + 1, epochs, totalError / trainingData.size());
            }
        }
    }
};

#endif
