
#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <random>
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

    void train(const std::vector<Eigen::VectorXf>& trainingData, const std::vector<Eigen::VectorXf>& expectedOutputs, u32 epochs, u32 batchSize, f32 learningRate, const AnyLossPolicy& lossFunction) {
        if (trainingData.size() != expectedOutputs.size()) {
            throw std::invalid_argument("Training data and expected outputs must have the same size.");
        }

        const u32 numSamples = trainingData.size();
        std::vector<u32> indices(numSamples);
        std::iota(indices.begin(), indices.end(), 0);

        for (u32 epoch = 0; epoch < epochs; ++epoch) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);

            f32 totalError = 0;
            for (u32 i = 0; i < numSamples; i += batchSize) {
                u32 currentBatchSize = std::min(batchSize, numSamples - i);

                Input inputBatch(trainingData[0].size(), currentBatchSize);
                Output expectedBatch(expectedOutputs[0].size(), currentBatchSize);
                for (u32 j = 0; j < currentBatchSize; ++j) {
                    inputBatch.col(j) = trainingData[indices[i + j]];
                    expectedBatch.col(j) = expectedOutputs[indices[i + j]];
                }

                Output actual = run(inputBatch);

                totalError += std::visit([&](const auto& policy) {
                    return policy.calculate(actual, expectedBatch) * currentBatchSize;
                }, lossFunction);

                Output delta;
                std::visit([&](const auto& lastLayer) {
                    using LastLayerType = std::decay_t<decltype(lastLayer)>;
                    bool isSoftmaxWithCCE = std::holds_alternative<CategoricalCrossEntropyPolicy>(lossFunction) &&
                                            std::is_same_v<LastLayerType, Layer<SoftmaxPolicy>>;

                    Output loss_derivative = std::visit([&](const auto& policy) {
                        return policy.derivative(actual, expectedBatch);
                    }, lossFunction);

                    if (isSoftmaxWithCCE) {
                        delta = loss_derivative;
                    } else {
                        delta = loss_derivative.cwiseProduct(lastLayer.activationDerivative());
                    }
                }, _layers.back());

                for (i64 j = _layers.size() - 1; j >= 0; --j) {
                    // 1. Определяем вход для текущего слоя j (это выход слоя j-1)
                    const auto& prevLayerOutput = (j > 0) ?
                        std::visit([](auto& l){ return l.getLastOutput(); }, _layers[j-1]) :
                        inputBatch;

                    // 2. Вычисляем градиенты для весов и смещений слоя j, используя текущий delta
                    WeightMatrix weightGrad = delta * prevLayerOutput.transpose();
                    BiasVector biasGrad = delta.rowwise().mean();

                    // 3. Вычисляем delta для *следующей* итерации (для слоя j-1),
                    //    используя веса текущего слоя j.
                    if (j > 0) {
                        delta = std::visit([&](const auto& currentLayer) -> Output {
                            return (currentLayer.getWeights().transpose() * delta).cwiseProduct(
                                std::visit([](auto& prevLayer){ return prevLayer.activationDerivative(); }, _layers[j-1])
                            );
                        }, _layers[j]);
                    }

                    // 4. Обновляем веса и смещения для слоя j
                    std::visit([&](auto& layer) {
                        layer.getWeights() -= learningRate * weightGrad;
                        layer.getBiases() -= learningRate * biasGrad;
                    }, _layers[j]);
                }
            }
            if ((epoch + 1) % 10 == 0) {
                 Log::Logger().debug("Epoch {}/{}, Avg Error: {}", epoch + 1, epochs, totalError / numSamples);
            }
        }
    }
};

#endif
