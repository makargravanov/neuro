
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


template<typename ComputePolicy>
class Network {

    using AnyLayer = std::variant<
        Layer<SigmoidPolicy, ComputePolicy>,
        Layer<LinearPolicy, ComputePolicy>,
        Layer<ReLUPolicy, ComputePolicy>,
        Layer<SoftmaxPolicy, ComputePolicy>
    >;

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
                _layers.emplace_back(Layer<ReLUPolicy, ComputePolicy>(layerSize, lastLayerSize));
            } else if (activation == PolicyType::SIGMOID) {
                _layers.emplace_back(Layer<SigmoidPolicy, ComputePolicy>(layerSize, lastLayerSize));
            } else if (activation == PolicyType::LINEAR) {
                _layers.emplace_back(Layer<LinearPolicy, ComputePolicy>(layerSize, lastLayerSize));
            } else if (activation == PolicyType::SOFTMAX) {
                _layers.emplace_back(Layer<SoftmaxPolicy, ComputePolicy>(layerSize, lastLayerSize));
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
            std::mt19937 shuffling_g(rd());
            std::ranges::shuffle(indices, shuffling_g);

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

                // --- Backpropagation ---
                Output delta;
                std::visit([&](const auto& lastLayer) {
                    using LastLayerType = std::decay_t<decltype(lastLayer)>;
                    bool isSoftmaxWithCCE = std::holds_alternative<CategoricalCrossEntropyPolicy>(lossFunction) &&
                                            std::is_same_v<LastLayerType, Layer<SoftmaxPolicy, ComputePolicy>>;

                    Output loss_derivative = std::visit([&](const auto& policy) {
                        return policy.derivative(actual, expectedBatch);
                    }, lossFunction);

                    if (isSoftmaxWithCCE) {
                        delta = loss_derivative; // Упрощенная производная
                    } else {
                        // Используем политику для поэлементного умножения
                        delta = ComputePolicy::elementwiseProduct(loss_derivative, lastLayer.activationDerivative());
                    }
                }, _layers.back());

                for (i64 j = _layers.size() - 1; j >= 0; --j) {
                    const auto& prevLayerOutput = (j > 0) ?
                        std::visit([](auto& l){ return l.getLastOutput(); }, _layers[j-1]) :
                        inputBatch;

                    // Вычисляем градиенты через политику
                    WeightMatrix weightGrad = ComputePolicy::calculateWeightGradient(delta, prevLayerOutput);
                    BiasVector biasGrad = ComputePolicy::calculateBiasGradient(delta);

                    if (j > 0) {
                        // Вычисляем delta для предыдущего слоя через политику
                        delta = std::visit([&](const auto& currentLayer) -> Output {
                            auto prevActivationDerivative = std::visit([](auto& prevLayer){ return prevLayer.activationDerivative(); }, _layers[j-1]);
                            return ComputePolicy::calculateNextDelta(currentLayer.getWeights(), delta, prevActivationDerivative);
                        }, _layers[j]);
                    }

                    // Обновляем веса и смещения через политику
                    std::visit([&](auto& layer) {
                        ComputePolicy::updateWeights(layer.getWeights(), learningRate, weightGrad);
                        ComputePolicy::updateBiases(layer.getBiases(), learningRate, biasGrad);
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
