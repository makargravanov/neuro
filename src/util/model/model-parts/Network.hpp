
#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <random>
#include <stdexcept>
#include <variant>

#include "ActivationPolicies.hpp"
#include "layers/DenseLayer.hpp"
#include "LossPolicies.hpp"
#include "../../types/eigen_types.hpp"
#include "../../logging.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/FlattenLayer.hpp"
#include "layers/PoolingLayer.hpp"


template<typename ComputePolicy>
class Network {

    using AnyLayer = std::variant<
        DenseLayer<SigmoidPolicy, ComputePolicy>,
        DenseLayer<LinearPolicy, ComputePolicy>,
        DenseLayer<ReLUPolicy, ComputePolicy>,
        DenseLayer<SoftmaxPolicy, ComputePolicy>,
        ConvLayer<ReLUPolicy, ComputePolicy>,
        ConvLayer<LinearPolicy, ComputePolicy>,
        PoolingLayer<ComputePolicy>,
        FlattenLayer<ComputePolicy>
    >;

    std::vector<AnyLayer> _layers{};
    // Хранилище выходов каждого слоя для использования в backpropagation
    std::vector<OutputType> _lastLayerOutputs{};
public:

    Network() = default;

    void addLayer(AnyLayer layer) {
        _layers.emplace_back(std::move(layer));
    }

    explicit Network(u32 inputSize, const std::vector<std::pair<u32, PolicyType>>& layersConfig) {
        if (layersConfig.empty()) {
            throw std::invalid_argument("Layers config must not be empty");
        }

        u32 lastLayerSize = inputSize;
        for (const auto& config : layersConfig) {
            u32 layerSize = config.first;
            const PolicyType& activation = config.second;

            if (activation == PolicyType::RELU) {
                _layers.emplace_back(DenseLayer<ReLUPolicy, ComputePolicy>(layerSize, lastLayerSize));
            } else if (activation == PolicyType::SIGMOID) {
                _layers.emplace_back(DenseLayer<SigmoidPolicy, ComputePolicy>(layerSize, lastLayerSize));
            } else if (activation == PolicyType::LINEAR) {
                _layers.emplace_back(DenseLayer<LinearPolicy, ComputePolicy>(layerSize, lastLayerSize));
            } else if (activation == PolicyType::SOFTMAX) {
                _layers.emplace_back(DenseLayer<SoftmaxPolicy, ComputePolicy>(layerSize, lastLayerSize));
            } else {
                throw std::invalid_argument("Unknown activation function");
            }
            lastLayerSize = layerSize;
        }
    }

    OutputType run(const InputType& input) {
        if (_layers.empty()) throw std::invalid_argument("Network has no layers.");

        _lastLayerOutputs.clear();
        _lastLayerOutputs.reserve(_layers.size());

        OutputType currentData = input;

        for (auto& layerVariant : _layers) {
            currentData = std::visit([&](auto& layer) -> OutputType {
                return std::visit([&](const auto& inputData) -> OutputType {
                    // Проверяем во время компиляции, может ли текущий слой обработать текущий тип данных
                    if constexpr (std::is_invocable_r_v<OutputType, decltype(&std::decay_t<decltype(layer)>::activate), decltype(layer), decltype(inputData)>) {
                        return layer.activate(inputData);
                    } else {
                        throw std::runtime_error("Layer sequence mismatch: A layer received an incompatible input type.");
                    }
                }, currentData);
            }, layerVariant);
            _lastLayerOutputs.push_back(currentData);
        }
        return currentData;
    }

    void train(const InputType& inputBatch, const Output& expectedBatch, f32 learningRate, const AnyLossPolicy& lossFunction) {
        // --- Прямое распространение ---
        OutputType actualOutputVariant = run(inputBatch);

        // --- Обратное распространение ---
        OutputType delta;

        // 1. Вычисление дельты для последнего слоя
        std::visit([&](const auto& lastLayer) {
            using LastLayerType = std::decay_t<decltype(lastLayer)>;

            if constexpr (
                std::is_same_v<LastLayerType, DenseLayer<SigmoidPolicy, ComputePolicy>> ||
                std::is_same_v<LastLayerType, DenseLayer<LinearPolicy, ComputePolicy>> ||
                std::is_same_v<LastLayerType, DenseLayer<ReLUPolicy, ComputePolicy>> ||
                std::is_same_v<LastLayerType, DenseLayer<SoftmaxPolicy, ComputePolicy>> ||
                std::is_same_v<LastLayerType, FlattenLayer<ComputePolicy>>
            ) {
                const auto& actualDenseOutput = std::get<DenseOutput>(actualOutputVariant);
                bool isSoftmaxWithCCE = std::holds_alternative<CategoricalCrossEntropyPolicy>(lossFunction) &&
                                        std::is_same_v<LastLayerType, DenseLayer<SoftmaxPolicy, ComputePolicy>>;

                Output lossDerivative = std::visit([&](const auto& policy) {
                    return policy.derivative(actualDenseOutput, expectedBatch);
                }, lossFunction);

                if (isSoftmaxWithCCE) {
                    delta = lossDerivative;
                } else {
                    delta = ComputePolicy::elementwiseProduct(lossDerivative, lastLayer.activationDerivative());
                }
            } else {
                throw std::runtime_error("The last layer must be a Dense or Flatten layer to be used with the specified loss functions.");
            }
        }, _layers.back());


        // 2. Распространение ошибки назад по слоям
        for (i64 j = _layers.size() - 1; j >= 0; --j) {
            auto& currentLayerVariant = _layers[j];
            const InputType& prevLayerOutputVariant = (j > 0) ? _lastLayerOutputs[j - 1] : inputBatch;

            delta = std::visit([&](auto& layer) -> OutputType {
                using LayerT = std::decay_t<decltype(layer)>;

                if constexpr (
                    std::is_same_v<LayerT, DenseLayer<SigmoidPolicy, ComputePolicy>> ||
                    std::is_same_v<LayerT, DenseLayer<LinearPolicy, ComputePolicy>> ||
                    std::is_same_v<LayerT, DenseLayer<ReLUPolicy, ComputePolicy>> ||
                    std::is_same_v<LayerT, DenseLayer<SoftmaxPolicy, ComputePolicy>>
                ) {
                    auto& d = std::get<DenseOutput>(delta);
                    const auto& prevOut = std::get<DenseOutput>(prevLayerOutputVariant);

                    WeightMatrix weightGrad = ComputePolicy::calculateWeightGradient(d, prevOut);
                    BiasVector biasGrad = ComputePolicy::calculateBiasGradient(d);
                    ComputePolicy::updateWeights(layer.getWeights(), learningRate, weightGrad);
                    ComputePolicy::updateBiases(layer.getBiases(), learningRate, biasGrad);

                    if (j > 0) {
                        OutputType prevActivationDerivativeVariant = std::visit([](auto& prevLayer) -> OutputType { return prevLayer.activationDerivative(); }, _layers[j-1]);
                        const auto& prevAD = std::get<DenseOutput>(prevActivationDerivativeVariant);
                        return ComputePolicy::calculateNextDelta(layer.getWeights(), d, prevAD);
                    }
                }
                else if constexpr (
                    std::is_same_v<LayerT, ConvLayer<ReLUPolicy, ComputePolicy>> ||
                    std::is_same_v<LayerT, ConvLayer<LinearPolicy, ComputePolicy>>
                ) {
                    auto& d = std::get<Tensor4f>(delta);
                    const auto& prevOut = std::get<Tensor4f>(prevLayerOutputVariant);
                    const auto& config = layer.getConfig();

                    KernelTensor kernelGrad = ComputePolicy::calculateKernelGradient(prevOut, d, config.stride, config.paddingMode);
                    BiasVector biasGrad = ComputePolicy::calculateBiasGradient(d);
                    ComputePolicy::updateWeights(layer.getKernels(), learningRate, kernelGrad);
                    ComputePolicy::updateBiases(layer.getBiases(), learningRate, biasGrad);

                    if (j > 0) {
                        OutputType prevActivationDerivativeVariant = std::visit([](auto& prevLayer) -> OutputType { return prevLayer.activationDerivative(); }, _layers[j-1]);
                        const auto& prevAD = std::get<Tensor4f>(prevActivationDerivativeVariant);
                        return ComputePolicy::calculateNextDeltaConv(layer.getKernels(), d, prevAD, config.stride, config.paddingMode);
                    }
                }
                else if constexpr (std::is_same_v<LayerT, PoolingLayer<ComputePolicy>>) {
                    auto& d = std::get<Tensor4f>(delta);
                    return ComputePolicy::maxPoolingBackward(d, layer.getMaxIndices(), layer.getLastInput().dimensions());
                }
                else if constexpr (std::is_same_v<LayerT, FlattenLayer<ComputePolicy>>) {
                    auto& d = std::get<DenseOutput>(delta);
                    return ComputePolicy::unflatten(d, layer.getLastInputShape());
                }

                return delta;
            }, currentLayerVariant);
        }
    }

};


#endif
