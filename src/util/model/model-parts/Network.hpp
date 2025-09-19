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
    explicit Network(u32 inputSize, const std::vector<std::pair<u32, PolicyType> > &layersConfig) {
        if (layersConfig.empty()) {
            throw std::invalid_argument("Layers config must not be empty");
        }

        u32 lastLayerSize = inputSize;
        for (const auto &config: layersConfig) {
            u32 layerSize = config.first;
            const PolicyType &activation = config.second;

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
        if constexpr (std::is_same_v<ComputePolicy, CpuEigenPolicy>) {
            if (_layers.empty()) throw std::invalid_argument("No layers provided");
            std::variant<Input, OclBuffer> temp_variant = input;

            for (auto& layer_variant : _layers) {
                std::visit([&](auto& concrete_layer) {
                    // Передаем владение temp_variant в слой
                    concrete_layer.activate_inplace(std::move(temp_variant), input.cols());
                    // Забираем владение результатом из слоя
                    temp_variant = std::move(concrete_layer.getLastOutput());
                }, layer_variant);
            }
            return std::get<Output>(temp_variant);
        } else {
            throw std::logic_error("Use run_gpu for GpuPolicy");
        }
    }

    OclBuffer run_gpu(OclBuffer input_buf, u32 batch_size) {
        if constexpr (std::is_same_v<ComputePolicy, GpuPolicy>) {
            if (_layers.empty()) throw std::invalid_argument("No layers provided");

            std::variant<Input, OclBuffer> temp_variant = std::move(input_buf);

            for (auto& layer_variant : _layers) {
                std::visit([&](auto& concrete_layer) {
                   // Передаем владение буфером в слой
                   concrete_layer.activate_inplace(std::move(temp_variant), batch_size);
                   // Забираем результат, чтобы передать его следующему слою
                   temp_variant = std::move(concrete_layer.getLastOutput());
               }, layer_variant);
            }
            return std::move(std::get<OclBuffer>(temp_variant));
        } else {
            throw std::logic_error("Use run for CpuPolicy");
        }
    }


    void train(const std::vector<Eigen::VectorXf> &trainingData, const std::vector<Eigen::VectorXf> &expectedOutputs,
               u32 epochs, u32 batchSize, f32 learningRate, const AnyLossPolicy &lossFunction) {
        if constexpr (std::is_same_v<ComputePolicy, CpuEigenPolicy>) {
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

                    totalError += std::visit([&](const auto &policy) {
                        return policy.calculate(actual, expectedBatch) * currentBatchSize;
                    }, lossFunction);

                    // --- Backpropagation ---
                    Output delta;
                    std::visit([&](const auto &lastLayer) {
                        using LastLayerType = std::decay_t<decltype(lastLayer)>;
                        bool isSoftmaxWithCCE = std::holds_alternative<CategoricalCrossEntropyPolicy>(lossFunction) &&
                                                std::is_same_v<LastLayerType, Layer<SoftmaxPolicy, ComputePolicy> >;

                        Output loss_derivative = std::visit([&](const auto &policy) {
                            return policy.derivative(actual, expectedBatch);
                        }, lossFunction);

                        if (isSoftmaxWithCCE) {
                            delta = loss_derivative; // Упрощенная производная
                        } else {
                            // Используем политику для поэлементного умножения
                            delta = ComputePolicy::elementwiseProduct(loss_derivative,
                                                                      lastLayer.activationDerivative());
                        }
                    }, _layers.back());

                    for (i64 j = _layers.size() - 1; j >= 0; --j) {
                        const auto &prevLayerOutput = (j > 0)
                                                          ? std::visit([](auto &l) -> const auto& { return l.getLastOutput(); },
                                                                       _layers[j - 1])
                                                          : inputBatch;

                        // Вычисляем градиенты через политику
                        WeightMatrix weightGrad = ComputePolicy::calculateWeightGradient(delta, std::get<Input>(prevLayerOutput));
                        BiasVector biasGrad = ComputePolicy::calculateBiasGradient(delta);

                        if (j > 0) {
                            // Вычисляем delta для предыдущего слоя через политику
                            delta = std::visit([&](const auto &currentLayer) -> Output {
                                auto prevActivationDerivative = std::visit(
                                    [](auto &prevLayer) { return prevLayer.activationDerivative(); }, _layers[j - 1]);
                                return ComputePolicy::calculateNextDelta(
                                    currentLayer.getWeights_cpu(), delta, std::get<Output>(prevActivationDerivative));
                            }, _layers[j]);
                        }

                        // Обновляем веса и смещения через политику
                        std::visit([&](auto &layer) {
                            ComputePolicy::updateWeights(layer.getWeights_cpu(), learningRate, weightGrad);
                            ComputePolicy::updateBiases(layer.getBiases_cpu(), learningRate, biasGrad);
                        }, _layers[j]);
                    }
                }
                if ((epoch + 1) % 10 == 0) {
                    Log::Logger().debug("Epoch {}/{}, Avg Error: {}", epoch + 1, epochs, totalError / numSamples);
                }
            }
        } else {
            if (trainingData.size() != expectedOutputs.size()) {
                throw std::invalid_argument("Training data and expected outputs must have the same size.");
            }

            const u32 numSamples = trainingData.size();
            const u32 inputSize = trainingData[0].size();
            const u32 outputSize = expectedOutputs[0].size();
            std::vector<u32> indices(numSamples);
            std::iota(indices.begin(), indices.end(), 0);

            for (u32 epoch = 0; epoch < epochs; ++epoch) {
                std::random_device rd;
                std::mt19937 shuffling_g(rd());
                std::ranges::shuffle(indices, shuffling_g);

                f32 totalError = 0;
                for (u32 i = 0; i < numSamples; i += batchSize) {
                    u32 currentBatchSize = std::min(batchSize, numSamples - i);

                    // 1. Подготовка батчей на CPU
                    Input inputBatch(inputSize, currentBatchSize);
                    Output expectedBatch(outputSize, currentBatchSize);
                    for (u32 j = 0; j < currentBatchSize; ++j) {
                        inputBatch.col(j) = trainingData[indices[i + j]];
                        expectedBatch.col(j) = expectedOutputs[indices[i + j]];
                    }

                    // 2. Копирование батчей на GPU
                    OclBuffer input_buf, expected_buf;
                    input_buf.create(inputBatch.size() * sizeof(f32));
                    expected_buf.create(expectedBatch.size() * sizeof(f32));
                    input_buf.write(inputBatch.data());
                    expected_buf.write(expectedBatch.data());

                    // 3. Прямое распространение на GPU
                    OclBuffer actual_buf = run_gpu(std::move(input_buf), currentBatchSize);

                    // 4. Вычисление ошибки и обратное распространение (часть на CPU, часть на GPU)
                    // Для простоты, расчет ошибки и ее производной оставим на CPU
                    Output actual_cpu(outputSize, currentBatchSize);
                    actual_buf.read(actual_cpu.data());

                    totalError += std::visit([&](const auto &policy) {
                        return policy.calculate(actual_cpu, expectedBatch) * currentBatchSize;
                    }, lossFunction);

                    Output loss_derivative_cpu = std::visit([&](const auto &policy) {
                        return policy.derivative(actual_cpu, expectedBatch);
                    }, lossFunction);

                    OclBuffer delta_buf;
                    delta_buf.create(loss_derivative_cpu.size() * sizeof(f32));
                    delta_buf.write(loss_derivative_cpu.data());

                    // Комбинированный шаг для Softmax+CCE
                    std::visit([&](const auto &lastLayer) {
                        using LastLayerType = std::decay_t<decltype(lastLayer)>;
                        // --- ИСПРАВЛЕНИЕ 2 ---
                        // Заменили ActivationPolicy на ActivationPolicyT
                        bool isSoftmaxWithCCE = std::holds_alternative<CategoricalCrossEntropyPolicy>(lossFunction) &&
                                                std::is_same_v<typename LastLayerType::ActivationPolicyT, SoftmaxPolicy>;
                        if (!isSoftmaxWithCCE) {
                            auto deriv_buf = std::visit([&](auto &l) {
                                return l.activationDerivative(currentBatchSize);
                            }, _layers.back());
                            delta_buf = ComputePolicy::elementwiseProduct(
                                delta_buf, std::get<OclBuffer>(deriv_buf), outputSize * currentBatchSize);
                        }
                    }, _layers.back());


                    // --- Backpropagation на GPU ---
                    for (i64 j = _layers.size() - 1; j >= 0; --j) {
                        auto &current_layer_variant = _layers[j];

                        std::visit([&](auto &current_layer) {
                            // --- ИСПРАВЛЕНИЕ 3 ---
                            // Получаем ссылку на variant, а не копируем его
                            const auto &prevLayerOutput_variant = (j > 0)
                                ? std::visit([](auto &l) -> const auto& { return l.getLastOutput(); }, _layers[j - 1])
                                : current_layer.getLastInput();

                            auto &prevLayerOutput_buf = std::get<OclBuffer>(prevLayerOutput_variant);

                            u32 M = current_layer.getNeuronCount();
                            u32 K = current_layer.getPrevNeuronCount();
                            u32 N = currentBatchSize;

                            OclBuffer weightGrad = ComputePolicy::calculateWeightGradient(
                                delta_buf, prevLayerOutput_buf, M, N, K);
                            OclBuffer biasGrad = ComputePolicy::calculateBiasGradient(delta_buf, M, N);

                            if (j > 0) {
                                auto prev_deriv_variant = std::visit([&](auto &l) { return l.activationDerivative(N); },
                                                                     _layers[j - 1]);
                                auto &prev_deriv_buf = std::get<OclBuffer>(prev_deriv_variant);
                                auto& weights_buf = current_layer.getWeights_gpu();
                                delta_buf = ComputePolicy::calculateNextDelta(weights_buf, delta_buf, prev_deriv_buf, M, K, N);
                            }

                            auto& weights_to_update = current_layer.getWeights_gpu();
                            auto& biases_to_update = current_layer.getBiases_gpu();
                            ComputePolicy::updateWeights(weights_to_update, learningRate, weightGrad);
                            ComputePolicy::updateBiases(biases_to_update, learningRate, biasGrad);
                        }, current_layer_variant);
                    }
                }
                if ((epoch + 1) % 10 == 0) {
                    Log::Logger().debug("Epoch {}/{}, Avg Error: {}", epoch + 1, epochs, totalError / numSamples);
                }
            }
        }
    }
};


#endif