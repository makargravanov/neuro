#ifndef MODEL_HPP
#define MODEL_HPP

#include <iostream>
#include <memory>
#include <ranges>

#include "model-parts/Network.hpp"
#include "Parser.hpp"
#include "Normalizer.hpp"
#include "../logging.hpp"
#include "model-parts/LayerConfig.hpp"
#include "model-parts/Metrics.hpp"

template<typename ActiveComputePolicy>
class Model {
    std::unique_ptr<Network<ActiveComputePolicy>> network = nullptr;
    std::vector<Eigen::VectorXf> trainingInputs{};
    std::vector<Eigen::VectorXf> trainingOutputs{};
    std::vector<Eigen::VectorXf> originalInputs{};
    std::vector<Eigen::VectorXf> originalOutputs{};

    std::vector<Normalizer> inputNormalizers{};
    std::optional<Normalizer> outputNormalizer{};

    // Размеры входа
    u32 _inputVectorSize = 0;
    u32 _inputChannels = 0;
    u32 _inputHeight = 0;
    u32 _inputWidth = 0;

    u32 _outputSize = 0;
    bool _isClassification = false;
    bool _normalizationEnabled = false;
    bool _isCnn = false;

public:
    Model() = default;

    Model& fromVectors(std::vector<Eigen::VectorXf> inputs, std::vector<Eigen::VectorXf> outputs) {
        Log::Logger().info("--- 1. Loading data from memory ---");
        originalInputs = std::move(inputs);
        originalOutputs = std::move(outputs);
        trainingInputs = originalInputs;
        trainingOutputs = originalOutputs;
        if (trainingInputs.empty()) throw std::runtime_error("Input data is empty.");
        _inputVectorSize = trainingInputs[0].size();
        _outputSize = trainingOutputs[0].size();
        _isClassification = _outputSize > 1;
        Log::Logger().info("Dataset loaded: {} samples.", originalInputs.size());
        Log::Logger().info("Input vector size: {}. Output size: {}.", _inputVectorSize, _outputSize);
        Log::Logger().info("Task type: {}.\n", _isClassification ? "Classification" : "Regression");
        return *this;
    }

    Model& fromCSV(const std::string& filepath, const std::vector<u32>& featureColumns, u32 targetColumn, bool hasHeader = true) {
        Log::Logger().info("--- 1. Loading data from {} ---", filepath);
        Parser parser(filepath, featureColumns, targetColumn, hasHeader);
        originalInputs = parser.getInputs();
        originalOutputs = parser.getOutputs();
        trainingInputs = originalInputs;
        trainingOutputs = originalOutputs;
        _inputVectorSize = parser.getInputSize();
        _outputSize = parser.getOutputSize();
        _isClassification = _outputSize > 1;
        Log::Logger().info("Dataset loaded: {} samples.", originalInputs.size());
        Log::Logger().info("Input vector size: {}. Output size: {}.", _inputVectorSize, _outputSize);
        Log::Logger().info("Task type: {}.\n", _isClassification ? "Classification" : "Regression");
        return *this;
    }

    // Новый метод для задания формы входа для CNN
    Model& withInputShape(u32 channels, u32 height, u32 width) {
        if (channels * height * width != _inputVectorSize) {
            throw std::invalid_argument("Input shape (C*H*W) does not match the size of the input vector from data.");
        }
        _inputChannels = channels;
        _inputHeight = height;
        _inputWidth = width;
        _isCnn = true;
        Log::Logger().info("Input shape for CNN set to: [{}, {}, {}].\n", channels, height, width);
        return *this;
    }

    Model& normalize(bool enabled) {
        _normalizationEnabled = enabled;
        if (_normalizationEnabled) {
            Log::Logger().info("--- 2. Normalization enabled ---");
            inputNormalizers.resize(_inputVectorSize);
            for (u64 i = 0; i < _inputVectorSize; ++i) {
                inputNormalizers[i].fit(originalInputs, i);
            }
            if (!_isClassification) {
                outputNormalizer.emplace();
                outputNormalizer->fit(originalOutputs, 0);
            }
            Log::Logger().info("Normalizers fitted to data.\n");
        }
        return *this;
    }

    // Новый метод для построения сети по конфигурации
    Model& withArchitecture(const ArchitectureConfig& archConfig) {
        if (_inputVectorSize == 0) throw std::runtime_error("Data must be loaded before configuring the network.");
        Log::Logger().info("--- 3. Configuring network architecture ---");

        network = std::make_unique<Network<ActiveComputePolicy>>();

        // Отслеживаем размеры по мере построения сети
        u32 currentChannels = _inputChannels;
        u32 currentHeight = _inputHeight;
        u32 currentWidth = _inputWidth;
        u32 lastDenseSize = _inputVectorSize;
        bool isAfterFlatten = false;

        for (const auto& layerConfig : archConfig) {
            switch (layerConfig.type) {
                case LayerType::CONV2D_RELU:
                case LayerType::CONV2D_LINEAR: {
                    if (!_isCnn) throw std::runtime_error("Convolutional layers can only be used if an input shape is specified.");
                    const auto& cfg = std::get<ConvLayerConfig>(layerConfig.config);
                    ConvConfig convCfg(currentChannels, cfg.outputChannels, cfg.kernelSize, cfg.stride, cfg.paddingMode);

                    if (layerConfig.type == LayerType::CONV2D_RELU) {
                        network->addLayer(ConvLayer<ReLUPolicy, ActiveComputePolicy>(currentHeight, currentWidth, convCfg));
                    } else {
                        network->addLayer(ConvLayer<LinearPolicy, ActiveComputePolicy>(currentHeight, currentWidth, convCfg));
                    }

                    // Обновляем размеры
                    if (cfg.paddingMode == PaddingMode::VALID) {
                        currentHeight = (currentHeight - cfg.kernelSize) / cfg.stride + 1;
                        currentWidth = (currentWidth - cfg.kernelSize) / cfg.stride + 1;
                    } else { // SAME
                        currentHeight = (currentHeight + cfg.stride - 1) / cfg.stride;
                        currentWidth = (currentWidth + cfg.stride - 1) / cfg.stride;
                    }
                    currentChannels = cfg.outputChannels;
                    break;
                }
                case LayerType::MAX_POOL2D: {
                    const auto& cfg = std::get<PoolLayerConfig>(layerConfig.config);
                    network->addLayer(PoolingLayer<ActiveComputePolicy>(PoolConfig{cfg.poolSize, cfg.stride}));
                    currentHeight = (currentHeight - cfg.poolSize) / cfg.stride + 1;
                    currentWidth = (currentWidth - cfg.poolSize) / cfg.stride + 1;
                    break;
                }
                case LayerType::FLATTEN: {
                    network->addLayer(FlattenLayer<ActiveComputePolicy>());
                    lastDenseSize = currentChannels * currentHeight * currentWidth;
                    isAfterFlatten = true;
                    break;
                }
                case LayerType::DENSE_RELU:
                case LayerType::DENSE_LINEAR:
                case LayerType::DENSE_SIGMOID:
                case LayerType::DENSE_SOFTMAX: {
                    const auto& cfg = std::get<DenseLayerConfig>(layerConfig.config);
                    if (layerConfig.type == LayerType::DENSE_RELU) {
                        network->addLayer(DenseLayer<ReLUPolicy, ActiveComputePolicy>(cfg.numberOfNeurons, lastDenseSize));
                    } else if (layerConfig.type == LayerType::DENSE_LINEAR) {
                        network->addLayer(DenseLayer<LinearPolicy, ActiveComputePolicy>(cfg.numberOfNeurons, lastDenseSize));
                    } else if (layerConfig.type == LayerType::DENSE_SIGMOID) {
                        network->addLayer(DenseLayer<SigmoidPolicy, ActiveComputePolicy>(cfg.numberOfNeurons, lastDenseSize));
                    } else { // SOFTMAX
                        network->addLayer(DenseLayer<SoftmaxPolicy, ActiveComputePolicy>(cfg.numberOfNeurons, lastDenseSize));
                    }
                    lastDenseSize = cfg.numberOfNeurons;
                    break;
                }
            }
        }
        Log::Logger().info("Network created successfully.\n");
        return *this;
    }


    Model& train(u32 epochs, f32 learningRate, u32 batchSize, std::optional<LossType> lossTypeOpt = std::nullopt) {
        if (!network) throw std::runtime_error("Network must be configured before training.");
        Log::Logger().info("--- 4. Starting training (batch size: {}) ---", batchSize);

        if (_normalizationEnabled) {
            Log::Logger().info("Applying normalization to training data...");
            for (auto& row : trainingInputs) {
                for (u32 i = 0; i < row.size(); ++i) {
                    row(i) = inputNormalizers[i].transform(row(i));
                }
            }
            if (outputNormalizer.has_value() && !_isClassification) {
                for (auto& row : trainingOutputs) {
                    row(0) = outputNormalizer->transform(row(0));
                }
            }
        }

        LossType lossType = lossTypeOpt.value_or(_isClassification ? LossType::CATEGORICAL_CROSS_ENTROPY : LossType::MEAN_SQUARED_ERROR);
        AnyLossPolicy lossPolicy;

        if (lossType == LossType::CATEGORICAL_CROSS_ENTROPY) {
            lossPolicy = CategoricalCrossEntropyPolicy{};
            Log::Logger().info("Using Categorical Cross-Entropy loss function.");
        } else {
            lossPolicy = MeanSquaredErrorPolicy{};
            Log::Logger().info("Using Mean Squared Error loss function.");
        }

        const u32 numSamples = trainingInputs.size();
        std::vector<u32> indices(numSamples);
        std::iota(indices.begin(), indices.end(), 0);

        for (u32 epoch = 0; epoch < epochs; ++epoch) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);

            f32 totalError = 0;
            for (u32 i = 0; i < numSamples; i += batchSize) {
                u32 currentBatchSize = std::min(batchSize, numSamples - i);

                InputType inputBatch;
                // Готовим батч в зависимости от типа сети
                if (_isCnn) {
                    Tensor4f batchTensor(currentBatchSize, _inputChannels, _inputHeight, _inputWidth);
                    for (u32 j = 0; j < currentBatchSize; ++j) {
                        // Преобразуем вектор в тензор
                        batchTensor.chip(j, 0) = Eigen::TensorMap<const Eigen::Tensor<const f32, 3, Eigen::RowMajor>> (
                            trainingInputs[indices[i + j]].data(),
                            _inputChannels, _inputHeight, _inputWidth
                        );
                    }
                    inputBatch = batchTensor;
                } else {
                    DenseInput batchMatrix(_inputVectorSize, currentBatchSize);
                    for (u32 j = 0; j < currentBatchSize; ++j) {
                        batchMatrix.col(j) = trainingInputs[indices[i + j]];
                    }
                    inputBatch = batchMatrix;
                }

                Output expectedBatch(_outputSize, currentBatchSize);
                for (u32 j = 0; j < currentBatchSize; ++j) {
                    expectedBatch.col(j) = trainingOutputs[indices[i + j]];
                }

                network->train(inputBatch, expectedBatch, learningRate, lossPolicy);

                // ... (вычисление ошибки пока уберем для простоты, т.к. train не возвращает loss)
            }
             if ((epoch + 1) % 10 == 0) {
                 Log::Logger().debug("Epoch {}/{} complete.", epoch + 1, epochs);
            }
        }
        Log::Logger().info("Training complete.\n");
        return *this;
    }

    Eigen::VectorXf predict(const Eigen::VectorXf& rawInput) {
        if (!network) throw std::runtime_error("Network is not trained yet.");
        Eigen::VectorXf processedInput = rawInput;
        if (_normalizationEnabled) {
            for (u32 i = 0; i < processedInput.size(); ++i) {
                processedInput(i) = inputNormalizers[i].transform(processedInput(i));
            }
        }

        InputType networkInput;
        if (_isCnn) {
            Tensor4f inputTensor(1, _inputChannels, _inputHeight, _inputWidth);
            inputTensor.chip(0, 0) = Eigen::TensorMap<const Eigen::Tensor<const f32, 3, Eigen::RowMajor>>(
                processedInput.data(), _inputChannels, _inputHeight, _inputWidth
            );
            networkInput = inputTensor;
        } else {
            DenseInput inputMatrix(_inputVectorSize, 1);
            inputMatrix.col(0) = processedInput;
            networkInput = inputMatrix;
        }

        OutputType resultVariant = network->run(networkInput);
        Eigen::VectorXf normalizedResult = std::get<DenseOutput>(resultVariant).col(0);

        if (_normalizationEnabled && outputNormalizer.has_value() && !_isClassification) {
            normalizedResult(0) = outputNormalizer->inverseTransform(normalizedResult(0));
        }
        return normalizedResult;
    }

    Model& evaluate() {
        Log::Logger().info("--- 5. Evaluating model performance ---");
        if (originalInputs.empty()) {
            Log::Logger().warning("No data to evaluate.");
            return *this;
        }

        std::vector<Eigen::VectorXf> predictions;
        predictions.reserve(originalInputs.size());
        for (const auto& input : originalInputs) {
            predictions.push_back(predict(input));
        }

        if (_isClassification) {
            auto metrics = MetricsService::calculateClassificationMetrics(predictions, originalOutputs);
            Log::Logger().info("Accuracy: {:.2f}% ({}/{} correct predictions)", metrics.accuracy, metrics.correctPredictions, metrics.totalSamples);
        } else { // Регрессия
            auto metrics = MetricsService::calculateRegressionMetrics(predictions, originalOutputs);
            Log::Logger().info("Mean Absolute Error (MAE): {:.4f}", metrics.meanAbsoluteError);
            Log::Logger().info("Root Mean Squared Error (RMSE): {:.4f}", metrics.rootMeanSquaredError);
            Log::Logger().info("Mean Absolute Percentage Error (MAPE): {:.2f}%", metrics.meanAbsolutePercentageError);
        }
        Log::Logger().info("");

        return *this;
    }
};
#endif
