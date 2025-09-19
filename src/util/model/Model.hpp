#ifndef MODEL_HPP
#define MODEL_HPP

#include <iostream>
#include <memory>
#include <ranges>

#include "model-parts/Network.hpp"
#include "Parser.hpp"
#include "Normalizer.hpp"
#include "../logging.hpp"
#include "model-parts/Metrics.hpp"

class Model {
    std::unique_ptr<Network> network = nullptr;
    std::vector<Input> trainingInputs{};
    std::vector<Output> trainingOutputs{};
    std::vector<Input> originalInputs{};
    std::vector<Output> originalOutputs{};

    std::vector<Normalizer> inputNormalizers{};
    std::optional<Normalizer> outputNormalizer{};

    u32 inputSize = 0;
    u32 outputSize = 0;
    bool isClassification = false;
    bool normalizationEnabled = false;

public:
    Model() = default;

    Model& fromCSV(const std::string& filepath, const std::vector<u32>& featureColumns, u32 targetColumn, bool hasHeader = true) {
        Log::Logger().info("--- 1. Loading data from {} ---", filepath);
        Parser parser(filepath, featureColumns, targetColumn, hasHeader);
        originalInputs = parser.getInputs();
        originalOutputs = parser.getOutputs();
        trainingInputs = originalInputs;
        trainingOutputs = originalOutputs;
        inputSize = parser.getInputSize();
        outputSize = parser.getOutputSize();
        isClassification = outputSize > 1;
        Log::Logger().info("Dataset loaded: {} samples.", originalInputs.size());
        Log::Logger().info("Input size: {}. Output size: {}.", inputSize, outputSize);
        Log::Logger().info("Task type: {}.\n", isClassification ? "Classification" : "Regression");
        return *this;
    }

    // --- МЕТОД NORMALIZE ИСПРАВЛЕН ---
    Model& normalize(bool enabled) {
        normalizationEnabled = enabled;
        if (normalizationEnabled) {
            Log::Logger().info("--- 2. Normalization enabled ---");

            // Подготовка нормализаторов для входов
            inputNormalizers.resize(inputSize);
            for (u64 i = 0; i < inputSize; ++i) {
                // Передаем originalInputs напрямую, без конвертации
                inputNormalizers[i].fit(originalInputs, i);
            }

            // Для регрессии нормализуем и выход
            if (!isClassification) {
                outputNormalizer.emplace();
                // Передаем originalOutputs напрямую
                outputNormalizer->fit(originalOutputs, 0);
            }
            Log::Logger().info("Normalizers fitted to data.\n");
        }
        return *this;
    }

    Model& withNetwork(const std::vector<std::pair<u32, PolicyType>>& layersConfig) {
        if (inputSize == 0) throw std::runtime_error("Data must be loaded before configuring the network.");
        Log::Logger().info("--- 3. Configuring network architecture ---");
        network = std::make_unique<Network>(inputSize, layersConfig);
        Log::Logger().info("Network created successfully.\n");
        return *this;
    }

    Model& train(u32 epochs, f32 learningRate, std::optional<LossType> lossTypeOpt = std::nullopt) {
        if (!network) {
            throw std::runtime_error("Network must be configured before training.");
        }
        Log::Logger().info("--- 4. Starting training ---");

        if (normalizationEnabled) {
            Log::Logger().info("Applying normalization to training data...");
            for (auto& row : trainingInputs) {
                for (u32 i = 0; i < row.size(); ++i) {
                    row(i) = inputNormalizers[i].transform(row(i));
                }
            }
            if (outputNormalizer.has_value() && !isClassification) {
                for (auto& row : trainingOutputs) {
                    row(0) = outputNormalizer->transform(row(0));
                }
            }
        }

        LossType lossType = lossTypeOpt.value_or(isClassification ? LossType::CATEGORICAL_CROSS_ENTROPY : LossType::MEAN_SQUARED_ERROR);

        AnyLossPolicy lossPolicy;
        if (lossType == LossType::CATEGORICAL_CROSS_ENTROPY) {
            lossPolicy = CategoricalCrossEntropyPolicy{};
            Log::Logger().info("Using Categorical Cross-Entropy loss function.");
        } else {
            lossPolicy = MeanSquaredErrorPolicy{};
            Log::Logger().info("Using Mean Squared Error loss function.");
        }

        network->train(trainingInputs, trainingOutputs, epochs, learningRate, lossPolicy);
        Log::Logger().info("Training complete.\n");

        return *this;
    }

    Output predict(const Input& rawInput) {
        if (!network) throw std::runtime_error("Network is not trained yet.");
        Input processedInput = rawInput;
        if (normalizationEnabled) {
            for (u32 i = 0; i < processedInput.size(); ++i) {
                processedInput(i) = inputNormalizers[i].transform(processedInput(i));
            }
        }
        Output normalizedResult = network->run(processedInput);
        if (normalizationEnabled && outputNormalizer.has_value() && !isClassification) {
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

        std::vector<Output> predictions;
        predictions.reserve(originalInputs.size());
        for (const auto& input : originalInputs) {
            predictions.push_back(predict(input));
        }

        if (isClassification) {
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
