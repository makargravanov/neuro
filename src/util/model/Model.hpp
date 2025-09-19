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
    std::vector<Eigen::VectorXf> trainingInputs{};
    std::vector<Eigen::VectorXf> trainingOutputs{};
    std::vector<Eigen::VectorXf> originalInputs{};
    std::vector<Eigen::VectorXf> originalOutputs{};

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

    Model& normalize(bool enabled) {
        normalizationEnabled = enabled;
        if (normalizationEnabled) {
            Log::Logger().info("--- 2. Normalization enabled ---");
            inputNormalizers.resize(inputSize);
            for (u64 i = 0; i < inputSize; ++i) {
                inputNormalizers[i].fit(originalInputs, i);
            }
            if (!isClassification) {
                outputNormalizer.emplace();
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

    Model& train(u32 epochs, f32 learningRate, u32 batchSize, std::optional<LossType> lossTypeOpt = std::nullopt) {
        if (!network) {
            throw std::runtime_error("Network must be configured before training.");
        }
        Log::Logger().info("--- 4. Starting training (batch size: {}) ---", batchSize);

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

        network->train(trainingInputs, trainingOutputs, epochs, batchSize, learningRate, lossPolicy);
        Log::Logger().info("Training complete.\n");

        return *this;
    }

    Eigen::VectorXf predict(const Eigen::VectorXf& rawInput) {
        if (!network) throw std::runtime_error("Network is not trained yet.");
        Eigen::VectorXf processedInput = rawInput;
        if (normalizationEnabled) {
            for (u32 i = 0; i < processedInput.size(); ++i) {
                processedInput(i) = inputNormalizers[i].transform(processedInput(i));
            }
        }
        Input batchInput(processedInput.size(), 1);
        batchInput.col(0) = processedInput;

        Output normalizedResultBatch = network->run(batchInput);

        Eigen::VectorXf normalizedResult = normalizedResultBatch.col(0);

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

        std::vector<Eigen::VectorXf> predictions;
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
