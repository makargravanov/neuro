#ifndef MODEL_HPP
#define MODEL_HPP

#include <iostream>
#include <memory>
#include <ranges>

#include "model-parts/Network.hpp"
#include "model-parts/Parser.hpp"
#include "util/Normalizer.hpp"

class Model {
    std::unique_ptr<Network> network = nullptr;
    std::vector<Input> trainingInputs{};
    std::vector<Output> trainingOutputs{};
    std::vector<Input> originalInputs{};
    std::vector<Output> originalOutputs{};

    std::vector<Normalizer> inputNormalizers{};
    std::optional<Normalizer> outputNormalizer{}; // optional, т.к. нужен только для регрессии

    u32 inputSize = 0;
    u32 outputSize = 0;
    bool isClassification = false;
    bool normalizationEnabled = false;

public:
    Model() = default;

    Model& fromCSV(const std::string& filepath, const std::vector<u32>& featureColumns, u32 targetColumn, bool hasHeader = true) {
        std::println(std::cout,"--- 1. Loading data from {} ---", filepath);
        Parser parser(filepath, featureColumns, targetColumn, hasHeader);

        // оригинальные копии данных
        originalInputs = parser.getInputs();
        originalOutputs = parser.getOutputs();
        // рабочие
        trainingInputs = originalInputs;
        trainingOutputs = originalOutputs;

        inputSize = parser.getInputSize();
        outputSize = parser.getOutputSize();

        // определяем тип задачи: если выходной вектор больше 1, это классификация (one-hot)
        isClassification = outputSize > 1;

        std::println(std::cout, "Dataset loaded: {} samples.", originalInputs.size());
        std::println(std::cout, "Input size: {}. Output size: {}.", inputSize, outputSize);
        std::println(std::cout, "Task type: {}.\n", isClassification ? "Classification" : "Regression");

        return *this;
    }

    // включение нормализации - опция
    Model& normalize(bool enabled) {
        normalizationEnabled = enabled;
        if (normalizationEnabled) {
            std::println(std::cout,"--- 2. Normalization enabled ---");
            // Подготовка нормализаторов
            inputNormalizers.resize(inputSize);
            for (u64 i = 0; i < inputSize; ++i) {
                inputNormalizers[i].fit(originalInputs, i);
            }
            // Для регрессии нормализуем и выход
            if (!isClassification) {
                outputNormalizer.emplace();
                outputNormalizer->fit(originalOutputs, 0);
            }
             std::println(std::cout,"Normalizers fitted to data.\n");
        }
        return *this;
    }

    // конфигурация архитектуры сети
    Model& withNetwork(const std::vector<std::pair<u32, PolicyType>>& layersConfig) {
        if (inputSize == 0) {
            throw std::runtime_error("Data must be loaded before configuring the network.");
        }
        std::println(std::cout,"--- 3. Configuring network architecture ---");
        network = std::make_unique<Network>(inputSize, layersConfig);
        std::println(std::cout,"Network created successfully.\n");

        return *this;
    }

    // обучение
    Model& train(u32 epochs, f32 learningRate) {
        if (!network) {
            throw std::runtime_error("Network must be configured before training.");
        }
        std::println(std::cout,"--- 4. Starting training ---");
        
        // нормализуем
        if (normalizationEnabled) {
            std::println(std::cout,"Applying normalization to training data...");
            for (auto& row : trainingInputs) {
                for (u32 i = 0; i < row.size(); ++i) {
                    row(i) = inputNormalizers[i].transform(row(i));
                }
            }
            if (outputNormalizer.has_value()) {
                for (auto& row : trainingOutputs) {
                    row(0) = outputNormalizer->transform(row(0));
                }
            }
        }

        network->train(trainingInputs, trainingOutputs, epochs, learningRate);
        std::println(std::cout,"Training complete.\n");

        return *this;
    }

    //предсказание на новых данных
    Output predict(const Input& rawInput) {
        if (!network) {
            throw std::runtime_error("Network is not trained yet.");
        }
        
        Input processedInput = rawInput;
        // нормализуем вход, если нужно
        if (normalizationEnabled) {
            for (u32 i = 0; i < processedInput.size(); ++i) {
                processedInput(i) = inputNormalizers[i].transform(processedInput(i));
            }
        }

        Output normalizedResult = network->run(processedInput);

        // денормализуем выход, если это была регрессия с нормализацией
        if (normalizationEnabled && outputNormalizer.has_value()) {
            normalizedResult(0) = outputNormalizer->inverseTransform(normalizedResult(0));
        }

        return normalizedResult;
    }

    // оценка качества на исходном датасете
    Model& evaluate() {
        std::println(std::cout,"--- 5. Evaluating model performance ---");
        if (isClassification) {
            u32 correctPredictions = 0;
            for (u32 i = 0; i < originalInputs.size(); ++i) {
                Output prediction = predict(originalInputs[i]);

                Eigen::Index predictedIndex, expectedIndex;
                prediction.maxCoeff(&predictedIndex);
                originalOutputs[i].maxCoeff(&expectedIndex);

                if (predictedIndex == expectedIndex) {
                    correctPredictions++;
                }
            }
            f32 accuracy = static_cast<f32>(correctPredictions) / originalInputs.size() * 100.0f;
            std::println(std::cout,"Accuracy: {:.2f}% ({}/{} correct predictions)", accuracy, correctPredictions, originalInputs.size());

        } else { // Регрессия
            f32 totalPercentageError = 0;
            for (u32 i = 0; i < originalInputs.size(); ++i) {
                Output prediction = predict(originalInputs[i]);
                f32 predictedValue = prediction(0);
                f32 expectedValue = originalOutputs[i](0);

                if (expectedValue != 0) {
                    f32 error = std::abs(predictedValue - expectedValue);
                    totalPercentageError += (error / expectedValue) * 100.0f;
                }
            }
            f32 meanPercentageError = totalPercentageError / originalInputs.size();
            std::println(std::cout,"Mean Absolute Percentage Error (MAPE): {:.2f}%", meanPercentageError);
        }
        std::println(std::cout,"");

        return *this;
    }
};
#endif
