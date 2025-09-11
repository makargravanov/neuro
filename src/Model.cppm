export module Model;

import Network;
import Parser;
import Normalizer;
import Neuron;
import types;
import std;

export class Model {
    // --- Приватные поля ---
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

    void loadDataFromCsv(const std::string& filepath, const std::vector<u32>& featureColumns, u32 targetColumn, bool hasHeader = true) {
        std::println("--- 1. Loading data from {} ---", filepath);
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

        std::println("Dataset loaded: {} samples.", originalInputs.size());
        std::println("Input size: {}. Output size: {}.", inputSize, outputSize);
        std::println("Task type: {}.\n", isClassification ? "Classification" : "Regression");
    }

    // включение нормализации - опция
    void enableNormalization(bool enabled) {
        normalizationEnabled = enabled;
        if (normalizationEnabled) {
            std::println("--- 2. Normalization enabled ---");
            // Подготовка нормализаторов
            inputNormalizers.resize(inputSize);
            for (u32 i = 0; i < inputSize; ++i) {
                inputNormalizers[i].fit(originalInputs, i);
            }
            // Для регрессии нормализуем и выход
            if (!isClassification) {
                outputNormalizer.emplace();
                outputNormalizer->fit(originalOutputs, 0);
            }
             std::println("Normalizers fitted to data.\n");
        }
    }

    // конфигурация архитектуры сети
    void configureNetwork(const std::vector<std::pair<u32, PolicyType>>& layersConfig) {
        if (inputSize == 0) {
            throw std::runtime_error("Data must be loaded before configuring the network.");
        }
        std::println("--- 3. Configuring network architecture ---");
        network = std::make_unique<Network>(inputSize, layersConfig);
        std::println("Network created successfully.\n");
    }

    // обучение
    void train(u32 epochs, f32 learningRate) {
        if (!network) {
            throw std::runtime_error("Network must be configured before training.");
        }
        std::println("--- 4. Starting training ---");
        
        // нормализуем
        if (normalizationEnabled) {
            std::println("Applying normalization to training data...");
            for (auto& row : trainingInputs) {
                for (u32 i = 0; i < row.size(); ++i) {
                    row[i] = inputNormalizers[i].transform(row[i]);
                }
            }
            if (outputNormalizer.has_value()) {
                for (auto& row : trainingOutputs) {
                    row[0] = outputNormalizer->transform(row[0]);
                }
            }
        }

        network->train(trainingInputs, trainingOutputs, epochs, learningRate);
        std::println("Training complete.\n");
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
                processedInput[i] = inputNormalizers[i].transform(processedInput[i]);
            }
        }

        Output normalizedResult = network->run(processedInput);

        // денормализуем выход, если это была регрессия с нормализацией
        if (normalizationEnabled && outputNormalizer.has_value()) {
            normalizedResult[0] = outputNormalizer->inverseTransform(normalizedResult[0]);
        }

        return normalizedResult;
    }

    // оценка качества на исходном датасете
    void evaluate() {
        std::println("--- 5. Evaluating model performance ---");
        if (isClassification) {
            u32 correctPredictions = 0;
            for (u32 i = 0; i < originalInputs.size(); ++i) {
                Output prediction = predict(originalInputs[i]);
                
                // Находим индекс с максимальным значением (предсказанный класс)
                auto predictedIt = std::ranges::max_element(prediction);
                u32 predictedIndex = std::distance(prediction.begin(), predictedIt);

                auto expectedIt = std::ranges::max_element(originalOutputs[i]);
                u32 expectedIndex = std::distance(originalOutputs[i].begin(), expectedIt);

                if (predictedIndex == expectedIndex) {
                    correctPredictions++;
                }
            }
            f32 accuracy = static_cast<f32>(correctPredictions) / originalInputs.size() * 100.0f;
            std::println("Accuracy: {:.2f}% ({}/{} correct predictions)", accuracy, correctPredictions, originalInputs.size());

        } else { // Регрессия
            f32 totalPercentageError = 0;
            for (u32 i = 0; i < originalInputs.size(); ++i) {
                Output prediction = predict(originalInputs[i]);
                f32 predictedValue = prediction[0];
                f32 expectedValue = originalOutputs[i][0];

                if (expectedValue != 0) {
                    f32 error = std::abs(predictedValue - expectedValue);
                    totalPercentageError += (error / expectedValue) * 100.0f;
                }
            }
            f32 meanPercentageError = totalPercentageError / originalInputs.size();
            std::println("Mean Absolute Percentage Error (MAPE): {:.2f}%", meanPercentageError);
        }
        std::println("");
    }
};