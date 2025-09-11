import Normalizer;
import Network;
import Parser;
import Neuron;
import Model;
import types;
import std;

void printVector(const std::vector<f32>& vec) {
    std::print("[");
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::print("{:.2f}{}", vec[i], (i == vec.size() - 1 ? "" : ", "));
    }
    std::print("]");
}

void xorExample() {
    Network net(2, { {3, PolicyType::RELU}, {1, PolicyType::SIGMOID} });
    std::vector<Input> trainingData = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<Output> expectedOutputs = {
        {0}, {1}, {1}, {0}
    };
    net.train(trainingData, expectedOutputs, 10000, 0.2f);
    std::println("--- Results ---");
    for (const auto& input : trainingData) {
        Output result = net.run(input);
        std::println("Input: [{},{}], Output: {}",input.at(0), input.at(1), result.at(0));
    }
}

void irisExample() {
    try {
        Model irisClassifier;

        // Шаги 1-3: Загрузка и конфигурация
        irisClassifier.loadDataFromCsv("iris.csv", {0, 1, 2, 3}, 4);
        irisClassifier.configureNetwork({
            {8, PolicyType::RELU},
            {3, PolicyType::SIGMOID} // Выходной слой на 3 нейрона определился автоматически
        });

        // Шаг 4: Обучение
        irisClassifier.train(500, 0.1f);

        // Шаг 5: Оценка
        irisClassifier.evaluate();

        // Шаг 6: Пример предсказания
        std::println("--- Prediction Example ---");
        Input sample = {5.1, 3.5, 1.4, 0.2}; // Пример ириса сетоза
        Output prediction = irisClassifier.predict(sample);

        std::print("Input: ");
        printVector(sample);
        std::print(" -> Predicted output: ");
        printVector(prediction);
        std::println(" (Expected: [1.00, 0.00, 0.00])");

    } catch (const std::exception& e) {
        std::println("An error occurred: {}", e.what());
    }
}

void bjuExample() {
    try {
        Model calorieRegressor;

        // Шаг 1: Загрузка
        calorieRegressor.loadDataFromCsv("bju_calories_regression_with_names.csv", {1, 2, 3}, 4);
        // Шаг 2: Включение нормализации
        calorieRegressor.enableNormalization(true);
        // Шаг 3: Конфигурация сети
        calorieRegressor.configureNetwork({
           {1, PolicyType::LINEAR}
        });

        // Шаг 4: Обучение
        calorieRegressor.train(5000, 0.01f);
        // Шаг 5: Оценка
        calorieRegressor.evaluate();

        // Шаг 6: Пример предсказания
        std::println("--- Prediction Example ---");
        Input newProduct = {233, 1212, 332}; // 10г белка, 20г жира, 5г углеводов
        Output prediction = calorieRegressor.predict(newProduct);

        std::print("Predicting for B/J/U: ");
        printVector(newProduct);
        std::println(" -> Predicted kcal: {:.1f}", prediction[0]);
    } catch (const std::exception& e) {
        std::println("An error occurred: {}", e.what());
    }
}

i32 main() {
    std::println("--- Running BJU Regression Example ---");
    bjuExample();
    return 0;
}