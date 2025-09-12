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
        Model iris;

        iris.fromCSV("iris.csv", {0, 1, 2, 3}, 4);
        iris.withNetwork({
            {8, PolicyType::RELU},
            {3, PolicyType::SIGMOID}
        });

        iris.train(500, 0.1f);

        iris.evaluate();

        std::println("--- Prediction Example ---");
        Input sample = {5.1, 3.5, 1.4, 0.2}; // пример ириса сетоза
        Output prediction = iris.predict(sample);

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
        Model regressor;

        regressor.fromCSV("bju_calories_regression_with_names.csv", {1, 2, 3}, 4)
            .normalize(true)
            .withNetwork({
                {1, PolicyType::LINEAR}
            })
            .train(5000, 0.01f)
            .evaluate();

        Input newProduct = {2323442, 1000, 1342340};
        std::print("Predicting for B/J/U: ");printVector(newProduct);
        std::println(" -> Predicted kcal: {:.1f}", regressor.predict(newProduct)[0]);

    } catch (const std::exception& e) {
        std::println("An error occurred: {}", e.what());
    }
}

void placementExample() {
    try {
        Model classifier;

        classifier.fromCSV("Placement.csv", {1, 2}, 3) // features: CGPA(1), Internships(2); target: Placed(3)
            .normalize(true) // Нормализация признаков очень рекомендуется
            .withNetwork({
                {20, PolicyType::RELU},
                {10, PolicyType::RELU},
                {2, PolicyType::SIGMOID} // Выходной слой из 2 нейронов (по числу классов)
            })
            .train(10000, 0.001f) // Эпох может потребоваться больше для обучения классификации
            .evaluate();

        // Предсказание для нового студента
        Input newStudent1 = {8.5, 2}; // Высокий CGPA, 2 стажировки
        Input newStudent2 = {6.0, 0}; // Низкий CGPA, 0 стажировок

        Output result1 = classifier.predict(newStudent1);
        Output result2 = classifier.predict(newStudent2);

        // Интерпретация: находим индекс элемента с максимальной вероятностью
        auto predictedClass1 = std::distance(result1.begin(), std::ranges::max_element(result1));
        auto predictedClass2 = std::distance(result2.begin(), std::ranges::max_element(result2));

        std::println("Student 1 prediction: {} (probabilities: {:.2f}%, {:.2f}%)",
                     predictedClass1 == 0 ? "Yes" : "No", result1[0]*100, result1[1]*100);
        std::println("Student 2 prediction: {} (probabilities: {:.2f}%, {:.2f}%)",
                     predictedClass2 == 0 ? "Yes" : "No", result2[0]*100, result2[1]*100);

    } catch (const std::exception& e) {
        std::println("An error occurred: {}", e.what());
    }
}

i32 main() {
    //std::println("--- Running BJU Regression Example ---");
    //bjuExample();

    placementExample();
    return 0;
}