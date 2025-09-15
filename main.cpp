

#include "src/Model.hpp"
#include "src/model-parts/Network.hpp"

void printVector(const std::vector<f32>& vec) {
    std::print(std::cout, "[");
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::print(std::cout, "{:.2f}{}", vec[i], (i == vec.size() - 1 ? "" : ", "));
    }
    std::print(std::cout, "]");
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
    std::println(std::cout, "--- Results ---");
    for (const auto& input : trainingData) {
        Output result = net.run(input);
        std::println(std::cout, "Input: [{},{}], Output: {}",input.at(0), input.at(1), result.at(0));
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

        std::println(std::cout, "--- Prediction Example ---");
        Input sample = {5.1, 3.5, 1.4, 0.2}; // пример ириса сетоза
        Output prediction = iris.predict(sample);

        std::print(std::cout, "Input: ");
        printVector(sample);
        std::print(std::cout, " -> Predicted output: ");
        printVector(prediction);
        std::println(std::cout, " (Expected: [1.00, 0.00, 0.00])");

    } catch (const std::exception& e) {
        std::println(std::cout, "An error occurred: {}", e.what());
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
        std::print(std::cout, "Predicting for B/J/U: ");printVector(newProduct);
        std::println(std::cout, " -> Predicted kcal: {:.1f}", regressor.predict(newProduct)[0]);

    } catch (const std::exception& e) {
        std::println(std::cout, "An error occurred: {}", e.what());
    }
}

i32 main() {

    std::println(std::cout, "--- Running BJU Regression Example ---");
    bjuExample();

    return 0;
}