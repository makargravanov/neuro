

#include "src/util/model/Model.hpp"
#include "src/model-parts/Network.hpp"
#include "src/web-server/Starter.hpp"

void printVector(const Eigen::VectorXf& vec) {
    std::print(std::cout, "[");
    for (Eigen::Index i = 0; i < vec.size(); ++i) {
        std::print(std::cout, "{:.2f}{}", vec(i), (i == vec.size() - 1 ? "" : ", "));
    }
    std::print(std::cout, "]");
}

void xorExample() {
    Network net(2, { {3, PolicyType::RELU}, {1, PolicyType::SIGMOID} });

    std::vector trainingData = {
        (Input(2) << 0, 0).finished(),
        (Input(2) << 0, 1).finished(),
        (Input(2) << 1, 0).finished(),
        (Input(2) << 1, 1).finished()
    };
    std::vector expectedOutputs = {
        (Output(1) << 0).finished(),
        (Output(1) << 1).finished(),
        (Output(1) << 1).finished(),
        (Output(1) << 0).finished()
    };

    net.train(trainingData, expectedOutputs, 10000, 0.2f);
    std::println(std::cout, "--- Results ---");
    for (const auto& input : trainingData) {
        Output result = net.run(input);
        // Доступ к элементам через ()
        std::println(std::cout, "Input: [{},{}], Output: {}", input(0), input(1), result(0));
    }
}

void irisExample() {
    try {
        Model iris;

        iris.fromCSV("iris.csv", {0, 1, 2, 3}, 4)
            .withNetwork({
                {8, PolicyType::RELU},
                {3, PolicyType::SIGMOID}
            })
            .train(500, 0.1f)
            .evaluate();

        std::println(std::cout, "--- Prediction Example ---");

        Input sample = (Input(4) << 5.1, 3.5, 1.4, 0.2).finished();
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

        Input newProduct = (Input(3) << 150, 80, 120).finished();
        std::print(std::cout, "Predicting for B/J/U: ");
        printVector(newProduct);
        std::println(std::cout, " -> Predicted kcal: {:.1f}", regressor.predict(newProduct)(0));

    } catch (const std::exception& e) {
        std::println(std::cout, "An error occurred: {}", e.what());
    }
}

i32 main() {
    Starter::run(8080, 4);
    return 0;
}