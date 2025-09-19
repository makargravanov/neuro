

#include "src/util/model/Model.hpp"
#include "src/util/model/model-parts/Network.hpp"
#include "src/util/logging.hpp"
#include "src/web-server/Starter.hpp"

import EnableColors;

void printVector(const Eigen::VectorXf& vec) {
    std::print(std::cout, "[");
    for (Eigen::Index i = 0; i < vec.size(); ++i) {
        std::print(std::cout, "{:.2f}{}", vec(i), (i == vec.size() - 1 ? "" : ", "));
    }
    std::print(std::cout, "]");
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

void runRideStatusPrediction() {
    Log::Logger().message("--- Starting Ride Booking Status Prediction Example ---");

    try {
        Model rideModel;
        std::vector<u32> feature_indices(16);
        std::iota(feature_indices.begin(), feature_indices.end(), 0);

        constexpr u32 target_index = 16;

        rideModel
            .fromCSV("processed_rides.csv", feature_indices, target_index)
            .normalize(true)
            .withNetwork({
                {24, PolicyType::RELU},
                {12, PolicyType::RELU},
                {4,  PolicyType::SOFTMAX}
            })
            .train(1000, 0.00001f)
            .evaluate();
    } catch (const std::exception& e) {
        Log::Logger().error("An error occurred during the prediction task: {}", e.what());
    }
}

i32 main() {
    Log::Platform::enableColors();
    runRideStatusPrediction();

    //Starter::run(8080, 4);
    return 0;
}