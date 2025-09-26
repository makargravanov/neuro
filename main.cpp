

#include "src/util/model/Model.hpp"
#include "src/util/model/model-parts/Network.hpp"
#include "src/util/logging.hpp"
#include "src/util/model/model-parts/ComputePolicies.hpp"
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
        Model<CpuEigenPolicy> iris;

        iris.fromCSV("datasets/iris.csv", {0, 1, 2, 3}, 4)
            .withArchitecture({
                {LayerType::DENSE_RELU, DenseLayerConfig{8}},
                {LayerType::DENSE_SIGMOID, DenseLayerConfig{3}}
            })
            .train(1500, 0.1f, 10)
            .evaluate();

        std::println(std::cout, "--- Prediction Example ---");

        Eigen::VectorXf sample = (Eigen::VectorXf(4) << 5.1, 3.5, 1.4, 0.2).finished();
        Eigen::VectorXf prediction = iris.predict(sample);

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
        Model<CpuEigenPolicy> regressor;

        regressor.fromCSV("bju_calories_regression_with_names.csv", {1, 2, 3}, 4)
            .normalize(true)
            .withArchitecture({
                {LayerType::DENSE_LINEAR, DenseLayerConfig{1}}
            })
            .train(5000, 0.01f, 32)
            .evaluate();

        Eigen::VectorXf newProduct = (Eigen::VectorXf(3) << 150, 80, 120).finished();
        std::print(std::cout, "Predicting for B/J/U: ");
        printVector(newProduct);
        std::println(std::cout, " -> Predicted kcal: {:.1f}", regressor.predict(newProduct)(0));

    } catch (const std::exception& e) {
        std::println(std::cout, "An error occurred: {}", e.what());
    }
}

void runRideStatusPrediction() {
    Log::Logger().message("--- Starting Ride Booking Status Prediction Example (Dense Network) ---");
    try {
        Model<CpuEigenPolicy> rideModel;
        std::vector<u32> featureIndices(16);
        std::iota(featureIndices.begin(), featureIndices.end(), 0);

        rideModel
            .fromCSV("datasets/processed_rides.csv", featureIndices, 16)
            .normalize(true)
            .withArchitecture({
                {LayerType::DENSE_RELU,    DenseLayerConfig{24}},
                {LayerType::DENSE_RELU,    DenseLayerConfig{12}},
                {LayerType::DENSE_SOFTMAX, DenseLayerConfig{4}}
            })
            .train(400, 0.002f, 128)
            .evaluate();
    } catch (const std::exception& e) {
        Log::Logger().error("An error occurred during the prediction task: {}", e.what());
    }
}

void runLineDetectorCNN() {
    Log::Logger().message("--- Starting Line Detector Example (Convolutional Network) ---");
    try {
        std::vector<Eigen::VectorXf> inputs;
        std::vector<Eigen::VectorXf> outputs;
        const int imgSize = 5;
        const int numSamples = 100;

        for (int i = 0; i < numSamples; ++i) {
            Eigen::MatrixXf img = Eigen::MatrixXf::Zero(imgSize, imgSize);
            Eigen::VectorXf label(2);

            if (i % 2 == 0) {
                img.col(2).setOnes();
                label << 1.0f, 0.0f;
            } else {
                img.row(2).setOnes();
                label << 0.0f, 1.0f;
            }
            img += Eigen::MatrixXf::Random(imgSize, imgSize) * 0.1f;

            inputs.emplace_back(Eigen::Map<Eigen::VectorXf>(img.data(), img.size()));
            outputs.emplace_back(label);
        }

        Model<CpuEigenPolicy> cnn;
        cnn.fromVectors(inputs, outputs)
           .withInputShape(1, imgSize, imgSize) // 1 канал (Ч/Б), 5x5
           .withArchitecture({
               // 4 фильтра 3x3, ReLU. Выход: 4x3x3
               {LayerType::CONV2D_RELU, ConvLayerConfig{4, 3, 1, PaddingMode::VALID}},
               // Max Pooling: окно 2x2. Выход: 4x1x1
               {LayerType::MAX_POOL2D, PoolLayerConfig{2, 1}},
               // вектор размером 4*1*1 = 4
               {LayerType::FLATTEN, {}},
               {LayerType::DENSE_SOFTMAX, DenseLayerConfig{2}}
           })
           .train(400, 0.001f, 10)
           .evaluate();

        Log::Logger().info("--- CNN Prediction Example ---");
        Eigen::MatrixXf testImg = Eigen::MatrixXf::Zero(imgSize, imgSize);
        testImg.col(2).setOnes();
        Eigen::VectorXf testVec = Eigen::Map<Eigen::VectorXf>(testImg.data(), testImg.size());

        Eigen::VectorXf prediction = cnn.predict(testVec);
        std::print(std::cout, "Prediction for vertical line -> ");
        printVector(prediction);
        std::println(std::cout, " (Expected: [1.00, 0.00])");


    } catch (const std::exception& e) {
        Log::Logger().error("An error occurred during the CNN task: {}", e.what());
    }
}


int main() {
    Log::Platform::enableColors();
    Eigen::setNbThreads(std::thread::hardware_concurrency());
    Log::Logger().info("Eigen is configured to use up to {} threads.", Eigen::nbThreads());

    //runRideStatusPrediction(); // Проверка старого функционала с новым API
    runLineDetectorCNN();      // Проверка нового функционала CNN

    return 0;
}