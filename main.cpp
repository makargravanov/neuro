import Network;
import Neuron;
import types;
import std;

i32 main() {
    Network net(2, {3, 1});

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

    return 0;
}