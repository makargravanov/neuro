export module Network;

import Layer;
import types;
import std;

export class Network {
    std::vector<Layer> _layers{};
public:

    explicit Network(u32 inputSize, const std::vector<u32>& layersSizes) {
        if(layersSizes.size() > 0) {
            _layers.resize(layersSizes.size());
            for (u32 i = 0; i < layersSizes.size(); i++) {
                _layers.at(i) = Layer(layersSizes.at(i), inputSize);
                inputSize = layersSizes.at(i);
            }
        } else {
            throw std::invalid_argument("Layers size must be greater than 0");
        }
    }

    Output run(const Input& input){
        if (_layers.empty()) {
            throw std::invalid_argument("No layers provided");
        }
        if (input.empty()) {
            throw std::invalid_argument("No input provided");
        }
        Output result;
        Input temp(input);
        std::ranges::for_each(_layers, [&](Layer& layer) {
            result = layer.activate(temp);
            temp = result;
        });
        return result;
    };

    void train(const std::vector<Input>& trainingData, const std::vector<Output>& expectedOutputs, u32 epochs, f32 learningRate) {
        if (trainingData.size() != expectedOutputs.size()) {
            throw std::invalid_argument("Training data and expected outputs must have the same size.");
        }

        for (u32 epoch = 0; epoch < epochs; ++epoch) {
            f32 totalError = 0;
            for (u32 i = 0; i < trainingData.size(); ++i) {
                const auto& input = trainingData[i];
                const auto& expected = expectedOutputs[i];

                // прямое распространение
                Output actual = run(input);

                // СКО
                for(u32 j = 0; j < actual.size(); ++j) {
                    totalError += (expected[j] - actual[j]) * (expected[j] - actual[j]);
                }

                // обратное распространение
                backwardPass(expected);

                // обновление весов
                updateWeights(input, learningRate);
            }
            if (LOG_STATUS) {
                std::println("Epoch {}/{}, Error: {}" ,epoch + 1,epochs,totalError);
            }
        }
    }

private:
    void backwardPass(const Output& expectedOutput) {
        Layer& outputLayer = _layers.back();
        auto& outputNeurons = outputLayer.getNeurons();

        // дельты для выходного слоя
        for (u32 i = 0; i < outputNeurons.size(); ++i) {
            Neuron& neuron = outputNeurons[i];
            f32 error = expectedOutput[i] - neuron.getOutput();
            neuron.setDelta(error * neuron.activationDerivative());
        }

        // распространение ошибки на скрытые слои
        for (i64 i = _layers.size() - 2; i >= 0; --i) {
            Layer& hiddenLayer = _layers[i];
            Layer& nextLayer = _layers[i + 1];
            auto& hiddenNeurons = hiddenLayer.getNeurons();
            const auto& nextNeurons = nextLayer.getNeurons();

            for (u32 j = 0; j < hiddenNeurons.size(); ++j) {
                Neuron& neuron = hiddenNeurons[j];
                f32 error = 0.0f;
                for (const auto& nextNeuron : nextNeurons) {
                    error += nextNeuron.getWeights()[j] * nextNeuron.getDelta();
                }
                neuron.setDelta(error * neuron.activationDerivative());
            }
        }
    }

    void updateWeights(const Input& initialInput, f32 learningRate) {
        Input currentInput = initialInput;

        for (auto& layer : _layers) {
            auto& neurons = layer.getNeurons();
            for (auto& neuron : neurons) {
                neuron.updateWeights(currentInput, learningRate);
            }

            // выход текущего слоя становится входом для следующего
            Output layerOutput;
            layerOutput.reserve(neurons.size());
            for(const auto& neuron : neurons) {
                layerOutput.push_back(neuron.getOutput());
            }
            currentInput = layerOutput;
        }
    }
};
